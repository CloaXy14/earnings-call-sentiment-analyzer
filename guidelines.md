# Earnings Call Sentiment Analyzer — Guidelines

> Research tool. Outputs are not investment advice; validate out of sample before trading real capital.

## 1. Project overview

The system turns unstructured corporate text into trading signals:

1. **Collect** earnings call transcripts, 8-K earnings releases, investor presentations, and 10-Q/10-K MD&A text.
2. **Analyze** the text with financial NLP (FinBERT, lexicons, forward-looking-statement detection, theme extraction)
   to produce one feature row per earnings event.
3. **Validate** the features with an event study of abnormal returns from 5 trading days before (−5D)
   to 10 trading days after (+10D) the event.
4. **Trade**: convert the validated sentiment score into long/short signals and backtest them with costs.

The central question is whether sentiment measured at the call predicts abnormal returns **after** the call.
Only the [+1, +10] window answers that; the pre-event and day-0 windows show anticipation and immediate reaction.

```
EDGAR / licensed transcripts / prices
        │  src/ecsa/data
        ▼
documents.parquet, prices.parquet
        │  src/ecsa/nlp
        ▼
features.parquet  (one row per event)
        │  src/ecsa/events
        ▼
event_study.parquet  (AR −5..+10, CARs, statistics)
        │  src/ecsa/signals
        ▼
signals.parquet, backtest.parquet
        │  app/streamlit_app.py
        ▼
Dashboard
```

## 2. Environment setup

### Requirements
- Python 3.11
- Optional GPU: NVIDIA CUDA or Apple Silicon (MPS). CPU works for small universes.

### Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# PyTorch: pick the command for your platform from https://pytorch.org/get-started/locally/
pip install torch

pip install -r requirements.txt
```

`requirements.txt` contents:

| Purpose | Packages |
|---|---|
| ML / NLP | `torch`, `transformers`, `scikit-learn`, `pysbd` |
| Data | `pandas`, `numpy`, `pyarrow`, `scipy`, `statsmodels` |
| Collection | `requests`, `beautifulsoup4`, `lxml`, `yfinance` (prototyping only) |
| App | `streamlit`, `plotly` |
| Config / tests | `pydantic`, `pyyaml`, `python-dotenv`, `pytest` |
| Optional LLM | `openai` |

After the first working install, pin exact versions (`pip freeze > requirements.lock`, or use `uv`/`pip-tools`)
and install from the lock file in CI.

### Configuration
Copy `.env.example` to `.env` (never commit `.env`):

```bash
SEC_USER_AGENT="Your Name your.email@example.com"   # required by SEC EDGAR
OPENAI_API_KEY=""                                    # optional, only for the LLM backend
TRANSCRIPT_API_KEY=""                                # optional, licensed transcript vendor
```

Model ids and revisions, phrase lists, windows, thresholds, and costs live in `config/settings.yaml`.

### Run
```bash
pytest
python -m ecsa.pipeline all --config config/settings.yaml
streamlit run app/streamlit_app.py
```

The first FinBERT run downloads model weights from Hugging Face (about 400 MB).

## 3. Methodologies

### 3.1 Data collection

**Sources and compliance**

| Source | Content | Access |
|---|---|---|
| SEC EDGAR | 8-K Item 2.02 earnings releases (EX-99.x), filed presentations/transcripts, 10-Q/10-K MD&A | Free public API. Send a User-Agent with name and email; stay under 10 requests/second |
| Licensed transcript provider | Full call transcripts with speakers | Through a licensed API or exported files in `data/raw/transcripts/` |
| Seeking Alpha | Transcripts | Its terms prohibit scraping. Use only through a licensed arrangement |
| Price data | Daily OHLCV, adjusted close, benchmark | `yfinance` for prototyping; a licensed vendor for production research |

**EDGAR pipeline**
1. Map tickers to 10-digit CIKs (`company_tickers.json`).
2. Read the company's filing history (`data.sec.gov/submissions/CIK##########.json`, including paginated files).
3. Keep 8-Ks whose `items` include 2.02 and download their EX-99 exhibits; extract MD&A from 10-Q (Item 2) and 10-K (Item 7).
4. Cache every response on disk; retry 429/5xx with exponential backoff.

**Transcripts** are parsed into speaker turns, split into prepared remarks and Q&A, and tagged by role
(executive, analyst, operator). Q&A and analyst questions often carry more information than scripted remarks.

**Event timing** decides which trading day is day 0:

| Event time (ET) | Session | Day 0 |
|---|---|---|
| Before 09:30 | BMO | Event date |
| 09:30–16:00 | DURING | Event date |
| After 16:00, or unknown | AMC | Next trading day |

All timestamps are stored in UTC and converted to America/New_York for this logic (DST-aware).

### 3.2 NLP & sentiment analysis

**Preprocessing**
- Normalize text and remove operator housekeeping.
- Remove the safe-harbor disclaimer before analysis. It is boilerplate full of forward-looking language and would bias every FLS metric.
- Split into sentences with `pysbd`, keeping section, speaker, and role per sentence.

**FinBERT sentiment**
- Default model: `ProsusAI/finbert` (positive / negative / neutral), with the revision pinned in config. Read label order from `id2label`.
- Sentence-level inference in batches, truncated at 512 tokens.
- Sentence score: `net = p_positive − p_negative` in [−1, 1].
- Document scores: mean and length-weighted mean of `net`, positive/negative shares, computed for the whole call and separately
  for prepared remarks vs Q&A and executives vs analysts.

**Optional LLM backend.** A GPT-style model behind the same interface, with temperature 0 and JSON-schema output. It is cached
by text hash and its cost is logged. Use it for theme summaries or as a comparison, not as the only score: outputs can change
between model versions.

**Keyword lexicon**
- Base: Loughran-McDonald financial dictionary (positive, negative, uncertainty, litigious). Download it yourself and follow its license.
- Project phrase lists (configurable), matched as lemmatized multi-word phrases:
  - Positive: growth, expansion, record revenue, strong demand, margin expansion, raised guidance.
  - Negative: costs, margin pressure, uncertainty, headwinds, slowdown, lowered guidance, impairment.
- Negation within 3 tokens before a match flips polarity ("no margin pressure").
- Metrics: counts per 1,000 words; `lexicon_net = (pos − neg) / (pos + neg + 1)`; uncertainty per 1,000 words.

**Forward-looking statements (FLS)**
- Rule-based detector using future and modal cues (expect, anticipate, guidance, outlook, will, plan, target, next quarter, fiscal year)
  outside the removed disclaimer. An FLS classifier model can replace it through config.
- Metrics: FLS share of sentences, and **guidance tone** = mean FinBERT `net` of FLS sentences.

**Themes and key phrases.** TF-IDF on 1–3-grams with financial stop words, plus NMF topics across the corpus (scikit-learn).
Each document gets its top themes and its top 10 key phrases with supporting sentences.

**Sentiment surprise.** Tone relative to the company's own history:
`(score − mean of previous 4 events) / std of previous 4 events`, using prior events only. Companies have stable
"house styles"; a change in tone is more informative than its level.

### 3.3 Event study (−5D to +10D)

1. **Normal returns:** market model `r_i,t = α + β r_m,t` estimated by OLS over trading days [−250, −30] (minimum 120 observations).
   Fallback: market-adjusted returns `r_i − r_m`, flagged in the output.
2. **Abnormal returns:** `AR_t = r_i,t − (α̂ + β̂ r_m,t)` for t = −5…+10; `CAR(t1, t2) = Σ AR_t`.
3. **Windows**

   | Window | Meaning |
   |---|---|
   | [−5, −1] | Anticipation or leakage before the call |
   | [0, 0], [0, +1] | Immediate reaction (not tradeable from a signal formed at the call) |
   | [+1, +10] | Post-event drift: the test of predictive power |
   | [−5, +10] | Full picture for visualization |

4. **Statistics**
   - Average AR/CAR paths by sentiment quintile (formed within each calendar quarter) with 95% confidence intervals.
   - Cross-sectional t-test and Boehmer–Musumeci–Poulsen test per window.
   - Spearman IC between the score and CAR[+1, +10], overall and by quarter.
   - Regression: `CAR[+1,+10] ~ score + CAR[−5,−1] + CAR[0,+1] (+ earnings surprise)`, with standard errors clustered by day-0 date
     because events bunch in earnings season.
5. **Evidence threshold.** A useful score shows a monotonic spread in CAR[+1, +10] across quintiles, a positive out-of-sample IC that is
   stable across quarters, and significance after controlling for the day-0 reaction.

### 3.4 Trading signals

1. **Composite score:** equal-weighted average of expanding-window z-scores of FinBERT tone, sentiment surprise, lexicon net tone,
   guidance tone, and negative uncertainty. Weights may instead be learned with Ridge or logistic regression in walk-forward
   expanding windows, with a 10-trading-day purge/embargo between training and test sets.
2. **Ranking:** at each day-0 close, rank the new event against events from the trailing 90 calendar days
   (minimum 20 events; otherwise no trade).
3. **Rules:** long the top quintile, short the bottom quintile; enter at the day-0 close; exit at the day +10 close.
   Equal weight or inverse 60-day volatility; at most 5% gross per name; optional sector neutrality.
4. **Backtest:** daily returns from overlapping positions, costs of 10 bps per side by default plus optional borrow costs.
   Compare against SPY and a placebo with shuffled scores.
5. **Metrics:** annualized return and volatility, Sharpe, maximum drawdown, hit rate, turnover, IC and its t-stat,
   long-leg vs short-leg contribution.

**Bias checklist**
- Look-ahead: features and z-scores use only past data; tests perturb future data and assert signals do not change.
- Survivorship: build the universe as of each date, including delisted companies where data allows.
- Timing: AMC events start on the next trading day; never use the day-0 close-to-close return for an AMC signal.
- Data snooping: record how many score/threshold variants were tried and report out-of-sample results only.

## 4. Visualization guidelines (Streamlit)

**Principles**
- The app only reads processed Parquet files; heavy computation happens in the pipeline. Wrap loaders in `st.cache_data`.
- Every number shows its definition in a tooltip (`help=`), and every chart labels units (% returns, trading days).
- Use a diverging palette for sentiment: red for negative, grey for neutral, green for positive, applied consistently across components.

**Layout**
1. **Sidebar:** ticker, event (date and document type), score definition, bucket method.
2. **Header metrics (`st.metric`):** overall sentiment score (−1 to 1) with its historical percentile, sentiment surprise,
   guidance tone, and current signal (LONG / SHORT / NONE).
3. **Key extracted phrases:** two columns for top positive and negative phrases with counts. Each expands to its supporting sentences
   with the phrase highlighted, followed by themes and the list of forward-looking statements.
4. **Main chart — sentiment vs price movement (Plotly):**
   - x-axis: trading days −5 to +10; vertical line at day 0; shaded region for [+1, +10].
   - Primary y-axis: this event's cumulative raw return and CAR in %, both 0 at the day −6 close.
   - Overlay: average CAR paths of the high / neutral / low sentiment buckets with confidence bands, so the event is read against peers.
   - Secondary y-axis (or line color): the event's sentiment score as a labeled reference line, with its bucket in the legend.
5. **Call timeline chart:** rolling mean of sentence-level FinBERT `net` across the call, with the prepared remarks / Q&A boundary marked.
6. **Strategy tab:** equity curve vs SPY and placebo, metrics table, IC by quarter.

**Chart example**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def event_chart(days, car_pct, ret_pct, bucket_paths, sentiment):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_vrect(x0=1, x1=10, fillcolor="lightgrey", opacity=0.2, line_width=0)
    fig.add_vline(x=0, line_dash="dash")
    fig.add_trace(go.Scatter(x=days, y=ret_pct, name="Cumulative return (%)"))
    fig.add_trace(go.Scatter(x=days, y=car_pct, name="CAR (%)", line=dict(width=3)))
    for name, path in bucket_paths.items():
        fig.add_trace(go.Scatter(x=days, y=path, name=f"Avg CAR: {name}", line=dict(dash="dot")))
    fig.add_trace(
        go.Scatter(x=days, y=[sentiment] * len(days), name=f"Sentiment {sentiment:+.2f}", line=dict(color="green" if sentiment > 0 else "red")),
        secondary_y=True,
    )
    fig.update_xaxes(title="Trading days relative to earnings call", dtick=1)
    fig.update_yaxes(title="Return (%)", secondary_y=False)
    fig.update_yaxes(title="Sentiment score", range=[-1, 1], secondary_y=True)
    return fig
```

## 5. Conventions

- Returns are decimals internally; multiply by 100 only for display (`_pct` suffix).
- Every model id and revision, phrase list, window, and threshold lives in `config/settings.yaml`.
- Tests never call the network; NLP tests use stub models.
- New features must pass the look-ahead test before they enter `scoring.py`.

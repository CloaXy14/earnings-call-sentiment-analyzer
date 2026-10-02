"""Rebuild .ua/knowledge-graph.json and .ua/domain-graph.json for this project.

Structure (files, classes, functions, line ranges, imports, calls) is read from the code
with Python's AST, so it cannot drift from the source. Explanations, data artifacts,
concepts, layers, the tour and the domain view are written by hand below.

Run from the project root:  python3 .ua/build_graph.py
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = ROOT / ".ua"
NOW = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------- explanations

FILES: dict[str, tuple[str, str, list[str]]] = {
    # path: (plain-English summary, complexity, tags)
    "app/streamlit_app.py": (
        "Entry point for `streamlit run`. Loads config/settings.yaml and hands the processed-data "
        "folder to the dashboard. It does no analysis itself, so the page opens instantly.",
        "simple", ["entry-point", "ui"]),
    "src/ecsa/__init__.py": (
        "Package marker for `ecsa`. Records the project-wide unit rule: returns are decimals "
        "(0.05 = 5%) unless a name ends in `_pct`.", "simple", ["package"]),
    "src/ecsa/pipeline.py": (
        "The command-line conductor. `python -m ecsa.pipeline all` runs four stages in order: "
        "ingest (collect text and prices) → analyze (sentiment features) → event-study (how prices "
        "reacted) → signals (trades and backtest). Each stage reads the previous stage's Parquet "
        "files and writes its own, so stages can be re-run independently. `--demo` swaps in "
        "synthetic data and a toy scorer so everything runs offline.",
        "complex", ["entry-point", "orchestration", "cli"]),
    "src/ecsa/settings.py": (
        "One validated settings object built from config/settings.yaml: model ids and pinned "
        "revisions, keyword lists, event windows, ranking thresholds, costs. Secrets (SEC "
        "User-Agent, API keys) are read only from environment variables and are rejected if "
        "someone puts them in YAML.", "moderate", ["configuration", "security"]),
    "src/ecsa/schemas.py": (
        "The data contracts every stage shares: Document (a transcript or filing split into "
        "sections and speaker turns), PriceBar, EventFeatures (one row of sentiment numbers per "
        "earnings event) and Signal (a trade decision). Every record carries timezone-aware "
        "timestamps and an `available_at` time, which is how the system proves it never used "
        "information before it existed.", "complex", ["data-model", "no-look-ahead"]),
    "src/ecsa/storage.py": (
        "Small helpers for reading and writing Parquet files atomically (write to a temp file, "
        "then rename) and for converting timestamps to UTC.", "simple", ["storage"]),
    "src/ecsa/demo.py": (
        "Creates the offline demo world: 8 fictional companies (SYN00–SYN07), 12 earnings calls "
        "each, and 560 days of prices with a known relationship between tone and returns. Used by "
        "`--demo` and the tests. Not real market data.", "moderate", ["demo", "synthetic-data"]),
    "src/ecsa/dashboard.py": (
        "Everything you see in the Streamlit app. It only reads files in data/processed/ (no "
        "models, no downloads). Shows the event's sentiment scores, why it did or did not trade, "
        "the −5 to +10 day return chart against high/neutral/low sentiment peers, the phrases "
        "behind the score, tone across the call, and the strategy tab (equity curve vs SPY and "
        "placebo, metrics, information coefficient).", "complex", ["ui", "visualization"]),
    "src/ecsa/data/__init__.py": ("Package for Stage 1, data collection.", "simple", ["package"]),
    "src/ecsa/data/cache.py": (
        "The only place that talks to the internet. DiskCache stores responses on disk with an "
        "expiry time; HttpClient adds retries with backoff and rate limiting, so repeated runs "
        "don't re-download and SEC limits are respected.", "moderate", ["network", "caching"]),
    "src/ecsa/data/ratelimit.py": (
        "A token bucket that spaces out requests (default 8 per second, under the SEC's 10/s "
        "limit).", "simple", ["network"]),
    "src/ecsa/data/edgar.py": (
        "SEC EDGAR client. Maps a ticker to its CIK, reads the company's filing history, keeps "
        "8-K filings tagged Item 2.02 (earnings results) and downloads their EX-99 press-release "
        "exhibits. It can also pull the MD&A section from 10-Q/10-K filings, but returns nothing "
        "rather than guessing when the headings are ambiguous.", "complex", ["data-source", "sec"]),
    "src/ecsa/data/transcripts.py": (
        "Loads earnings-call transcripts you are licensed to use from data/raw/transcripts/ "
        "(JSON, CSV, or TXT with a .meta.json sidecar), splits them into prepared remarks and Q&A, "
        "and tags each speaker as executive, analyst or operator. Also decides whether a call was "
        "before the open, during, or after the close, which sets the event's day 0. A malformed "
        "file fails with its file name and row. The vendor-API source is a deliberate stub.",
        "complex", ["data-source", "parsing"]),
    "src/ecsa/data/prices.py": (
        "Daily price sources. LocalPriceSource reads a Parquet file; YFinancePriceSource downloads "
        "from Yahoo Finance for prototyping (cached, rate-limited). Also validates bars and turns "
        "adjusted closes into daily decimal returns without filling gaps.",
        "moderate", ["data-source", "prices"]),
    "src/ecsa/nlp/__init__.py": ("Package for Stage 2, text analysis.", "simple", ["package"]),
    "src/ecsa/nlp/cleaning.py": (
        "Tidies text (Unicode, whitespace, dash variants) and removes the legal safe-harbor "
        "disclaimer. The disclaimer is full of words like 'expect' and 'anticipate', so leaving it "
        "in would make every call look forward-looking. Removed text is kept for audit.",
        "moderate", ["nlp", "cleaning"]),
    "src/ecsa/nlp/segment.py": (
        "Splits cleaned text into sentences (pysbd) while remembering who said each one, their "
        "role, and whether it was in prepared remarks or Q&A.", "simple", ["nlp"]),
    "src/ecsa/nlp/finbert.py": (
        "Sentence sentiment. FinBertScorer runs the FinBERT financial model (pinned revision, "
        "batched, cached) and returns positive/negative/neutral probabilities; each sentence's "
        "score is positive − negative, from −1 to +1. ModelClient downloads only the pinned model "
        "files. DemoScorer is a clearly labelled toy used by the offline demo.",
        "complex", ["nlp", "model", "sentiment"]),
    "src/ecsa/nlp/llm.py": (
        "Optional alternative scorer that asks an OpenAI model for sentiment, themes and "
        "forward-looking statements as structured JSON (temperature 0, cost logged). Off by "
        "default.", "moderate", ["nlp", "optional", "llm"]),
    "src/ecsa/nlp/lexicon.py": (
        "Keyword tone. Matches the configured positive phrases (growth, expansion, raised "
        "guidance…) and negative phrases (costs, margin pressure, uncertainty…), plus the "
        "Loughran-McDonald financial dictionary if you supply it. Handles word forms and flips "
        "a match when a negation like 'no' or 'not' appears within three words ('no margin "
        "pressure').", "moderate", ["nlp", "lexicon"]),
    "src/ecsa/nlp/fls.py": (
        "Finds forward-looking statements (guidance, outlook, plans) using future and modal word "
        "cues, or an optional pinned classifier model. Their average sentiment becomes "
        "'guidance tone'.", "simple", ["nlp", "forward-looking"]),
    "src/ecsa/nlp/themes.py": (
        "Discovers recurring themes with TF-IDF and NMF topic modelling. The model is fitted only "
        "on documents published before the current call, so themes never leak the future.",
        "moderate", ["nlp", "themes", "no-look-ahead"]),
    "src/ecsa/nlp/features.py": (
        "Turns sentences into one feature row per earnings event: average and length-weighted "
        "sentiment, positive/negative shares, splits by section and speaker role, lexicon tone, "
        "uncertainty, guidance tone, and sentiment surprise (tone versus the company's previous "
        "four calls). Also outputs the evidence tables the dashboard shows.",
        "complex", ["nlp", "features"]),
    "src/ecsa/events/__init__.py": ("Package for Stage 3, the event study.", "simple", ["package"]),
    "src/ecsa/events/calendar.py": (
        "The trading calendar, built from the benchmark's actual trading days (so holidays and "
        "half-days are handled). Its day0() rule: calls before the open or during the session "
        "use that day; after-close or unknown-time calls use the next trading day.",
        "simple", ["event-study", "timing"]),
    "src/ecsa/events/market_model.py": (
        "Estimates what a stock 'normally' does given the market: an OLS regression of the "
        "stock's daily return on the benchmark's, fitted on days −250 to −30 before the event. "
        "Falls back to stock-minus-market when there is too little history.",
        "moderate", ["event-study", "statistics"]),
    "src/ecsa/events/study.py": (
        "The event study itself. For each event it computes abnormal returns (actual minus "
        "expected) for each day from −5 to +10, and cumulative abnormal returns (CAR) for the "
        "standard windows: pre-event [−5,−1], reaction [0] and [0,+1], post-event drift "
        "[+1,+10]. Only the drift window tests whether sentiment predicts returns you could "
        "trade.", "complex", ["event-study", "statistics"]),
    "src/ecsa/events/stats.py": (
        "Statistics over all events: groups events into sentiment quintiles and averages their "
        "return paths with 95% bands, runs t-tests and the BMP test per window, measures the "
        "information coefficient (rank correlation of score with later returns), and a drift "
        "regression. These look back at outcomes, so they are for evaluation only and never "
        "feed trading decisions.", "complex", ["event-study", "statistics", "evaluation"]),
    "src/ecsa/signals/__init__.py": ("Package for Stage 4, trading signals.", "simple", ["package"]),
    "src/ecsa/signals/scoring.py": (
        "Combines five measures into one composite score: length-weighted sentiment, sentiment "
        "surprise, lexicon tone, guidance tone, and (negatively) uncertainty. Each is turned into "
        "a z-score using only earlier events, then averaged with the configured weights.",
        "moderate", ["signals", "scoring", "no-look-ahead"]),
    "src/ecsa/signals/model.py": (
        "Optional learned weights. Walk-forward training only on events whose 10-day outcome had "
        "already finished, with a 10-trading-day gap (embargo) before each test date, so the "
        "model never sees the answers it is graded on.", "moderate", ["signals", "ml", "no-look-ahead"]),
    "src/ecsa/signals/portfolio.py": (
        "Turns scores into trades. At each event's day-0 close, it ranks the event against events "
        "from the past 90 days: top 20% → LONG, bottom 20% → SHORT, otherwise NONE. Needs at "
        "least 20 events to rank, caps each name at 5%, can be sector-neutral, and records the "
        "reason for every decision. Events whose features were not ready by the close are "
        "rejected, never backdated.", "complex", ["signals", "portfolio"]),
    "src/ecsa/signals/backtest.py": (
        "Simulates the strategy day by day: enter at the day-0 close, hold for 10 trading days, "
        "overlapping positions, 10 bps cost per side. The placebo function reruns it 100 times "
        "with shuffled scores; a real signal should clearly beat the placebo.",
        "complex", ["signals", "backtest"]),
    "src/ecsa/signals/metrics.py": (
        "Strategy scorecard: annualized return and volatility, Sharpe ratio, maximum drawdown, hit "
        "rate, turnover, long versus short contribution, and how many strategy variants were "
        "tried (to flag data snooping).", "moderate", ["signals", "evaluation"]),
}

SYMBOLS: dict[str, str] = {
    # "path:Name": plain-English summary (anything not listed falls back to its docstring)
    "app/streamlit_app.py:main": "Sets the page up, loads settings and calls dashboard.render().",
    "src/ecsa/pipeline.py:build_parser": "Defines the CLI: the four stages, `all`, and the --config, --demo, --tickers, --start and --end options.",
    "src/ecsa/pipeline.py:ingest": "Stage 1. Collects documents (demo fixtures, licensed transcript files, optional EDGAR) and daily prices for the tickers plus the benchmark, then saves them as Parquet.",
    "src/ecsa/pipeline.py:analyze": "Stage 2. Picks the scorer (FinBERT, LLM or demo), runs the NLP feature extraction and saves features plus evidence tables.",
    "src/ecsa/pipeline.py:event_study": "Stage 3. Computes abnormal returns per event and the summary statistics (window tests, quintile paths, IC, regression).",
    "src/ecsa/pipeline.py:signals": "Stage 4. Builds composite scores, makes trade decisions, backtests them, runs the placebo, and saves the scorecard.",
    "src/ecsa/pipeline.py:main": "Runs the requested stages in order, prints one JSON summary line per stage, and turns errors into a clear CLI message.",
    "src/ecsa/settings.py:Settings": "All configuration in one validated object. Relative paths resolve from the project root; unknown keys and secrets in YAML are rejected.",
    "src/ecsa/schemas.py:Document": "One transcript, press release or MD&A section, with its sections and speaker turns. `event_datetime` is when the call happened; `available_at` is when the full text could be read. They are kept separate on purpose.",
    "src/ecsa/schemas.py:EventFeatures": "The sentiment numbers for one earnings event. Validation rejects features that claim to be available before their inputs.",
    "src/ecsa/schemas.py:Signal": "One trading decision: side (LONG, SHORT or NONE), unsigned weight, entry and exit dates. Validation forbids using information after the signal time.",
    "src/ecsa/schemas.py:PriceBar": "One daily price bar with validated prices and volume.",
    "src/ecsa/schemas.py:document_id": "Builds a stable SHA-256 id from the source, URL and exact text, so the same document is never stored twice.",
    "src/ecsa/data/transcripts.py:session_timing": "Classifies a call as BMO (before 09:30 New York time), DURING, AMC (after 16:00) or UNKNOWN.",
    "src/ecsa/data/transcripts.py:parse_transcript": "Splits transcript text into speaker turns, finds where Q&A starts, and labels each speaker's role.",
    "src/ecsa/data/transcripts.py:LocalFileSource": "Reads licensed transcript files for one ticker and date range; malformed input fails with the file name and row number.",
    "src/ecsa/data/edgar.py:EdgarClient": "Talks to SEC EDGAR with your declared identity, at under 10 requests per second, with caching.",
    "src/ecsa/nlp/features.py:analyze_documents": "The main NLP function: cleans, segments and scores every document, then aggregates one feature row per event.",
    "src/ecsa/nlp/features.py:add_surprise": "Sentiment surprise: how unusual this call's tone is versus the company's previous four calls (a z-score). Needs at least two prior calls.",
    "src/ecsa/events/study.py:run_study": "For every event: find day 0, fit the market model on past data, then compute daily abnormal returns from −5 to +10 and the CAR windows.",
    "src/ecsa/events/market_model.py:fit_market_model": "Fits expected return = alpha + beta × market return on the estimation window, using only data available before the event.",
    "src/ecsa/events/calendar.py:TradingCalendar": "Lists real trading days and answers 'which day is day 0 for this call?'.",
    "src/ecsa/signals/portfolio.py:build_signals": "Ranks each new event against the trailing 90 days and returns LONG/SHORT/NONE with a reason, plus the list of rejected events.",
    "src/ecsa/signals/backtest.py:backtest": "Daily portfolio returns after costs from all open positions.",
    "src/ecsa/signals/backtest.py:placebo": "Shuffles scores among events on the same day and reruns the backtest, 100 times, as a 'luck' baseline.",
    "src/ecsa/signals/scoring.py:composite_scores": "Computes each event's composite score from past-only z-scores of the five measures.",
    "src/ecsa/dashboard.py:render": "Draws the whole page: sidebar filters, metric cards, signal explanation, and the two tabs.",
    "src/ecsa/dashboard.py:event_chart": "The main chart: this event's return and CAR from −5 to +10 days, peer-bucket averages with 95% bands, and its sentiment score on the right axis.",
    "src/ecsa/dashboard.py:signal_explanation": "One sentence explaining the event's trade decision, using the composite score and the recorded reason.",
}

# Processed and intermediate data files: (path, summary, writer stage, readers)
TABLES = [
    ("data/interim/documents.parquet", "All collected documents with sections, speakers and timestamps.", "ingest", ["analyze"]),
    ("data/interim/prices.parquet", "Daily prices and decimal returns for every ticker and the benchmark (SPY).", "ingest", ["event_study", "signals"]),
    ("data/processed/features.parquet", "One row of sentiment features per earnings event.", "analyze", ["event_study"]),
    ("data/processed/sentences.parquet", "Every scored sentence with speaker, section and sentiment; feeds the dashboard's tone-across-the-call chart.", "analyze", ["render"]),
    ("data/processed/phrases.parquet", "Matched positive/negative phrases with their sentences; feeds 'Words behind the score'.", "analyze", ["render"]),
    ("data/processed/removed_text.parquet", "Safe-harbor and boilerplate text removed before analysis, kept for audit.", "analyze", ["render"]),
    ("data/processed/event_study.parquet", "Per-event abnormal returns and CAR windows joined to the features.", "event_study", ["signals"]),
    ("data/processed/event_paths.parquet", "Day-by-day AR/CAR path of each event (−5…+10) for the chart.", "event_study", ["render"]),
    ("data/processed/bucket_paths.parquet", "Average return paths of high/neutral/low sentiment peers with 95% bands.", "event_study", ["render"]),
    ("data/processed/event_buckets.parquet", "Which sentiment quintile each event falls in.", "event_study", ["render"]),
    ("data/processed/window_tests.parquet", "t-test and BMP results for each return window.", "event_study", []),
    ("data/processed/information_coefficients.parquet", "Rank correlation of each score with post-event drift, overall and by quarter.", "event_study", []),
    ("data/processed/regression.parquet", "Drift regression controlling for anticipation and reaction.", "event_study", []),
    ("data/processed/scored_events.parquet", "Events with their composite score.", "signals", ["render"]),
    ("data/processed/signals.parquet", "Every trade decision (LONG/SHORT/NONE), weight, dates and reason.", "signals", ["render"]),
    ("data/processed/rejected_signals.parquet", "Events that could not be traded, with the reason.", "signals", []),
    ("data/processed/backtest.parquet", "Daily strategy returns after costs.", "signals", ["render"]),
    ("data/processed/placebo.parquet", "Average equity path of the shuffled-score placebo.", "signals", ["render"]),
    ("data/processed/strategy_metrics.parquet", "Strategy scorecard (Sharpe, drawdown, hit rate, IC…).", "signals", ["render"]),
    ("data/processed/strategy_ic.parquet", "Information coefficient of the composite score by quarter.", "signals", ["render"]),
    ("data/processed/run_metadata.parquet", "Run settings: demo flag, seed, model revision, costs, placebo count.", "signals", ["render"]),
]

CONCEPTS = [
    ("no-look-ahead", "No look-ahead (point-in-time data)",
     "The rule that protects every result: a decision may only use information that existed at "
     "that moment. Every record has an `available_at` time; z-scores, themes and models use only "
     "earlier events; late features are rejected, not backdated. Tests change future data and "
     "check that past signals stay the same.",
     ["src/ecsa/schemas.py", "src/ecsa/signals/scoring.py", "src/ecsa/nlp/themes.py",
      "src/ecsa/signals/model.py", "src/ecsa/signals/portfolio.py"]),
    ("day-zero", "Day 0 and call timing",
     "Day 0 is the first trading session that could react to the call. Before-open or "
     "in-session calls: that day. After-close or unknown-time calls: the next trading day. "
     "Positions enter at the day-0 close.",
     ["src/ecsa/data/transcripts.py", "src/ecsa/events/calendar.py"]),
    ("abnormal-return", "Abnormal return and CAR",
     "Abnormal return = what the stock did minus what the market model expected. CAR adds them "
     "up over a window. [−5,−1] shows anticipation, [0] and [0,+1] the immediate reaction, and "
     "[+1,+10] the post-call drift — the only window a signal formed at the call could trade.",
     ["src/ecsa/events/market_model.py", "src/ecsa/events/study.py"]),
    ("sentence-sentiment", "Sentence sentiment (−1 to +1)",
     "Each sentence gets probabilities for positive, negative and neutral from FinBERT. Its score "
     "is positive minus negative. Event scores are averages, overall and by section and speaker.",
     ["src/ecsa/nlp/finbert.py", "src/ecsa/nlp/features.py"]),
    ("sentiment-surprise", "Sentiment surprise",
     "Companies have a house style, so tone is compared with the same company's previous four "
     "calls. A big positive surprise means unusually upbeat for that company.",
     ["src/ecsa/nlp/features.py"]),
    ("composite-score", "Composite score",
     "The number that drives trades: a weighted average of past-only z-scores of sentiment, "
     "surprise, lexicon tone, guidance tone and (negative) uncertainty. The dashboard's sidebar "
     "score is for display; trades always use this composite.",
     ["src/ecsa/signals/scoring.py", "src/ecsa/signals/portfolio.py"]),
    ("placebo-test", "Placebo test",
     "The strategy is rerun 100 times with scores shuffled among same-day events. If the real "
     "strategy doesn't beat these, its results could be luck.",
     ["src/ecsa/signals/backtest.py"]),
    ("safe-harbor", "Safe-harbor removal",
     "The legal 'forward-looking statements' disclaimer read at the start of most calls is "
     "removed before analysis; otherwise it would inflate every forward-looking measure.",
     ["src/ecsa/nlp/cleaning.py", "src/ecsa/nlp/fls.py"]),
]

DOCS = {
    "README.md": "Start here: what the project does, how to run the offline demo, how to use real data, and its limitations.",
    "guidelines.md": "The design document: the methodology for data collection, NLP, the event study and trading signals, plus the dashboard specification.",
    "change-report.md": "What was built and verified: test, lint, type, build and browser checks, and the bugs fixed.",
    "HANDOFF.md": "Checkpoint used to hand the task between AI assistants; currently marked complete.",
}
CONFIGS = {
    "config/settings.yaml": ("All tunable parameters in one place: model ids and revisions, keyword lists, event windows, ranking thresholds, costs, placebo count.", "src/ecsa/settings.py"),
    ".env.example": ("Template for secrets (SEC_USER_AGENT, optional API keys). Copy to .env, fill in, and export; never committed.", "src/ecsa/settings.py"),
    "pyproject.toml": ("Package definition, dependencies, and tool settings for pytest (network disabled), ruff and strict mypy.", "src/ecsa/__init__.py"),
    "Makefile": ("Shortcuts: `make demo-data`, `make app`, `make test`, `make check`.", "src/ecsa/pipeline.py"),
}

LAYERS = [
    ("foundations", "1 · Foundations", "Shared contracts every stage relies on: data records, settings and Parquet storage.",
     ["src/ecsa/__init__.py", "src/ecsa/schemas.py", "src/ecsa/settings.py", "src/ecsa/storage.py"]),
    ("collection", "2 · Data collection (ingest)", "Gets earnings text and prices: licensed transcripts, SEC EDGAR filings, price sources, with caching and rate limiting.",
     ["src/ecsa/data/"]),
    ("nlp", "3 · NLP & sentiment (analyze)", "Turns text into numbers: cleaning, sentences, FinBERT/LLM sentiment, keyword tone, forward-looking statements, themes, event features.",
     ["src/ecsa/nlp/"]),
    ("events", "4 · Event study (event-study)", "Measures how prices moved around each call (−5 to +10 days) and whether sentiment explains the move.",
     ["src/ecsa/events/"]),
    ("signals", "5 · Trading signals & backtest (signals)", "Turns sentiment into LONG/SHORT decisions and tests them honestly with costs and a placebo.",
     ["src/ecsa/signals/"]),
    ("app", "6 · Orchestration & dashboard", "The CLI that runs the stages, the synthetic demo, and the Streamlit dashboard.",
     ["src/ecsa/pipeline.py", "src/ecsa/demo.py", "src/ecsa/dashboard.py", "app/"]),
    ("data", "7 · Data files", "The Parquet tables each stage writes and the next stage (or the dashboard) reads.", []),
    ("concepts", "8 · Key ideas", "Short explanations of the ideas you need to read the code and the results.", []),
    ("docs", "9 · Docs & configuration", "Project documents and configuration files.", []),
]

TOUR = [
    ("What this project does", "The system reads earnings-call text, scores its tone, checks whether tone explains the stock's move over the following days, and turns that into long/short trades. Start with the README for the big picture and the guidelines for the method.",
     ["document:README.md", "document:guidelines.md"], None),
    ("How a run flows", "pipeline.py runs four stages: ingest → analyze → event-study → signals. Each stage writes Parquet files the next stage reads (see the Data files layer). `--demo` runs it all offline on synthetic data.",
     ["file:src/ecsa/pipeline.py", "function:src/ecsa/pipeline.py:main", "config:config/settings.yaml"], None),
    ("The rule behind everything: no look-ahead", "Every record says when it became available. Read the Document and Signal schemas, then the concept note. Most design choices later in the tour exist to enforce this rule.",
     ["file:src/ecsa/schemas.py", "concept:no-look-ahead"],
     "Pydantic models validate data when it is created: wrong types, naive timestamps or impossible availability times raise errors immediately instead of silently corrupting results."),
    ("Stage 1 — collecting text and prices", "Transcripts come from licensed local files (Seeking Alpha is not scraped), filings from SEC EDGAR, prices from a local file or yfinance. All network access goes through one cached, rate-limited client.",
     ["file:src/ecsa/data/transcripts.py", "file:src/ecsa/data/edgar.py", "file:src/ecsa/data/prices.py", "file:src/ecsa/data/cache.py"], None),
    ("Day 0: when could the market react?", "A call after the 16:00 close can only move prices the next trading day. The calendar turns call times into day 0, which anchors the event study and the trade entry.",
     ["concept:day-zero", "class:src/ecsa/events/calendar.py:TradingCalendar"], None),
    ("Stage 2 — from words to sentiment", "Text is cleaned (safe-harbor disclaimer removed), split into sentences, scored by FinBERT, and matched against keyword lists. features.py combines everything into one row per event, including sentiment surprise.",
     ["file:src/ecsa/nlp/cleaning.py", "file:src/ecsa/nlp/finbert.py", "file:src/ecsa/nlp/lexicon.py", "file:src/ecsa/nlp/features.py", "concept:sentence-sentiment", "concept:sentiment-surprise"], None),
    ("Stage 3 — did the price react?", "The market model estimates a normal return; the difference is the abnormal return. Summing it over windows gives CAR. stats.py tests, across all events, whether high-sentiment calls drift up after the call.",
     ["file:src/ecsa/events/market_model.py", "file:src/ecsa/events/study.py", "file:src/ecsa/events/stats.py", "concept:abnormal-return"],
     "statsmodels OLS is used for the market model and for the drift regression; standard errors are clustered by date because many companies report on the same days."),
    ("Stage 4 — turning tone into trades", "scoring.py builds the composite score from past-only z-scores; portfolio.py ranks each event against the last 90 days to decide LONG, SHORT or NONE and records why.",
     ["file:src/ecsa/signals/scoring.py", "file:src/ecsa/signals/portfolio.py", "concept:composite-score"], None),
    ("Testing the strategy honestly", "The backtest charges costs and holds 10 days; the placebo reruns it with shuffled scores 100 times; metrics.py reports Sharpe, drawdown, hit rate, IC and the number of variants tried.",
     ["file:src/ecsa/signals/backtest.py", "file:src/ecsa/signals/metrics.py", "concept:placebo-test"], None),
    ("Seeing the results", "The dashboard only reads the processed tables. Pick a ticker and event to see its scores, the reason for its trade decision, the −5 to +10 day chart against peers, and the phrases behind the score.",
     ["file:app/streamlit_app.py", "file:src/ecsa/dashboard.py", "table:data/processed/signals.parquet"],
     "Streamlit reruns the script on every interaction; @st.cache_data keeps the Parquet reads from repeating."),
]

# Business view (domain graph): domain -> flows -> steps (name, summary, file)
DOMAINS = [
    ("text-acquisition", "Earnings text acquisition",
     "Getting the words management said, with exact timing and licensing respected.",
     ["Document", "Section", "Utterance"],
     ["Only licensed transcripts; no scraping of sites that forbid it", "SEC requests declare an identity and stay under 10 per second", "Call time and text availability are recorded separately"],
     [("collect-documents", "Collect documents for an earnings event", "python -m ecsa.pipeline ingest", "cli", [
         ("Load licensed transcripts", "Read JSON/CSV/TXT files from data/raw/transcripts/.", "src/ecsa/data/transcripts.py"),
         ("Parse speakers and Q&A", "Split into speaker turns, mark prepared remarks vs Q&A, tag roles.", "src/ecsa/data/transcripts.py"),
         ("Fetch SEC earnings releases", "Optional: 8-K Item 2.02 EX-99 exhibits and MD&A from EDGAR.", "src/ecsa/data/edgar.py"),
         ("Classify call timing", "Before open, during, after close, or unknown.", "src/ecsa/data/transcripts.py"),
         ("Store documents", "Save validated documents to Parquet.", "src/ecsa/storage.py")]),
      ("collect-prices", "Collect daily prices", "python -m ecsa.pipeline ingest", "cli", [
         ("Download or load bars", "Local Parquet or yfinance (prototype only), cached.", "src/ecsa/data/prices.py"),
         ("Validate and compute returns", "Check bars; compute decimal returns without filling gaps.", "src/ecsa/data/prices.py")])]),
    ("language-analysis", "Language & sentiment analysis",
     "Measuring how positive, uncertain and forward-looking management sounded.",
     ["EventFeatures", "Sentence", "PhraseMatch"],
     ["Remove the safe-harbor disclaimer before measuring forward-looking language", "Model revisions are pinned", "Themes and surprise use only earlier calls"],
     [("score-call", "Score one earnings call", "python -m ecsa.pipeline analyze", "cli", [
         ("Clean text", "Normalize text and remove boilerplate disclaimers.", "src/ecsa/nlp/cleaning.py"),
         ("Split into sentences", "Keep speaker, role and section for every sentence.", "src/ecsa/nlp/segment.py"),
         ("Score sentiment", "FinBERT (or optional LLM) per sentence, −1 to +1.", "src/ecsa/nlp/finbert.py"),
         ("Match keywords", "Positive/negative phrases with negation handling.", "src/ecsa/nlp/lexicon.py"),
         ("Find forward-looking statements", "Guidance and outlook sentences and their tone.", "src/ecsa/nlp/fls.py"),
         ("Build event features", "Aggregate to one row per event, add sentiment surprise.", "src/ecsa/nlp/features.py")])]),
    ("market-reaction", "Market reaction measurement",
     "Measuring how the stock moved relative to the market around each call.",
     ["Abnormal return", "CAR window"],
     ["Day 0 is the first session that could react", "Normal returns are estimated on days −250 to −30 only", "Only [+1,+10] tests tradeable prediction"],
     [("event-study", "Run the event study", "python -m ecsa.pipeline event-study", "cli", [
         ("Find day 0", "Map call time to the reacting trading day.", "src/ecsa/events/calendar.py"),
         ("Fit market model", "Expected return from the market on past data.", "src/ecsa/events/market_model.py"),
         ("Compute abnormal returns", "Daily AR from −5 to +10 and CAR windows.", "src/ecsa/events/study.py"),
         ("Test across events", "Quintile paths, t/BMP tests, IC, regression.", "src/ecsa/events/stats.py")])]),
    ("trading-strategy", "Trading strategy",
     "Turning sentiment into positions and checking whether they would have made money.",
     ["Signal", "Portfolio", "Backtest"],
     ["Enter at the day-0 close, hold 10 trading days", "Rank only against the past 90 days; at least 20 events", "Max 5% per name; 10 bps per side", "Compare with 100 placebo runs"],
     [("generate-trades", "Generate and test trades", "python -m ecsa.pipeline signals", "cli", [
         ("Composite score", "Past-only z-scores combined with configured weights.", "src/ecsa/signals/scoring.py"),
         ("Rank and decide", "LONG top 20%, SHORT bottom 20%, NONE otherwise, with reasons.", "src/ecsa/signals/portfolio.py"),
         ("Backtest with costs", "Daily returns of overlapping 10-day positions.", "src/ecsa/signals/backtest.py"),
         ("Placebo comparison", "100 shuffled-score reruns.", "src/ecsa/signals/backtest.py"),
         ("Score the strategy", "Sharpe, drawdown, hit rate, IC, variants tried.", "src/ecsa/signals/metrics.py")])]),
    ("research-presentation", "Research presentation",
     "Showing a researcher what changed in a call, how the stock reacted, and how the strategy performed.",
     ["Processed tables"],
     ["The dashboard reads processed files only; no models or downloads", "Demo results are labelled as synthetic"],
     [("explore-event", "Explore an earnings event", "streamlit run app/streamlit_app.py", "manual", [
         ("Choose ticker and event", "Sidebar filters and score definition.", "src/ecsa/dashboard.py"),
         ("Read the scores and decision", "Metric cards and the signal explanation.", "src/ecsa/dashboard.py"),
         ("Compare the price path with peers", "The −5 to +10 day chart.", "src/ecsa/dashboard.py"),
         ("Inspect the evidence", "Phrases, themes, forward-looking statements, tone across the call.", "src/ecsa/dashboard.py"),
         ("Review the strategy", "Equity curve vs SPY and placebo, metrics, IC.", "src/ecsa/dashboard.py")])]),
]
CROSS_DOMAIN = [
    ("text-acquisition", "language-analysis", "Documents with timing feed the sentiment analysis."),
    ("language-analysis", "market-reaction", "Event features are joined to abnormal returns."),
    ("market-reaction", "trading-strategy", "Event-study rows supply scores and the calendar for trades."),
    ("trading-strategy", "research-presentation", "Signals, backtest and metrics are displayed."),
    ("market-reaction", "research-presentation", "Return paths and peer buckets are charted."),
]

# ---------------------------------------------------------------- build


# Small utilities called from almost everywhere; their call edges would only add clutter.
HELPERS = {"write_parquet", "utc", "table", "load_table", "lemma", "words", "load_documents", "save_documents"}


def first_line(text: str | None) -> str:
    return (text or "").strip().split("\n")[0]


def complexity(lines: int) -> str:
    return "simple" if lines < 25 else "moderate" if lines < 80 else "complex"


def kebab(text: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def build() -> tuple[dict, dict]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(**data: object) -> None:
        nodes[str(data["id"])] = {k: v for k, v in data.items() if v is not None}

    def edge(source: str, target: str, kind: str, weight: float, description: str | None = None) -> None:
        if source in nodes and target in nodes and source != target:
            item = {"source": source, "target": target, "type": kind, "direction": "forward", "weight": weight}
            if description:
                item["description"] = description
            if item not in edges:
                edges.append(item)

    module_of: dict[str, str] = {}  # "ecsa.nlp.features" -> path
    symbols: dict[tuple[str, str], str] = {}  # (path, name) -> node id
    trees: dict[str, ast.Module] = {}
    for path, (summary, level, tags) in FILES.items():
        source = (ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        trees[path] = tree
        node(id=f"file:{path}", type="file", name=Path(path).name, filePath=path,
             lineRange=[1, len(source.splitlines())], summary=summary, tags=tags, complexity=level)
        if path.startswith("src/"):
            module = path[4:-3].replace("/", ".").removesuffix(".__init__")
            module_of[module] = path
        for item in tree.body:
            if not isinstance(item, (ast.FunctionDef, ast.ClassDef)) or item.name.startswith("_"):
                continue
            kind = "class" if isinstance(item, ast.ClassDef) else "function"
            node_id = f"{kind}:{path}:{item.name}"
            symbols[(path, item.name)] = node_id
            span = item.end_lineno - item.lineno + 1
            node(id=node_id, type=kind, name=item.name, filePath=path,
                 lineRange=[item.lineno, item.end_lineno],
                 summary=SYMBOLS.get(f"{path}:{item.name}") or first_line(ast.get_docstring(item)) or f"{kind} {item.name}",
                 tags=[t for t in tags if t not in {"entry-point", "package"}] or ["code"],
                 complexity=complexity(span))
            edge(f"file:{path}", node_id, "contains", 1.0)

    # imports and calls between project modules
    for path, tree in trees.items():
        imported: dict[str, tuple[str, str]] = {}  # local name -> (path, symbol)
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom) and item.module and item.module.startswith("ecsa"):
                target = module_of.get(item.module)
                if target:
                    edge(f"file:{path}", f"file:{target}", "imports", 0.6)
                    for alias in item.names:
                        imported[alias.asname or alias.name] = (target, alias.name)
        for item in tree.body:
            if not isinstance(item, (ast.FunctionDef, ast.ClassDef)) or item.name.startswith("_"):
                continue
            caller = symbols[(path, item.name)]
            for call in ast.walk(item):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    name = call.func.id
                    target = symbols.get(imported[name]) if name in imported else symbols.get((path, name))
                    if target and target.rsplit(":", 1)[-1] not in HELPERS:
                        edge(caller, target, "calls", 0.8)

    # data files: stage functions write them, later stages / dashboard read them
    stage = {name: f"function:src/ecsa/pipeline.py:{name}" for name in ("ingest", "analyze", "event_study", "signals")}
    stage["render"] = "function:src/ecsa/dashboard.py:render"
    for path, summary, writer, readers in TABLES:
        table_id = f"table:{path}"
        node(id=table_id, type="table", name=Path(path).name, filePath=path, summary=summary,
             tags=["data", "parquet", path.split("/")[1]], complexity="simple")
        edge(stage[writer], table_id, "writes_to", 0.9)
        for reader in readers:
            edge(stage[reader], table_id, "reads_from", 0.9)

    for key, name, summary, files in CONCEPTS:
        node(id=f"concept:{key}", type="concept", name=name, summary=summary, tags=["concept"], complexity="simple")
        for path in files:
            edge(f"file:{path}", f"concept:{key}", "related", 0.7, "implements this idea")

    for path, summary in DOCS.items():
        node(id=f"document:{path}", type="document", name=path, filePath=path, summary=summary,
             tags=["documentation"], complexity="simple")
    edge("document:README.md", "document:guidelines.md", "related", 0.5, "method details")
    edge("document:README.md", "file:src/ecsa/pipeline.py", "documents", 0.5)
    edge("document:guidelines.md", "file:src/ecsa/dashboard.py", "documents", 0.5)
    edge("document:change-report.md", "file:src/ecsa/pipeline.py", "documents", 0.3)

    for path, (summary, target) in CONFIGS.items():
        node(id=f"config:{path}", type="config", name=path, filePath=path, summary=summary,
             tags=["configuration"], complexity="simple")
        edge(f"config:{path}", f"file:{target}", "configures", 0.7)

    layers = []
    placed: set[str] = set()
    for key, name, description, prefixes in LAYERS:
        if key == "data":
            members = [n for n in nodes if n.startswith("table:")]
        elif key == "concepts":
            members = [n for n in nodes if n.startswith("concept:")]
        elif key == "docs":
            members = [n for n in nodes if n.startswith(("document:", "config:"))]
        else:
            # Files only: the dashboard shows each file's classes and functions inside it.
            members = [n for n, v in nodes.items()
                       if v["type"] == "file"
                       and any(v["filePath"] == p or (p.endswith("/") and v["filePath"].startswith(p)) for p in prefixes)]
        placed.update(members)
        layers.append({"id": f"layer:{key}", "name": name, "description": description, "nodeIds": members})
    missing = {n for n, v in nodes.items() if v["type"] not in {"function", "class"}} - placed
    assert not missing, f"nodes without a layer: {sorted(missing)}"

    tour = []
    for order, (title, description, ids, lesson) in enumerate(TOUR, start=1):
        unknown = [i for i in ids if i not in nodes]
        assert not unknown, f"tour step {order} references unknown nodes {unknown}"
        step = {"order": order, "title": title, "description": description, "nodeIds": ids}
        if lesson:
            step["languageLesson"] = lesson
        tour.append(step)

    project = {
        "name": "earnings-call-sentiment-analyzer",
        "languages": ["python"],
        "frameworks": ["PyTorch", "Hugging Face Transformers", "pandas", "scikit-learn", "statsmodels", "Streamlit", "Plotly", "Pydantic"],
        "description": "Turns earnings-call transcripts and SEC filings into sentiment features, tests "
                       "them with a −5 to +10 day event study, and converts them into cost-aware "
                       "long/short signals shown in a Streamlit dashboard.",
        "analyzedAt": NOW,
        "gitCommitHash": "none",
    }
    graph = {"version": "1.0.0", "kind": "codebase", "project": project,
             "nodes": list(nodes.values()), "edges": edges, "layers": layers, "tour": tour}

    # domain graph
    dnodes, dedges = [], []
    for key, name, summary, entities, rules, flows in DOMAINS:
        dnodes.append({"id": f"domain:{key}", "type": "domain", "name": name, "summary": summary,
                       "tags": ["domain"], "complexity": "moderate",
                       "domainMeta": {"entities": entities, "businessRules": rules,
                                      "crossDomainInteractions": [d for s, t, d in CROSS_DOMAIN if key in (s, t)]}})
        for fkey, fname, entry, entry_type, steps in flows:
            dnodes.append({"id": f"flow:{fkey}", "type": "flow", "name": fname,
                           "summary": " → ".join(s[0] for s in steps) + ".", "tags": ["flow"],
                           "complexity": "moderate", "domainMeta": {"entryPoint": entry, "entryType": entry_type}})
            dedges.append({"source": f"domain:{key}", "target": f"flow:{fkey}", "type": "contains_flow", "direction": "forward", "weight": 1.0})
            for index, (sname, ssummary, sfile) in enumerate(steps, start=1):
                sid = f"step:{fkey}:{kebab(sname)}"
                dnodes.append({"id": sid, "type": "step", "name": sname, "summary": ssummary, "tags": ["step"],
                               "complexity": "simple", "filePath": sfile})
                dedges.append({"source": f"flow:{fkey}", "target": sid, "type": "flow_step", "direction": "forward",
                               "weight": round(min(1.0, max(0.1, index / len(steps))), 2)})
    for source, target, description in CROSS_DOMAIN:
        dedges.append({"source": f"domain:{source}", "target": f"domain:{target}", "type": "cross_domain",
                       "direction": "forward", "description": description, "weight": 0.6})
    domain = {"version": "1.0.0", "project": project, "nodes": dnodes, "edges": dedges, "layers": [], "tour": []}
    return graph, domain


if __name__ == "__main__":
    graph, domain = build()
    (UA / "knowledge-graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (UA / "domain-graph.json").write_text(json.dumps(domain, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {"lastAnalyzedAt": graph["project"]["analyzedAt"], "gitCommitHash": "none", "version": "1.0.0",
            "analyzedFiles": len(FILES) + len(DOCS) + len(CONFIGS)}
    (UA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for item in graph["nodes"]:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(json.dumps({"nodes": counts, "edges": len(graph["edges"]), "layers": len(graph["layers"]),
                      "tour_steps": len(graph["tour"]), "domain_nodes": len(domain["nodes"])}))

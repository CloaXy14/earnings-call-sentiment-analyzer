"""Read-only processed-data dashboard; no collection, model loading or fitting."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

LABELS = {
    "finbert_net_weighted": "Length-weighted sentiment",
    "finbert_net_mean": "Mean sentiment",
    "lexicon_net": "Lexicon tone",
    "fls_net": "Guidance tone",
    "sentiment_surprise": "Sentiment surprise (z-score)",
}

SIGNAL_REASONS = {
    "ranked": "ranked in the top or bottom quintile of the trailing window",
    "middle_quantile": "composite score fell between the long and short thresholds",
    "middle_quantile_or_capacity": (
        "composite score fell between the thresholds, or position caps were full"
    ),
    "insufficient_ranking_events": "too few events in the trailing ranking window",
    "insufficient_score_history": "too little score history to standardize",
    "insufficient_volatility_history": "too little price history for volatility weighting",
    "features_unavailable_at_day0_close": "features were not available by the day-0 close",
}


def signal_explanation(signal: pd.DataFrame) -> str:
    """Explain why the event did or did not trade; signals use the composite score."""
    if signal.empty:
        return "Signal: no record for this event (it was not eligible for ranking)."
    row = signal.iloc[0]
    reason = str(row.get("reason", ""))
    text = SIGNAL_REASONS.get(reason, reason.replace("_", " ") or "no reason recorded")
    score = row.get("score")
    composite = f"composite score {float(score):+.2f}; " if pd.notna(score) else ""
    return f"Signal {row.side}: {composite}{text}."


@st.cache_data(show_spinner=False)
def load_table(path: str, modified_ns: int) -> pd.DataFrame:
    """Cache a processed Parquet table; file modification time invalidates stale results."""
    return pd.read_parquet(path)


def table(directory: Path, name: str) -> pd.DataFrame:
    """Read a known processed table or return an empty frame when optional data is absent."""
    path = directory / (name + ".parquet")
    return load_table(str(path), path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


def highlighted(sentence: str, phrase: str) -> str:
    """Escape source text before highlighting literal phrase spans; never execute source HTML."""
    escaped = html.escape(sentence)
    pattern = re.compile(re.escape(html.escape(phrase)), re.I)
    return pattern.sub(lambda m: "<mark>" + m.group(0) + "</mark>", escaped)


def event_chart(
    path: pd.DataFrame, peers: pd.DataFrame, score: float, bucket: str, *, bounded: bool = True
) -> go.Figure:
    """Display decimal returns as percentages; baseline is day -6 close."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    ordered = path.sort_values("relative_day")
    figure.add_trace(
        go.Scatter(
            x=ordered.relative_day,
            y=ordered.cumulative_return * 100,
            name="Event cumulative return",
            line={"color": "#94a3b8"},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=ordered.relative_day,
            y=ordered.car * 100,
            name="Event CAR",
            line={"color": "#2563eb", "width": 3},
        ),
        secondary_y=False,
    )
    for number, name, color in [
        (0, "Low", "#dc2626"),
        (2, "Neutral", "#64748b"),
        (4, "High", "#059669"),
    ]:
        group = (
            peers[peers.bucket == number].sort_values("relative_day")
            if not peers.empty
            else pd.DataFrame()
        )
        if group.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=group.relative_day,
                y=group.car_upper * 100,
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=group.relative_day,
                y=group.car_lower * 100,
                line={"width": 0},
                fill="tonexty",
                fillcolor={
                    0: "rgba(220,38,38,.08)",
                    2: "rgba(100,116,139,.08)",
                    4: "rgba(5,150,105,.08)",
                }[number],
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=group.relative_day,
                y=group.mean_car * 100,
                name=name + " mean CAR",
                line={"color": color, "dash": "dot"},
            ),
            secondary_y=False,
        )
    if pd.notna(score):
        figure.add_trace(
            go.Scatter(
                x=[-5, 10],
                y=[score, score],
                name=f"Score {score:+.2f} · {bucket}",
                line={"color": "#7c3aed", "dash": "dash"},
            ),
            secondary_y=True,
        )
    figure.add_vline(x=0, line_dash="dash", line_color="#475569")
    figure.add_vrect(x0=1, x1=10, fillcolor="#2563eb", opacity=0.04, line_width=0)
    figure.update_xaxes(title="Trading days relative to event", dtick=1)
    figure.update_yaxes(title="Cumulative return / CAR (%)", secondary_y=False)
    figure.update_yaxes(title="Sentiment / selected score", secondary_y=True)
    figure.update_layout(
        height=510,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 25},
    )
    figure.update_traces(mode="lines")
    figure.update_yaxes(showgrid=False, zeroline=False, secondary_y=True)
    if bounded:
        figure.update_yaxes(
            range=[-1, 1], tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1], secondary_y=True
        )
    return figure


def render(processed: Path) -> None:
    """Render metrics and evidence from processed Parquet only; compute no research models."""
    events = table(processed, "scored_events")
    if events.empty:
        st.info("No processed results yet. Run `make demo-data` or the research pipeline first.")
        return
    metadata = table(processed, "run_metadata")
    demo = not metadata.empty and bool(metadata.iloc[0].demo)
    st.caption("EARNINGS RESEARCH  /  SYNTHETIC DEMO" if demo else "EARNINGS RESEARCH")
    st.title("What changed in the call?")
    if demo:
        st.info(
            "Synthetic companies, prices and a toy lexical scorer. These "
            "results demonstrate the workflow; they are not FinBERT results or "
            "evidence of trading performance."
        )
    ticker = st.sidebar.selectbox("Ticker", sorted(events.ticker.unique()), key="ticker")
    company = events[events.ticker == ticker].sort_values("event_datetime", ascending=False)
    labels = {
        row.event_id: f"{pd.Timestamp(row.event_datetime).date()} · {row.doc_type}"
        for _, row in company.iterrows()
    }
    selected_id = st.sidebar.selectbox(
        "Event", list(labels), format_func=lambda value: labels[value], key=f"event_{ticker}"
    )
    score_name = st.sidebar.selectbox(
        "Score definition", list(LABELS), format_func=lambda value: LABELS[value], key="score"
    )
    method = st.sidebar.selectbox(
        "Bucket method",
        ["quarter_retrospective", "trailing_history"],
        format_func=lambda value: (
            "Calendar-quarter quintiles (retrospective)"
            if value == "quarter_retrospective"
            else "Prior 90-day history"
        ),
        key="bucket",
    )
    event = company[company.event_id == selected_id].iloc[0]
    signals = table(processed, "signals")
    selected_signal = (
        signals[signals.event_id == selected_id] if not signals.empty else pd.DataFrame()
    )
    side = str(selected_signal.iloc[0].side) if not selected_signal.empty else "NONE"
    history = company[
        (company.available_at < event.available_at)
        & (company.event_datetime < event.event_datetime)
    ]
    percentile = (
        float((history[score_name].dropna() <= event[score_name]).mean() * 100)
        if history[score_name].notna().any()
        else float("nan")
    )
    score = float(event[score_name])
    membership = table(processed, "event_buckets")
    if not membership.empty:
        membership = membership[
            (membership.event_id == selected_id)
            & (membership.score_definition == score_name)
            & (membership.bucket_method == method)
        ]
    bucket = (
        f"Quintile {int(membership.iloc[0].bucket) + 1}" if not membership.empty else "Unclassified"
    )
    columns = st.columns(4)
    columns[0].metric(
        "Overall tone",
        f"{event.finbert_net_mean:+.2f}",
        help="Mean sentence net sentiment, from -1 to +1. Demo uses a synthetic scorer.",
    )
    columns[1].metric(
        "Sentiment surprise",
        f"{event.sentiment_surprise:+.2f}" if pd.notna(event.sentiment_surprise) else "—",
        help="Z-score relative to the company's previous four known events; requires at least two.",
    )
    columns[2].metric(
        "Forward-looking tone",
        f"{event.fls_net:+.2f}" if pd.notna(event.fls_net) else "—",
        help="Mean net sentiment of forward-looking business statements after disclaimer removal.",
    )
    columns[3].metric(
        "Signal",
        side,
        help=(
            "Position formed at day-0 close using only then-available features "
            "and historical thresholds."
        ),
    )
    st.caption(
        f"{LABELS[score_name]}: {score:+.3f} · {percentile:.0f}th percentile vs "
        f"prior company history · {bucket}. Text available "
        f"{event.document_available_at}; features available "
        f"{event.available_at}."
    )
    st.caption(signal_explanation(selected_signal))
    research, strategy = st.tabs(["Event & evidence", "Strategy"])
    with research:
        paths = table(processed, "event_paths")
        peers = table(processed, "bucket_paths")
        peers = (
            peers[(peers.score_definition == score_name) & (peers.bucket_method == method)]
            if not peers.empty
            else peers
        )
        st.plotly_chart(
            event_chart(
                paths[paths.event_id == selected_id],
                peers,
                score,
                bucket,
                bounded=score_name != "sentiment_surprise",
            ),
            width="stretch",
            key="event_chart",
        )
        st.caption(
            "[-5,-1]: anticipation. Day 0: reaction. Shaded [+1,+10]: "
            "post-event prediction from entry at day-0 close. Peer averages "
            "are retrospective; 95% bands cluster by day-0 date."
        )
        st.subheader("Words behind the score")
        phrases = table(processed, "phrases")
        phrases = phrases[phrases.event_id == selected_id] if not phrases.empty else phrases
        left, right = st.columns(2)
        for container, category in ((left, "positive"), (right, "negative")):
            with container:
                st.markdown("**" + category.title() + " phrases**")
                group = phrases[phrases.category == category] if not phrases.empty else phrases
                if group.empty:
                    st.caption("No matched phrases.")
                else:
                    for phrase, count in group.phrase.value_counts().head(8).items():
                        with st.expander(f"{phrase} · {count}"):
                            for _, evidence in (
                                group[group.phrase == phrase].drop_duplicates("sentence").iterrows()
                            ):
                                st.markdown(
                                    highlighted(
                                        str(evidence.sentence),
                                        str(evidence.get("matched_text", phrase)),
                                    ),
                                    unsafe_allow_html=True,
                                )
        topics = json.loads(event.themes)
        st.markdown("**Themes learned from prior documents**")
        st.write(" · ".join(topics) if topics else "More prior documents are needed for topics.")
        with st.expander("Top key phrases and supporting sentences"):
            for phrase in json.loads(event.key_phrases):
                st.markdown(
                    highlighted(phrase["sentence"], phrase["phrase"]), unsafe_allow_html=True
                )
        sentences = table(processed, "sentences")
        sentences = sentences[sentences.event_id == selected_id].sort_values("event_ordinal")
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=sentences.event_ordinal, y=sentences.net, name="Sentence sentiment", opacity=0.35
            )
        )
        chart.add_trace(
            go.Scatter(
                x=sentences.event_ordinal,
                y=sentences.net.rolling(5, min_periods=1).mean(),
                name="5-sentence rolling mean",
                line={"color": "#2563eb", "width": 3},
            )
        )
        qa = sentences[sentences.section == "qa"]
        if not qa.empty:
            chart.add_vline(
                x=int(qa.event_ordinal.min()), line_dash="dash", annotation_text="Q&A begins"
            )
        chart.update_layout(
            title="Tone across the call",
            xaxis_title="Sentence number (zero-based)",
            yaxis_title="Net sentiment (-1 to +1)",
            template="plotly_white",
            height=320,
        )
        st.plotly_chart(chart, width="stretch", key="sentence_chart")
        with st.expander("Forward-looking statements"):
            for _, sentence in sentences[sentences.forward_looking].iterrows():
                st.write(f"{sentence.speaker}: {sentence.text}")
        with st.expander("Removed boilerplate"):
            removed = table(processed, "removed_text")
            st.dataframe(
                removed[removed.event_id == selected_id][["reason", "text"]], hide_index=True
            )
    with strategy:
        daily = table(processed, "backtest")
        placebo = table(processed, "placebo")
        curve = go.Figure()
        for column, name, color in [
            ("equity", "Strategy after costs", "#2563eb"),
            ("benchmark_equity", "SPY", "#64748b"),
        ]:
            curve.add_trace(
                go.Scatter(x=daily.date, y=daily[column], name=name, line={"color": color})
            )
        if not placebo.empty:
            curve.add_trace(
                go.Scatter(
                    x=placebo.date, y=placebo.placebo_upper, showlegend=False, line={"width": 0}
                )
            )
            curve.add_trace(
                go.Scatter(
                    x=placebo.date,
                    y=placebo.placebo_lower,
                    showlegend=False,
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(124,58,237,.10)",
                )
            )
            curve.add_trace(
                go.Scatter(
                    x=placebo.date,
                    y=placebo.placebo_equity,
                    name="Shuffled-score placebo mean",
                    line={"color": "#7c3aed", "dash": "dot"},
                )
            )
        curve.update_layout(
            template="plotly_white",
            height=430,
            yaxis_title="Growth of 1 unit of capital",
            xaxis_title="Session date",
        )
        curve.update_traces(mode="lines")
        st.plotly_chart(curve, width="stretch", key="strategy_chart")
        st.dataframe(table(processed, "strategy_metrics"), hide_index=True, width="stretch")
        ic = table(processed, "strategy_ic")
        st.caption(
            "Predictive information coefficient over events whose features were "
            "available by the day-0 close only."
        )
        st.dataframe(ic, hide_index=True, width="stretch")
        if not metadata.empty:
            st.caption(
                f"{metadata.iloc[0].placebo_seeds} placebo seeds · "
                f"{metadata.iloc[0].transaction_cost_bps:g} bps per side, "
                f"including rebalancing · {metadata.iloc[0].variants_tried} "
                f"strategy variants reported. Returns in tables are decimals. "
                f"Hit rate and average win/loss come from the same daily "
                f"target-weight rebalancing ledger as the equity curve."
            )
        st.caption(
            "Research diagnostics: BMP does not remove cross-firm correlation. "
            "Current-vintage adjusted prices and historical universe selection "
            "require validation before interpreting empirical results."
        )

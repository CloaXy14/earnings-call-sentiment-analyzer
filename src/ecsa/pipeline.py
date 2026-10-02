"""CLI: ingest → analyze → event-study → signals; --demo is fully offline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ecsa.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    """Build CLI without loading models or connecting to data sources."""
    parser = argparse.ArgumentParser(description="Earnings Call Sentiment Analyzer (ecsa)")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    stages = parser.add_subparsers(dest="command", required=True)
    for name, description in {
        "ingest": "Collect licensed documents and prices",
        "analyze": "Generate sentiment features",
        "event-study": "Estimate abnormal returns and statistical evidence",
        "signals": "Rank events, backtest and run placebo seeds",
        "all": "Run the complete pipeline",
    }.items():
        stage = stages.add_parser(name, help=description)
        stage.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        stage.add_argument(
            "--demo",
            action="store_true",
            help="Use bundled synthetic sources and the explicit demo scorer; no network",
        )
        if name in {"ingest", "all"}:
            stage.add_argument("--tickers", nargs="+")
            stage.add_argument("--start", type=date.fromisoformat)
            stage.add_argument("--end", type=date.fromisoformat)
    return parser


def ingest(
    settings: Settings,
    tickers: list[str] | None,
    start: date | None,
    end: date | None,
    *,
    demo: bool = False,
) -> dict[str, int]:
    """Persist source documents and decimal-price daily bars as raw/interim Parquet."""
    import pandas as pd

    from ecsa.data.edgar import EdgarClient
    from ecsa.data.prices import LocalPriceSource, YFinancePriceSource, daily_returns, validate_bars
    from ecsa.data.transcripts import LocalFileSource
    from ecsa.schemas import Document
    from ecsa.storage import load_documents, save_documents, write_parquet

    if demo:
        existing = settings.interim_dir / "documents.parquet"
        if existing.exists() and any(
            d.source != "synthetic_fixture" for d in load_documents(existing)
        ):
            raise ValueError(
                "Demo refuses to overwrite real intermediate documents; choose "
                "a separate data_dir in YAML"
            )
        fixtures = Path(__file__).resolve().parent / "sample_data"
        # Bundled package data also permits a wheel installation outside the checkout.
        documents = [
            Document.model_validate(row)
            for row in json.loads((fixtures / "sample_documents.json").read_text())
        ]
        prices = validate_bars(pd.read_parquet(fixtures / "sample_prices.parquet"))
    else:
        if not tickers or start is None or end is None or start > end:
            raise ValueError("ingest requires --tickers, --start and --end (inclusive) or --demo")
        documents = []
        local = LocalFileSource(settings.raw_dir / "transcripts")
        sec = EdgarClient(settings) if settings.include_edgar or settings.include_mdna else None
        for ticker in tickers:
            documents.extend(local.fetch(ticker, start, end))
            if sec:
                cik = sec.ticker_to_cik(ticker)
                if settings.include_edgar:
                    documents.extend(sec.earnings_8ks(cik, start, end, ticker))
                if settings.include_mdna:
                    for form in ("10-Q", "10-K"):
                        # Retain historical MD&A by querying each filing date.
                        dates = {f.filing_date for f in sec.list_filings(cik, [form], start, end)}
                        for filing_day in sorted(dates):
                            document = sec.mdna(
                                cik, form, start=filing_day, end=filing_day, ticker=ticker
                            )
                            if document:
                                documents.append(document)
        source = (
            LocalPriceSource(settings.raw_dir / "prices.parquet")
            if settings.price_backend == "local"
            else YFinancePriceSource(settings)
        )
        prices = pd.concat(
            [
                source.daily_bars(
                    ticker,
                    start - timedelta(days=2 * abs(settings.estimation_start) + 30),
                    end + timedelta(days=3 * settings.event_end),
                )
                for ticker in sorted(set(tickers + [settings.benchmark]))
            ],
            ignore_index=True,
        )
    documents = list({d.doc_id: d for d in documents}.values())
    if not documents:
        raise ValueError(
            "No licensed documents found. Supply local files or enable SEC collection."
        )
    if prices.empty:
        raise ValueError("No price bars found")
    save_documents(documents, settings.raw_dir / "documents.parquet")
    save_documents(documents, settings.interim_dir / "documents.parquet")
    write_parquet(prices, settings.raw_dir / "collected_prices.parquet")
    write_parquet(daily_returns(prices), settings.interim_dir / "prices.parquet")
    return {"documents": len(documents), "price_bars": len(prices)}


def analyze(settings: Settings) -> dict[str, int]:
    """Write one features row per event and sentence/phrase/removal evidence."""
    from ecsa.nlp.features import analyze_documents
    from ecsa.nlp.finbert import DemoScorer, FinBertScorer, SentimentModel
    from ecsa.nlp.llm import LlmScorer
    from ecsa.storage import load_documents, write_parquet

    documents = load_documents(settings.interim_dir / "documents.parquet")
    scorer: SentimentModel = (
        DemoScorer()
        if settings.nlp_backend == "demo"
        else LlmScorer(settings)
        if settings.llm_enabled
        else FinBertScorer(settings)
    )
    result = analyze_documents(documents, scorer, settings)
    if result.features.empty:
        raise ValueError("No analyzable sentences after cleaning")
    for name, frame in (
        ("features", result.features),
        ("sentences", result.sentences),
        ("phrases", result.phrases),
        ("removed_text", result.removed),
    ):
        write_parquet(frame, settings.processed_dir / (name + ".parquet"))
    return {"events": len(result.features), "sentences": len(result.sentences)}


def event_study(settings: Settings) -> dict[str, int]:
    """Persist retrospective event returns, grouped paths, window tests, IC and regression."""
    import pandas as pd

    from ecsa.events.stats import (
        SCORES,
        bucket_membership,
        bucket_paths,
        drift_regression,
        information_coefficient,
        window_tests,
    )
    from ecsa.events.study import run_study
    from ecsa.storage import write_parquet

    events, paths = run_study(
        pd.read_parquet(settings.processed_dir / "features.parquet"),
        pd.read_parquet(settings.interim_dir / "prices.parquet"),
        settings,
    )
    if events.empty:
        raise ValueError("No events align to supplied benchmark sessions")
    write_parquet(events, settings.processed_dir / "event_study.parquet")
    write_parquet(paths, settings.processed_dir / "event_paths.parquet")
    write_parquet(window_tests(events), settings.processed_dir / "window_tests.parquet")
    memberships = pd.concat(
        [
            bucket_membership(
                events,
                score,
                method=method,
                ranking_days=settings.ranking_days,
                minimum_events=settings.minimum_ranking_events,
            )
            for score in SCORES
            for method in ("quarter_retrospective", "trailing_history")
        ],
        ignore_index=True,
    )
    write_parquet(memberships, settings.processed_dir / "event_buckets.parquet")
    write_parquet(
        pd.concat(
            [
                bucket_paths(
                    events,
                    paths,
                    score,
                    method=method,
                    ranking_days=settings.ranking_days,
                    minimum_events=settings.minimum_ranking_events,
                )
                for score in SCORES
                for method in ("quarter_retrospective", "trailing_history")
            ],
            ignore_index=True,
        ),
        settings.processed_dir / "bucket_paths.parquet",
    )
    write_parquet(
        pd.concat([information_coefficient(events, score) for score in SCORES], ignore_index=True),
        settings.processed_dir / "information_coefficients.parquet",
    )
    write_parquet(
        drift_regression(events, "finbert_net_weighted"),
        settings.processed_dir / "regression.parquet",
    )
    return {"event_study_rows": len(events)}


def signals(settings: Settings) -> dict[str, int]:
    """Generate causal signals and overlapping portfolios, then retrospective evaluation."""
    import pandas as pd

    from ecsa.events.calendar import TradingCalendar
    from ecsa.events.stats import decision_available, information_coefficient
    from ecsa.signals.backtest import backtest, placebo
    from ecsa.signals.metrics import performance
    from ecsa.signals.model import walk_forward
    from ecsa.signals.portfolio import build_signals
    from ecsa.signals.scoring import composite_scores
    from ecsa.storage import write_parquet

    events = composite_scores(
        pd.read_parquet(settings.processed_dir / "event_study.parquet"), settings
    )
    prices = pd.read_parquet(settings.interim_dir / "prices.parquet")
    calendar = TradingCalendar(list(prices[prices.ticker == settings.benchmark].date))
    if settings.learned_model != "none":
        events = walk_forward(events, calendar, settings)
        events["score"] = events.model_score
    selected, rejected = build_signals(events, prices, calendar, settings)
    daily = backtest(selected, prices, settings)
    placebo_path, placebo_summary = placebo(events, prices, calendar, settings)
    tables = {
        "scored_events": events,
        "signals": selected,
        "rejected_signals": rejected,
        "backtest": daily,
        "placebo": placebo_path,
        "placebo_seeds": placebo_summary,
        "strategy_metrics": performance(daily, selected, prices, events, settings),
        "strategy_ic": information_coefficient(decision_available(events, calendar), "score"),
        "strategy_ic_retrospective": information_coefficient(events, "score"),
    }
    for name, frame in tables.items():
        write_parquet(frame, settings.processed_dir / (name + ".parquet"))
    metadata = pd.DataFrame(
        [
            {
                "demo": settings.nlp_backend == "demo",
                "seed": settings.seed,
                "variants_tried": settings.variants_tried,
                "model_id": str(events.iloc[0].model_id),
                "model_revision": str(events.iloc[0].model_revision),
                "placebo_seeds": settings.placebo_seeds,
                "transaction_cost_bps": settings.transaction_cost_bps,
                "data_vintage_note": "Synthetic fixture"
                if settings.nlp_backend == "demo"
                else "Check provider point-in-time revisions and universe survivorship.",
            }
        ]
    )
    write_parquet(metadata, settings.processed_dir / "run_metadata.parquet")
    return {
        "signals": int((selected.side != "NONE").sum()),
        "rejected_events": len(rejected),
        "placebo_seeds": settings.placebo_seeds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute selected stages; errors are visible and never reported as successful runs."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        settings = Settings.from_yaml(args.config)
        if args.demo:
            settings = settings.model_copy(
                update={
                    "nlp_backend": "demo",
                    "llm_enabled": False,
                    "include_edgar": False,
                    "include_mdna": False,
                }
            )
        commands = (
            ["ingest", "analyze", "event-study", "signals"]
            if args.command == "all"
            else [args.command]
        )
        for command in commands:
            logging.info("Running %s", command)
            result: dict[str, Any]
            if command == "ingest":
                result = ingest(
                    settings,
                    args.tickers or settings.tickers,
                    args.start or settings.start_date,
                    args.end or settings.end_date,
                    demo=args.demo,
                )
            elif command == "analyze":
                result = analyze(settings)
            elif command == "event-study":
                result = event_study(settings)
            else:
                result = signals(settings)
            print(json.dumps({"stage": command, **result}))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

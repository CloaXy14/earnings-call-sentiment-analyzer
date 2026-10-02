"""Read-only dashboard entry point: streamlit run app/streamlit_app.py."""

from pathlib import Path

import streamlit as st

from ecsa.dashboard import render
from ecsa.settings import Settings


def main() -> None:
    """Load YAML and render cached processed tables; no heavy inference or network."""
    st.set_page_config(page_title="Earnings Call Sentiment Analyzer", page_icon="◈", layout="wide")
    settings = Settings.from_yaml(Path(__file__).resolve().parents[1] / "config/settings.yaml")
    render(settings.processed_dir)


if __name__ == "__main__":
    main()

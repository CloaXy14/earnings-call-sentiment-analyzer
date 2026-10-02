# Audit expectations (recorded before implementation review)
Source: original change-report.md, README.md, global Agent Team v0.9.
- Downloaded sample retains truthful issuer, event date and availability metadata from the source; no invented financial-event timestamps.
- LocalFileSource parses the saved transcript; pinned FinBERT weights load and output finite probabilities summing to one; lexicon score runs. This is a smoke test, not measured model accuracy.
- Missing input or inference failure must be visible, not reported as a successful evaluation.
- Existing pytest suite runs offline; make check passes ruff and strict mypy; package builds.
- Demo runs all four stages with finite valid artifacts and deterministic outputs; availability/no-look-ahead tests remain passing.
- Streamlit pages load generated data without exceptions. Real browser evidence will be recorded if available.

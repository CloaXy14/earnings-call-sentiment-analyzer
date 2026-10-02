# Step 0 change report

## Summary

- Workflow: Agent Team manual `feature+quality` (global v0.9 specification, v4 runbook).
- Task checklist: feature, with data/integration contract checks.
- Project root: `/Users/leonhardaudie/Documents/antigravity/Projects/earnings-call-sentiment-analyzer`.
- Run ID: `ecsa-step-0-scaffold-20261001`.
- Source: no Git repository. Intake contained only `guidelines.md`, which is unchanged.
- Scope/outcome: Step 0 completed; Steps 1-6 remain unimplemented.
- Source hashes: [source-sha256.json](.agent-team/evidence/step-0/source-sha256.json).
- Created-file inventory: [created-files.txt](.agent-team/evidence/step-0/created-files.txt).

## Acceptance evidence

Expectations were taken from the user's Step 0/global rules and recorded in the
intake checkpoint before implementation. Representative contracts:

| Input | Expected result | Evidence |
|---|---|---|
| `python -m ecsa.pipeline --help` | Exit 0, all five commands listed, no data access | CLI smoke and subprocess test outside project cwd |
| Any unfinished command | Clear error, exit 2, no writes | Parametrized CLI tests |
| Same source/URL/text | Stable SHA-256; changing any field changes ID | Identity tests and JSON round trip |
| Naive record timestamp | Validation error | Every timestamp field on Document, PriceBar, EventFeatures and Signal |
| Winter/summer offset | Normalize to correct UTC instant | Synthetic winter/summer cases |
| Future source/feature availability | Reject feature/signal; equality allowed | Information-time boundary tests |
| YAML secret or invalid config | Validation error | YAML field, revision, rate, timeout, timezone tests |
| Environment credential | Available to Settings, absent from repr/dump | Synthetic monkeypatched credentials |
| Weight 0.05 | Decimal 5% exposure, distinct LONG/SHORT/NONE rules | Signal range and exposure tests |
| Invalid OHLC/volume or sentiment share | Validation error | Synthetic record tests |

These are scaffold contract checks. They do not establish absence of look-ahead
bias in future algorithms, model fits, or data-vendor histories.

## Added

All requested module paths, six typed Pydantic records, a single Settings entry
point, CLI parser/entry point, offline test fixtures, YAML configuration, README,
dependency metadata, and Streamlit status placeholder. Model inference and future
stage modules are explicitly reserved. Added `py.typed`, exact hashed development
dependency lock, and ignored local data/runtime directories.

The required document availability timestamp separates the time of a call from
the time its completed text was knowable. Feature and signal records validate
their availability boundaries. Settings loads credentials only from the process
environment. FinBERT's configured revision was verified using upstream metadata;
no model weights were downloaded.

## Changed

No pre-existing source or expectations were changed. The first lint command also
scanned the locally installed `.python/` standard library. Added an explicit Ruff
runtime exclusion, then verified all application and test files. Formatting changed
three newly created test files without changing expectations.

## Removed and discarded

No existing behavior or tests removed. No new tests discarded: retained cases cover
distinct fields, boundaries, error conditions, or actual entry points. Per-test
coverage attribution and mutation testing were not measured.

## Checks and numbers

Environment: Python 3.11.16, pytest 9.1.1, pytest-socket 0.8.1, Pydantic 2.13.5,
PyYAML 6.0.3, Ruff 0.16.9, mypy 1.20.2 on macOS x86_64. Full version evidence is
in [versions.txt](.agent-team/evidence/step-0/versions.txt).

| Check | Result | Raw evidence |
|---|---|---|
| Baseline `python3 -m ecsa.pipeline --help` | Failed: package did not exist | Intake checkpoint; `ModuleNotFoundError: No module named 'ecsa'` |
| Baseline `python3 -m pytest` | Blocked: pytest absent in Python 3.10.7 | Intake checkpoint; `No module named pytest` |
| `.venv/bin/pytest -q` | 78 passed, 0.87 seconds; sockets disabled | [pytest.txt](.agent-team/evidence/step-0/pytest.txt) |
| `.venv/bin/python -m ecsa.pipeline --help` | Passed, exit 0 | [cli-help.txt](.agent-team/evidence/step-0/cli-help.txt) |
| `.venv/bin/ruff check .` | Passed | [ruff.txt](.agent-team/evidence/step-0/ruff.txt) |
| `.venv/bin/ruff format --check src app tests` | Passed, 35 files formatted | Verification command output |
| `.venv/bin/mypy` | Passed, 30 package source files | [mypy.txt](.agent-team/evidence/step-0/mypy.txt) |
| `uv --cache-dir /private/tmp/ecsa-uv-cache build --offline --python .venv/bin/python --wheel` | Passed; wheel built | [build.txt](.agent-team/evidence/step-0/build.txt) |

Development packages were installed with hash verification from
`requirements-dev.lock`. Python and dependencies required setup downloads; the
test suite and final wheel build ran offline. The wheel is
`dist/ecsa-0.1.0-py3-none-any.whl`. No coverage or mutation percentage is claimed.

## Capabilities unavailable / not done

Research extras and Streamlit rendering were not installed or verified. Model
inference, network clients, Parquet pipelines, event statistics, strategy results,
the offline dashboard, and `make demo` are outside Step 0. There are no external
data credentials, downloaded datasets, commits, publication actions, or automatic
Agent Team scheduler installed by this task.

## Bugs found

No unresolved in-scope product defects. The lint runtime discovery issue was
corrected. No test expectations were weakened to obtain passing results.

## Next step

Stop here as requested. On the user's instruction to continue, start Step 1 data
collection; verify `HANDOFF.md` and source evidence before editing.

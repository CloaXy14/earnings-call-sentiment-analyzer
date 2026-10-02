# Verification report — 2026-10-02

> **Remediation update — 2026-10-02 (manual `feature+quality`).** All eight audit
> findings (F1–F8) below have since been fixed by the primary assistant under the
> manual feature+quality contract, and the fixes were verified. This section records
> the remediation; the original audit findings are preserved verbatim beneath it as
> the historical record.
>
> **Post-remediation status: PASS on the executed gates.**
>
> | Check | Result |
> |---|---|
> | `make test` (`.venv/bin/python -m pytest`) | **PASS: 138 tests** (134 prior + 4 new F2/F4/F6/F7 regressions) |
> | `make check` → `.venv/bin/ruff check .` | **PASS: all checks passed** (was FAIL with 183 findings) |
> | `make check` → `.venv/bin/mypy` | **PASS: 33 source files** |
> | Fixed `scripts/download_data.py` against live dataset | **PASS: 153/153 segments**, contiguous, metadata left `unverified` (no fabrication) |
> | Fixed `scripts/test_analyzer.py` failure contract | **PASS: exits nonzero** on unverified data, no documents, or empty features |
> | Full CLI `all --demo`, isolated data directory | **PASS: 96 events, 451 sentences, 30 signals, 8 rejected, 100 placebo seeds** |
>
> **What changed, by finding:**
>
> - **F1 (invented metadata):** `scripts/download_data.py` rewritten to keep `ticker`,
>   `event_datetime` and `available_at` `null` with `metadata_status: "unverified"`
>   rather than deriving a ticker from an issuer name or a date from a filename. The
>   staging artifact is explicitly not pipeline-ready until verified provenance is
>   supplied. `scripts/download_hf_data.py` (which reproduced the same fabrication and
>   was superseded) was removed.
> - **F2 (previous-close leak):** `src/ecsa/events/study.py` now derives each return's
>   availability as `max(available_at[t], available_at[t-1])` and applies it to
>   estimation eligibility and every outcome label, so a delayed prior close can no
>   longer mature a label early or alter a learned prediction.
> - **F3 (silent truncation):** the downloader paginates to the end of each call,
>   verifies segment-id continuity and expected counts, and records a completeness
>   flag; the full 153-segment Alphabet call is retained instead of the first 100.
> - **F4 (missing-bar horizon):** `src/ecsa/events/study.py` and
>   `src/ecsa/events/calendar.py` (`planned_offset_or_none`) anchor the event window to
>   the scheduled NYSE sessions. A missing benchmark bar leaves the window NaN and the
>   label unmatured instead of sliding +10 onto a later observed session.
> - **F5 (false success):** `scripts/test_analyzer.py` rewritten to exit nonzero on any
>   pipeline exception, on empty feature output, and on the unverified staging artifact;
>   failures propagate rather than printing `Pipeline error:` and exiting 0.
> - **F6 (post-outcome IC):** `src/ecsa/events/stats.py` adds `decision_available()`,
>   which restricts the predictive information coefficient to events whose features were
>   available by the day-0 close. `src/ecsa/signals/metrics.py` and
>   `src/ecsa/pipeline.py` report that predictive IC, and keep the all-events
>   correlation separately as `ic_retrospective` / `strategy_ic_retrospective`.
> - **F7 (hit-rate accounting):** `src/ecsa/signals/metrics.py` computes hit rate and
>   average win/loss by replaying each trade through the production daily target-weight
>   `backtest()` ledger, so trade statistics reconcile with the equity curve instead of
>   using endpoint buy-and-hold returns.
> - **F8 (lint gate):** script lint findings fixed; generated `.ua` tooling excluded
>   from the lint gate in `pyproject.toml` (project `src/`, `scripts/`, `tests/`, `app/`
>   remain fully linted); the superseded legacy downloader removed. `make check` now
>   passes ruff and mypy.
>
> **Files changed in remediation:** `src/ecsa/events/study.py`,
> `src/ecsa/events/calendar.py`, `src/ecsa/events/stats.py`,
> `src/ecsa/signals/metrics.py`, `src/ecsa/pipeline.py`, `src/ecsa/dashboard.py`,
> `scripts/download_data.py`, `scripts/test_analyzer.py`, `pyproject.toml`,
> `tests/test_events_signals.py`, `data/raw/real_sample_transcripts.json`; removed
> `scripts/download_hf_data.py`.
>
> **Scope note / limits retained:** no labeled sentiment benchmark, real-data
> end-to-end trading validation, or live vendor/SEC/LLM integration was performed; those
> accuracy limitations from the audit still stand. Running the F2 reproducer during
> verification overwrote `.agent-team/evidence/audit-20261002/core-leakage.json` with
> its now-passing output.

---

## Summary

**Original audit (preserved): overall quality verdict FAIL.** The findings in this
section describe the state **before** the remediation recorded above. The software ran,
but the real-data sample and several statistical paths were not reliable enough for a
correctness sign-off. Successful inference is not evidence of sentiment accuracy or predictive value.

- Workflow: Agent Team v0.9 manual `quality-only`, following Section 0, RUNBOOK and TASKS.
- Task checklist: quality audit, data/integration, dashboard verification.
- Frozen root: `/Users/leonhardaudie/Documents/antigravity/Projects/earnings-call-sentiment-analyzer`.
- Run ID: `ecsa-audit-20261002`. Coordinator ran gates and independently reproduced the numerical findings returned by a read-only specialist.
- No Git repository. Intake hashes: [source-sha256.json](.agent-team/evidence/audit-20261002/source-sha256.json). Existing source/config/tests matched the prior final snapshot; three scripts were absent from that snapshot. Absence establishes the comparison scope, not their exact creation time. All intake-hashed files remain unchanged at completion.
- Original [change report](.agent-team/evidence/audit-20261002/original-change-report.md) and [handoff](.agent-team/evidence/audit-20261002/original-HANDOFF.md) are preserved. Defects below remain unfixed under this audit's read-only production scope.

## Acceptance evidence and checks

| Check | Observed result | Evidence |
|---|---|---|
| `make test` | **PASS: 134 tests**, 36.07 s pytest time; sockets disabled | [pytest.txt](.agent-team/evidence/audit-20261002/pytest.txt) |
| `.venv/bin/mypy` | **PASS: 33 source files** | [mypy.txt](.agent-team/evidence/audit-20261002/mypy.txt) |
| `make check` / `.venv/bin/ruff check .` | **FAIL: 183 lint findings**: 177 in `.ua/`, 6 in `scripts/` | [make-check.txt](.agent-team/evidence/audit-20261002/make-check.txt), [ruff JSON](.agent-team/evidence/audit-20261002/ruff-json.txt) |
| `uv pip check --python .venv/bin/python` | **PASS: 100 packages compatible** | [dependencies.txt](.agent-team/evidence/audit-20261002/dependencies.txt) |
| Original `scripts/download_data.py` | **PASS transport**, one record written in an isolated directory; contents match original sample. Data correctness fails F1/F3 below | [download.txt](.agent-team/evidence/audit-20261002/download.txt) |
| Original `.venv/bin/python scripts/test_analyzer.py` | **PASS smoke**, one event; does not execute price ingestion, event study, signals or backtest | [real-analyzer.txt](.agent-team/evidence/audit-20261002/real-analyzer.txt) |
| Fresh FinBERT inference, isolated empty sentence cache | **PASS runtime**: 217 sentences; finite probabilities in [0,1], sums within 1e-6 of 1 | [model verification](.agent-team/evidence/audit-20261002/model-verification.json) |
| Pinned model provenance | **PASS**: local 437,992,753-byte weight file SHA-256 matches upstream LFS metadata for `4556d13015211d73dccd3fdd39d39232506f3e43` | [upstream verification](.agent-team/evidence/audit-20261002/upstream-verification.json) |
| Full CLI `all --demo`, default 100 placebo seeds, isolated data directory | **PASS twice**: 96 documents/events, 451 sentences, 30 signals, 8 rejected events; 43.28 s and 49.87 s | [run 1](.agent-team/evidence/audit-20261002/demo-1.txt), [run 2](.agent-team/evidence/audit-20261002/demo-2.txt) |
| Demo repeatability | **PASS**: all processed Parquet byte hashes identical between runs | [hashes 1](.agent-team/evidence/audit-20261002/demo-1-hashes.json), [hashes 2](.agent-team/evidence/audit-20261002/demo-2-hashes.json) |
| Isolated `uv build --wheel` and extracted-wheel import/data smoke | **PASS**: package contains fixtures; imports from extracted wheel and ingests 96 documents/5,040 bars | [build.txt](.agent-team/evidence/audit-20261002/build.txt), [wheel-smoke.txt](.agent-team/evidence/audit-20261002/wheel-smoke.txt) |
| Headless Chrome, existing dashboard data, 1400px desktop / 390px mobile | **PASS checked flows**: 4 metrics, 3 chart elements, no page/app exceptions or horizontal document overflow; ticker selection and Strategy tab exercised | [browser.json](.agent-team/evidence/audit-20261002/browser.json) |
| New-script failure contracts and numerical validity reproducers | **FAIL desired contracts**, confirmed defects F1–F7 | linked reproducers below |

Desktop event and Strategy screenshots and the mobile initial view were visually inspected. Streamlit uses an internal scrolling container, so these screenshots do not establish a complete mobile visual audit of every lower-page table. Existing AppTest coverage also exercises score and bucket controls. Browser verification used the current project dashboard artifacts; the fresh CLI demo was kept in an isolated directory.

Fresh sample scores: FinBERT mean **0.2867957961**, lexicon net **0.3225806452**, forward-looking net **0.0408439406**. These are scores for an incomplete, incorrectly dated transcript, not accuracy percentages. There are no human sentiment labels, held-out test set or confidence intervals supporting task accuracy. The lexicon uses configured phrases (`dictionary_path: null`), not a supplied Loughran–McDonald dictionary.

## Bugs found

### F1 — P1: Downloaded financial-event metadata is invented — **RESOLVED (see remediation)**

Location: [scripts/download_data.py:55](scripts/download_data.py#L55), especially lines 57, 65 and 80–82.

The saved Alphabet record uses ticker `ALPHA`, event time `2025-01-01T12:00:00Z`, and availability `2025-01-01T14:00:00Z`. The script derives a ticker by truncating an issuer name and assigns January 1 from a filename year. Alphabet's official Q1 2025 call occurred **April 24, 2025 at 1:30 PM PT**; the earnings release identifies the company tickers as GOOG/GOOGL. See [official call record](https://abc.xyz/investor/events/event-details/2025/2025-Q1-Earnings-Call/) and [official SEC-filed release](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000040/googexhibit991q12025.htm).

Impact: wrong security joins and false historical availability. Downstream point-in-time checks cannot correct truthful-looking but fabricated timestamps. This sample is unsuitable for event-study/backtest use.

Remediation: require verified issuer/security mapping, event time and actual source-availability provenance. Keep unknown metadata unknown or reject it; do not replace it with invented dates. Evidence: [script-contracts.json](.agent-team/evidence/audit-20261002/script-contracts.json), original and freshly downloaded sample.

### F2 — P1: Unavailable previous-close prices leak into historical predictions — **RESOLVED (see remediation)**

Location: [src/ecsa/events/study.py:59](src/ecsa/events/study.py#L59), lines 59–61 and 91–104.

Returns depend on two closing prices, but estimation eligibility checks only the current bar's availability. Drift-label availability likewise omits the day-0 denominator price. In the saved reproducer, changing a price unavailable until **2023-03-28** changes a prediction dated **2023-02-14** from approximately zero to **−0.5251315545**. Its affected label is incorrectly marked available January 17.

Impact: the optional learned model can train on information unavailable at prediction time, contradicting README's no-look-ahead requirement. This is an existing production defect, not introduced by the new scripts; default `learned_model: none` does not activate this demonstrated learned-model path.

Remediation: propagate `max(current_price_available_at, previous_price_available_at)` into every return; use it for estimation eligibility and all outcome labels. Preserve missing inputs. Add this delayed-historical-price perturbation regression.

Evidence: [core-leakage.json](.agent-team/evidence/audit-20261002/core-leakage.json); [runnable reproducer](.agent-team/evidence/audit-20261002/ecsa-core-repro.py.txt).

### F3 — P1: Downloader silently truncates the call — **RESOLVED (see remediation)**

Location: [scripts/download_data.py:24](scripts/download_data.py#L24), lines 24–35 and 74–76.

Only offset 0, length 100 is requested. The live dataset has **153 rows**; rows 101–153 are additional segments of the same Alphabet call. The script writes only the first 100, dropping **53 segments (34.6%)**, without identifying the output as partial. The saved text stops mid-discussion.

Impact: whole-call sentiment and Q&A evidence are biased by an incomplete input. The sample is not a complete transcript despite the assembly comment.

Remediation: paginate to the end of each selected call, verify segment continuity/deduplication and expected counts, and persist source revision/completeness metadata. If partial sampling is intentional, label and limit its use explicitly.

Evidence: [live next-page response](.agent-team/evidence/audit-20261002/upstream-next-page.json), [verification summary](.agent-team/evidence/audit-20261002/upstream-verification.json), [dataset](https://huggingface.co/datasets/finosfoundation/EarningsCallTranscript).

### F4 — P1: Missing benchmark bars silently change the event horizon — **RESOLVED (see remediation)**

Location: [src/ecsa/events/study.py:39](src/ecsa/events/study.py#L39) and [src/ecsa/events/calendar.py:39](src/ecsa/events/calendar.py#L39).

The calendar counts observed SPY bars. Removing one bar moves an event's +10 session from **February 14 to February 15**, changes CAR from **0.01000000 to 0.01032784**, and still produces a valid label. This contradicts the fixed trading-session window in guidelines.md and the promise that missing observations invalidate incomplete windows.

Remediation: align to scheduled exchange sessions, preserve absent bars as missing, and invalidate incomplete return windows instead of compressing time. Evidence: [core-other.json](.agent-team/evidence/audit-20261002/core-other.json); [reproducer](.agent-team/evidence/audit-20261002/ecsa-core-other.py.txt). Existing production defect; frequency on actual datasets not measured.

### F5 — P2: The evaluation script reports process success after failure — **RESOLVED (see remediation)**

Location: [scripts/test_analyzer.py:61](scripts/test_analyzer.py#L61), lines 61–65.

A controlled scorer-construction exception prints `Pipeline error: AUDIT_INFERENCE_FAILURE` but exits normally with code 0. Empty feature output also returns normally. These are distinct branches of the same invalid-success contract.

Impact: automation can report a passing evaluation with no valid results. Pytest's configured `testpaths = ["tests"]` does not collect this script; its filename does not establish automated regression coverage.

Remediation: propagate failures or exit nonzero, treat empty features as failure, and add explicit integration assertions. Evidence: [script-contracts.json](.agent-team/evidence/audit-20261002/script-contracts.json); [reproducer](.agent-team/evidence/audit-20261002/ecsa-script-contracts.py.txt).

### F6 — P2: Strategy IC includes features unavailable until after the outcome — **RESOLVED (see remediation)**

Location: [src/ecsa/pipeline.py:255](src/ecsa/pipeline.py#L255), [src/ecsa/signals/metrics.py:42](src/ecsa/signals/metrics.py#L42) and [src/ecsa/events/stats.py:151](src/ecsa/events/stats.py#L151).

Four increasing scores arriving after all outcomes yield **zero accepted signals, four timing rejections, but strategy IC = 1.0**. Trading rejects the events correctly; the predictive-looking strategy metrics still include them.

Remediation: restrict predictive strategy evaluation to features available at the defined decision cutoff. A separate, clearly labeled retrospective correlation may include later information. Evidence: [core-other.json](.agent-team/evidence/audit-20261002/core-other.json). Existing production defect; not evidence of an executed-trade leak.

### F7 — P2: Hit rate uses different accounting from the actual backtest — **RESOLVED (see remediation)**

Location: [src/ecsa/signals/metrics.py:31](src/ecsa/signals/metrics.py#L31), lines 31–40; compare backtest.py's daily target-weight rebalancing.

With one completed 5% short, prices `100 → 200 → 99`, and zero costs, the backtest finishes at equity **0.9739875** while metrics report **100% hit rate** and **1% average win**. Metrics use endpoint buy-and-hold returns; the strategy rebalances daily.

Remediation: calculate trade statistics from the same position/accounting ledger or clearly label endpoint-return diagnostics separately. The specification does not explicitly define the accounting convention, but the measured inconsistency is confirmed. Evidence: [core-other.json](.agent-team/evidence/audit-20261002/core-other.json). Existing production defect.

### F8 — P2: Required lint gate fails — **RESOLVED (see remediation)**

There are **183 current findings**, including 6 across the three scripts and 177 in `.ua` helpers. The reported two-script addition also omits `scripts/download_hf_data.py`, which is absent from the prior source snapshot. Fix script imports/style and deliberately decide the treatment of generated tooling; do not suppress errors merely to claim success. The original report's assumed pytest result was not evidence; this audit supplies an actual passing run. Production type checking passes independently even though `make check` stops at Ruff.

## Added, changed, removed and discarded

- **Added:** source/policy hashes, exact command logs, live dataset/model provenance, isolated demo/build artifacts, browser captures and deterministic defect reproducers under `.agent-team/evidence/audit-20261002/`.
- **Changed:** this report replaces the unsupported “success / no bugs” conclusion with measured results; current HANDOFF.md records audit completion. Originals are archived above.
- **Removed:** no production behavior or existing tests.
- **Discarded:** no tests. Diagnostic scripts are artifacts, not a newly green test suite. Their assertions sometimes verify reproduction of the current bug, explicitly labeled in results.
- **Harness corrections:** initial wheel smoke omitted required configuration; rerun used the project's YAML. Initial browser capture was too early, and a later wait targeted a hidden tab; final check waited for attached charts and rendered plot SVGs before capture. These were audit-harness errors, not product defects. No application code or test expectations were altered.

## Numbers, capabilities unavailable and not done

Runtime: Python 3.11.16; pytest 9.1.1; Ruff 0.16.9; mypy 1.20.2; torch 2.2.2; transformers 4.49.0; Streamlit 1.64.0; Playwright 1.63.0. [Installed versions](.agent-team/evidence/audit-20261002/versions.json). Demo timings are repeatability observations, not a controlled performance benchmark. [Final integrity check](.agent-team/evidence/audit-20261002/final-integrity.json).

Coverage and mutation: **not measured**; tools not installed and no project-required gate exists. No coverage gain or mutation score claimed. No new behavioral test was removed or weakened. Initial network/localhost restrictions were resolved with approved execution; PyArrow's sandbox sysctl warnings did not fail tests. `.venv` has no pip module; installed `uv pip check` provided the dependency check.

No sentiment benchmark, real-data end-to-end trading validation, SEC/vendor/LLM live integration, dependency advisory audit, exhaustive accessibility review or representative performance benchmark was performed. The legacy `download_hf_data.py` path was inspected but not executed: it installs `datasets` at runtime, and the report explicitly replaced that path. The claimed “zero-dependency” downloader actually imports `requests` and attempts a pip install when missing; it is only independent of the `datasets` library.

All F1–F8 were open at audit time and have since been **resolved** under the manual
feature+quality remediation recorded at the top of this report. The remediation
prioritized authentic/complete input data and the two point-in-time/calendar defects,
then failure signaling, evaluation/accounting and lint. Validated sentiment accuracy
(labeled benchmark, real-data end-to-end trading validation, live integrations) remains
outstanding and out of scope for this fix.

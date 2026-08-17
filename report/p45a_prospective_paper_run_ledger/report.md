# P45A prospective paper run ledger and lifecycle

The P45A paper-operation lifecycle turns the source-agnostic P43/P44 two-phase
workflow into an append-only multi-run ledger system capable of accumulating
prospective evidence.

## Architecture

* **Pregame Run Creation / Freeze**: accepts normalized pregame inputs, enforces
  temporal guards, freezes Champion zero-EV decisions before first pitch, and
  derives deterministic logical run identity.
* **Postgame Settlement**: consumes immutable frozen runs and independent
  normalized final results. Supports partial settlement, rejects tampered or
  conflicting result authority, and updates lifecycle state (FROZEN →
  PARTIALLY_SETTLED → SETTLED).
* **Append-Only Ledger & Forward Evidence Isolation**: finalized records append
  exactly once. Historical rehearsal records are strictly segregated from
  prospective forward statistics (`FORWARD_SAMPLE_COUNT = 0`).
* **Cumulative Descriptive Forward Summary**: maintains running counts, wins,
  losses, units risked, net units, descriptive ROI, and drawdown strictly over
  forward-paper samples.

## Historical rehearsal parity

* Universe: 65 (62 eligible decisions, 3 NO_MARKET exclusions)
* Pregame: 22 BET, 40 PASS, 0 settled (FROZEN)
* Settlement: 22 settled BET, 40 settled PASS, W14 / L8 / P0, 22.0 units risked,
  +5.90 net paper units, descriptive ROI 0.26818181818181818181818181818181818181818181818182 (SETTLED)
* Feedback: 62 rows
* Forward sample count: 0 (isolated)
* Network required: false
* P35A / Live acquisition / TLS: untouched / NOT RUN

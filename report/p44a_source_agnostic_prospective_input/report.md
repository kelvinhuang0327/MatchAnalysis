# P44A source-agnostic prospective input boundary

The P43 two-phase paper workflow now consumes a source-independent
normalized pregame bundle and an independent normalized final-result
bundle. Historical P37/P39 artifact paths live only in the historical
adapter. Frozen P35A, TLS, and live acquisition were not touched.

## Historical adapter rehearsal

Pregame freeze: 62 decisions, 22 BET, 40 PASS, 0 settled.
Postgame settle: 22 settled BET, W14 / L8 / P0, units risked 22.0,
net +5.90, feedback 62. Decision fingerprints match committed P43
`pregame_summary.json`. Network required: false.

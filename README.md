# MatchAnalysis

MatchAnalysis is a minimal Python modular-monolith foundation for describing
matches and their evidence. It is paper-only and diagnostic-only: it does not
place, settle, or otherwise support real betting, and it is not production
ready.

## Architecture

Dependencies point inward:

```text
interfaces
    ↓
application
    ↓
core and baseball domain

infrastructure
    ↓
application ports and domain

core and domain
    ↓
Python standard library only
```

The `core` and `baseball.domain` packages cannot depend on application,
infrastructure, or interfaces. Application code cannot depend on
infrastructure or interfaces. Infrastructure may implement application ports,
and interfaces may call application use cases.

Betting-pool is a frozen, read-only reference implementation. Future migration
must proceed capability by capability, beginning with characterization and
ending with explicit parity evidence. No legacy capability is migrated by this
bootstrap.

## Run the checks

Python 3.14.4 was used to verify this bootstrap.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

## Explicit exclusions

This foundation contains no providers, scheduler, Telegram integration, live
transport, database, frontend, model activation, expected-value (EV) or Kelly
logic, or real-money execution.

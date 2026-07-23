# ADR-0001: Modular monolith and migration boundary

- Status: Accepted
- Date: 2026-07-23

## Context

The MatchAnalysis repository began as an empty, explicitly authorized
directory. Betting-pool exists as a legacy implementation, but it is not a
source of canonical state for this repository and remains read-only.

## Decision

MatchAnalysis will begin as a Python modular monolith organized around ports
and adapters. Dependencies point inward from interfaces and infrastructure
toward application ports, the domain, and the core. Core and domain code use
only the Python standard library.

Migration will use a capability-by-capability strangler approach. Each
capability must first be characterized, then implemented behind a boundary, and
finally supported by explicit parity evidence. Reports are projections, not
canonical source state.

Point-in-time claims, timestamps, and artifact provenance must be explicit and
must fail closed when required evidence is absent or invalid.

## Consequences

The legacy Betting-pool repository remains a frozen, read-only reference.
Microservices are deferred until independent deployment or scaling needs are
demonstrated. Real betting, production activation, legacy model migration,
providers, databases, schedulers, and live runtime integrations are out of
scope.

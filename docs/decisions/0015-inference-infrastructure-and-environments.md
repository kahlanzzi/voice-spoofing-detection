# 0015. Inference infrastructure, capacity and staging/production environments

- **Status:** Approved (all four; PR #111, 2026-11-10)
- **Date:** 2026-11-09
- **Register:** D53, D54
- **Deciders:** Iheb Zemzemi (Part E owner), Donia Mabrouk (author), Malak Ben Salem, Khalil Dridi
- **AI assistance:** Claude (claude.ai, Claude Opus 5.5) drafted this ADR from the repository state;
  Iheb reviews it.

## Context

- The API runs the analyzer in a worker thread with `analysis_timeout_s`; a timed-out thread cannot be
  killed, so repeated timeouts with a real model could pile up (Entry 116, open item).
- `docker-compose.yml` sizes the API's `/tmp` tmpfs for `x-vsd-concurrent-uploads: 4`, but nothing
  enforces that number (open item, Entry 125). The professor asked for a justified limit.
- Analyses, notes and D47 sessions are stored in SQLite. Live sessions (D51) and two deployed
  environments add concurrent writers; SQLite allows one writer at a time.
- NFR-01's latency target `L_MAX` is still open (SRS Appendix 7.4). The v2 model is not chosen yet
  (D49), so its inference cost is unknown; v1 is a 4-block CNN.
- The professor asked for separate staging and production environments, so that every change is
  tested on staging with realistic conditions before it reaches production. The repository is public,
  and CI already builds and tests the full stack (`checks`, `web`, `e2e`, all required on `main`).
- D47's session cookie is `Secure`, so any deployed environment needs HTTPS.

## Decision

- **D53:** CPU inference by default; a bounded **process** pool of K workers (a timed-out analysis can
  be terminated); a busy service answers `503` with `Retry-After` instead of queueing; at most L live
  sessions; K, L, `analysis_timeout_s` and `L_MAX` come from a load test on the staging host and are
  recorded by decision; PostgreSQL replaces SQLite behind the existing repository interface, and CI
  tests run against PostgreSQL; no message broker while there is one API host.
- **D54:** three environments (dev, staging, production); images built once per `main` commit and
  pushed to GitHub Container Registry; staging deploys automatically after green CI; production deploys
  only from a release tag through a GitHub Environment with a required reviewer; per-environment
  secrets; HTTPS with free certificates; both environments on one free VM, provider chosen and recorded
  in week 1.

Full texts: D53 and D54 in [`README.md`](README.md).

## Consequences

- **Easier:** overload becomes a clear, testable `503` instead of a slow failure; the concurrency
  number in Compose becomes a measured, enforced value; every production release is an image that
  already passed CI and staging; the methodology answer moves from CI to CI/CD.
- **Committed:** a migration path for existing SQLite data and D47 session tables; API tests in CI
  gain a PostgreSQL service container; deployment workflows, environment secrets and a smoke test after
  each deploy; the load-test method and numbers are documented (staging only, synthetic and demo audio).
- **Harder:** one more container (PostgreSQL), TLS certificate renewal, and a VM to keep patched.
  Staging must never hold real personal recordings or real analyst accounts.
- **Rule for later:** a GPU or a second API host is added only when a measurement on staging shows the
  need (D53 (1), (5)), each by its own decision.

## Options considered

1. **GPU inference server.** Rejected as a default: no measured need for a small CNN, and no free
   GPU hosting for a permanent service; kept as a measured fallback.
2. **Keep threads, add a queue with Redis or another broker.** Rejected: threads still cannot be
   killed; a broker adds an extra service with no measured need on one host.
3. **Keep SQLite.** Rejected: one writer at a time with uploads, live sessions and sign-in sessions,
   and a different database dialect from any later multi-host setup.
4. **Deploy from developer machines, or rebuild images per environment.** Rejected: production would
   not run the exact image that passed CI and staging.

## Links

- [Decision register](README.md): D15, D38, D47, D51–D54.
- [ADR 0002](0002-tech-stack.md), [ADR 0011](0011-analyst-sign-in-sessions.md),
  [ADR 0014](0014-live-analysis.md).
- [API design](../api_design.md), [setup](../setup.md), `docker-compose.yml`, `deploy/`,
  `.github/workflows/ci.yml`.

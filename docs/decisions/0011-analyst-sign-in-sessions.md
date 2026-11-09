# 0011. Analyst sign-in: username + password with server-side cookie sessions

- **Status:** Accepted (team decision, 2026-11-01; user-reported). The contract details in
  `docs/api_design.md` §4.8 and the §9 values marked proposed are under team review in this PR.
- **Date:** 2026-11-01
- **Register:** D47 in `README.md` (D46 is used by open PR #67; recheck the next free D-ID against
  `main` before merge)
- **Deciders:** Iheb Zemzemi (Part E owner), Donia Mabrouk, Malak Ben Salem, Khalil Dridi
- **AI assistance:** Claude Code (Claude Opus 5.5) drafted this ADR and the §4.8 contract; Iheb
  reviews both.

## Context

- SRS §7.4 leaves the access-control mechanism open. `docs/architecture.md` §7 listed it as "API
  key vs analyst login, decide before Stage 17". [doc]
- Threat model §10.1: anyone who can submit audio and read the result can tune a fake until it
  passes (oracle risk). The recommendation is that every client is authenticated and rate limits
  apply. §11 item 5 records this as a residual risk until access control is decided. [doc]
- HANDOFF §11 item 1 (open decision 1) and the prototype S1 screen show an analyst ID + password
  form with states for invalid credentials, access denied, temporarily locked, sign-in service
  unavailable and expired session. Password recovery and sign-out confirmation depended on the
  decision. [doc]
- The web app is same-origin with the API through nginx; no CORS (`api_design.md` §2, ADR 0008). [doc]
- The API already stores results and notes in SQLite (`api_design.md` §7). [doc]
- Notes had no author field until this decision (`api_design.md` §4.6). [doc]

## Decision

Open decision 1 is decided as **Option A: username + password, with server-side sessions in a
cookie**. Accounts are created only by an admin command (no self-signup), passwords are hashed with
argon2id, sessions live in SQLite and the browser holds only a random session ID in an `HttpOnly;
Secure; SameSite=Strict` cookie. The full contract is `docs/api_design.md` §4.8.

## Consequences

- Every API route needs a session except `POST /api/session` and `GET /api/health`, including
  `GET /status` and `GET /limits`. This addresses threat model §10.1: only signed-in analysts can
  use the detector.
- Requests that change data need an allowed `Origin` (CSRF). `SameSite=Strict` is the second line
  of defence.
- Session IDs are stored as SHA-256 hashes. Sign-out deletes the session. A password change or a
  disabled account ends all of the account's sessions. Idle and absolute limits apply.
- **Accepted risk, lockout as denial of service:** lockout is per username, so anyone who knows a
  username can lock that analyst out for `lockout_minutes` by failing on purpose. nginx's per-IP
  limit only slows this, and an attacker with several IPs is not slowed. The remedy is an admin
  `python -m api.admin unlock`. The team accepts this knowingly, because per-username lockout is
  what stops slow guessing against one account.
- **Failure counts are keyed by HMAC-SHA-256 with a server secret** (`VSD_SIGNIN_HMAC_KEY`,
  environment only, never in the repo). A plain SHA-256 of the typed username would let anyone
  holding a leaked database file guess offline a password typed into the username field. The cost
  is one more secret to provision. Losing or changing it only resets lockout counts.
- `GET /api/health` is the only route besides sign-in without a session. It carries no data
  (`{"ok": true}`) so that the container health check does not need credentials.
- A per-IP rate limit on sign-in lives in nginx, so the API never handles client IPs (§1). In
  development, without nginx, only lockout applies.
- Password recovery means asking an administrator; v1 has no password-change endpoint. A new
  dependency, `argon2-cffi`, is pinned in Step 7b and must install on Python 3.13 and 3.14 (D15).
- Notes record the account that last saved them (`updated_by`).
- The minimum password length is 15, following NIST SP 800-63B-4 §3.1.1.2 for a single-factor
  password (proposed, team to confirm). That section also requires a blocklist check of common or
  compromised passwords, which this contract does not include yet; it is an open item.
- To verify later: the Stage 18 S1 screen drops its "open product decision" banner, and the
  threat model §10.1 / §11 item 5 wording is updated once sign-in is implemented.

## Links

- D47 in [README.md](README.md); `docs/api_design.md` §4.6, §4.8, §6, §7, §9, §10
- `docs/design/HANDOFF.md` §1 (S1), §11 item 1; `docs/threat_model.md` §10.1, §11
- [ADR 0008](0008-web-frontend-react.md) (D38, the web app)
- NIST SP 800-63B-4 §3.1.1.2, <https://pages.nist.gov/800-63-4/sp800-63b.html>

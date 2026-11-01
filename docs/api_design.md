# API design: backend ↔ analyst web app (Stage 17)

- **Status:** Approved (PR #65, Donia, 2026-10-30).
- **Owner:** Iheb Zemzemi (Part E). **Reviewers (D26):** Donia Mabrouk, Malak Ben Salem, Khalil Dridi.
- **Serves:** the Stage 17 API (`api/`, FastAPI) and the Stage 18 web app (`web/`, React, D38).
- **Sources:** [HANDOFF.md](design/HANDOFF.md) §1, §11, §12; the
  [prototype v4](design/analyst-console-prototype.html); Protocol v1.0 §11
  ([experimental_protocol.md](experimental_protocol.md)); [architecture.md](architecture.md) §4–5;
  [ADR 0008](decisions/0008-web-frontend-react.md).

This note changes nothing in the §11 response, the ModelBundle or the risk policy. It turns
HANDOFF §11 item 10 (endpoints beyond §11) into a concrete proposal. Items marked
**Open decision** are not decided here.

## 1. Principles

- The API is the only source of verdicts, upload limits and statistics. The web app computes none
  of them and holds no risk-policy logic (ADR 0008 §3, §4, §8; NFR-08).
- Every result is decision support, not proof of fraud. The `notice` field says so on every result.
- Errors never carry a verdict: no `prediction`, `confidence` or `risk_level` in any error (NFR-06).
- No audio, original file name or IP address is ever stored (FR-09, NFR-04).

## 2. Conventions

- **Base path:** every endpoint is under `/api` (for example `POST /api/analyses`). The web app
  reaches it on the same origin through nginx, which serves the app and proxies `/api` to the
  `api` container. No CORS is needed or enabled.
- **Format:** JSON, field names in `snake_case`. File upload is the only multipart request.
- **Times:** UTC, ISO 8601 with `Z` (for example `2026-10-30T09:14:05Z`). **Dates** in queries:
  `YYYY-MM-DD`, read as days in the statistics time zone (section 9).
- **Request IDs:** made by the server, `req_` + 32 lowercase hex characters from a random UUID
  (the prototype shows a shortened form, `req_8fa98c02`). The browser never makes one.
- **Verdict values:** `synthetic`, `uncertain`, `authentic` (the §11 `prediction`).
  **Risk values:** `LOW`, `MEDIUM`, `HIGH`.

## 3. Endpoints at a glance

Paths are relative to `/api`.

| Method | Path | Purpose | Prototype screen |
|---|---|---|---|
| GET | `/status` | Service and bundle status | S10 System status; rail status line |
| GET | `/limits` | Accepted formats, size and duration caps | S2 Upload, S3 File selected |
| POST | `/analyses` | Analyze one recording | S4 Processing, S5 Error, S6 Result |
| GET | `/analyses` | Stored results, filtered and paged | S7 History |
| GET | `/analyses/{request_id}` | One stored result with its note | S8 Analysis details |
| PUT | `/analyses/{request_id}/note` | Save or remove the analyst note | S6 and S8 note panel |
| GET | `/stats` | Operational figures for a period | S9 Overview |
| POST | `/session` | Sign in (section 4.8) | S1 Sign in |
| GET | `/session` | Who am I | S1 on load |
| DELETE | `/session` | Sign out | Rail sign-out, D1 |
| GET | `/health` | Container liveness, no data | none |

## 4. Endpoints

### 4.1 `GET /status`

Request: no parameters.

Response `200`:

```json
{
  "state": "ready",
  "checked_at": "2026-10-30T09:14:05Z",
  "checks": { "analysis": "available", "history_and_stats": "available" },
  "versions": { "model_version": "mock", "profile_version": "preprocessing-v1", "policy_version": "risk-v1" },
  "bundle_manifest_sha256": null,
  "deferral_on_fit": 0.0993,
  "mock": true
}
```

- `state`: `ready`, `degraded` or `unavailable`. `unavailable` when analysis is unavailable.
  **Open decision 7:** the rule for `degraded` (prototype proposal: analysis works, History or
  statistics do not).
- `checks.history_and_stats` is `available` when a read-only query on the §7 database succeeds
  within a short timeout. The check never writes. A failure is reported here, never as an error.
- The connection check on S10 is done by the browser: no answer means "cannot reach the service".
- `versions`, `bundle_manifest_sha256`, `deferral_on_fit` and `mock` come from the analyzer
  (section 8). With the MockAnalyzer the manifest hash is `null`.
- The endpoint reports problems in its body; it answers `200` even when `state` is `unavailable`.
- Needs a session (section 4.8).

Status codes: `200`; `401`; `500 internal_error`.

### 4.2 `GET /limits`

Request: no parameters.

Response `200`:

```json
{
  "formats": ["wav", "flac", "mp3"],
  "max_size_bytes": 20971520,
  "max_duration_s": 60,
  "audio_deletion": "Deleted when processing finishes"
}
```

- All four values come from config. **Open decision 2:** the real limits; the values above are the
  prototype mock.
- Size is in bytes so the browser compares it with `File.size` directly (the prototype mock used
  `max_size_mb`).
- The browser runs the same checks before upload, using these values only (NFR-08). If this call
  fails, the app keeps analysis off (S2 "limits unavailable").
- Needs a session (section 4.8).

Status codes: `200`; `401`; `500 internal_error`.

### 4.3 `POST /analyses`

Request: `multipart/form-data` with one part, `file`. The server reads only the extension from the
part's file name, then discards the name. The part's content type is ignored.

Response `201`, header `Location: /api/analyses/{request_id}`, body = **exactly the 8 §11 fields**:

```json
{
  "prediction": "uncertain",
  "confidence": 0.62,
  "risk_level": "MEDIUM",
  "model_version": "mock",
  "profile_version": "preprocessing-v1",
  "policy_version": "risk-v1",
  "request_id": "req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10",
  "notice": "Decision support only. This result estimates whether the recording is synthetic. It does not prove fraud and does not verify identity."
}
```

**Processing order.** The `Origin` and session checks (section 4.8) run first, before the body is
read; their errors carry `request_id: null`. The request ID is assigned next, so every upload error
below carries it.

| Step | Check | On failure |
|---|---|---|
| 1 | Extension (lower case) is in `formats` | `415 unsupported_format` |
| 2 | Size cap (see the first note below); stream the file to a temp file in chunks and stop as soon as it passes `max_size_bytes` | `413 file_too_large` |
| 3 | File is not empty (more than 0 bytes) | `422 empty_file` |
| 4 | Decodes with `soundfile`; a file that decodes to 0 frames counts as empty | `422 corrupt_audio`; `422 empty_file` for 0 frames |
| 5 | Duration (frames ÷ sample rate) ≤ `max_duration_s` | `422 recording_too_long` |
| 6 | `analyzer.analyze(path)` within `analysis_timeout_s` | `503 analysis_unavailable`, `504 analysis_timeout`, `500 internal_error` |
| 7 | Store the analysis record (section 7) | `500 internal_error` |

- **Size protection (step 2):** Starlette reads the whole multipart body before the handler runs,
  so the handler's streamed check cannot stop a large upload on its own. The real protection is
  nginx's body-size limit plus a `Content-Length` check before the body is parsed. The streamed
  check in the handler is the second line of defence. `Content-Length` counts the whole multipart
  body, so the `Content-Length` check and the body limit allow 64 KiB of multipart framing above
  `max_size_bytes`; the streamed check enforces the exact cap on the file bytes.
- The temp file has a random name in the configured temp folder. It is deleted when the request
  ends, on success and on every error (NFR-03). **Open decision 4:** what happens if deletion fails.
- No record is stored when any step fails. A verdict is returned only after the record is stored,
  so every shown result can be found again in History.
- On `504` the API stops waiting and stores nothing, even if the analyzer finishes later.
- Stage 18: the prototype's timeout screen must not say a late result will appear in History;
  with this contract a server timeout stores nothing.
- **Open decision 3:** how recordings longer than 2 s are handled (the model works on 2 s clips).
  That rule lives inside the analyzer, not in this endpoint.

Status codes: `201`; `400 invalid_request` (no `file` part); `413`; `415`; `422`; `401`/`403`
(section 4.8); `503`; `504`; `500`.

### 4.4 `GET /analyses`

Query parameters:

| Name | Meaning | Default |
|---|---|---|
| `verdict` | Repeatable: `?verdict=synthetic&verdict=uncertain`. None given = all verdicts | all |
| `from`, `to` | Inclusive day range (`YYYY-MM-DD`) | no limit |
| `q` | Part of a request ID, case-insensitive | none |
| `page` | Page number, from 1 | 1 |
| `page_size` | Items per page, 1 to `page_size_max` | 10 |

Response `200`:

```json
{
  "items": [ { "analysis": { "...8 §11 fields...": "" }, "created_at": "2026-10-30T09:14:05Z", "note": null } ],
  "total": 37,
  "page": 1,
  "page_size": 10,
  "counts": { "synthetic": 9, "uncertain": 6, "authentic": 22 }
}
```

- `items`: stored results (section 5), newest first.
- `total`: number of results matching all filters.
- `counts`: results per verdict matching `from`, `to` and `q`, **ignoring** the verdict filter, so
  each verdict chip shows how many results it would add (as in the prototype).
- A page past the last one returns `200` with empty `items` and the correct `total`.
- Newest first means by `created_at`; results stored in the same second come last-stored first.
- `from` and `to` are exactly `YYYY-MM-DD` and are read as UTC days while `stats_time_zone` is `UTC`
  (open decision 6); no time-zone conversion is done. Any other `stats_time_zone` stops the API at start.
- `q` matches anywhere in the request ID, ignoring case; an empty `q` is no filter; a `q` that matches
  nothing returns empty `items`, not an error.
- `counts` always lists all three verdicts, with 0 when there are none. A repeated `verdict` counts once.
  Unknown query parameters are ignored. If `page_size_max` is below 10, the default `page_size` is `page_size_max`.
- `page < 1`, `page_size` out of range, an unknown `verdict`, a malformed date, or `from` after
  `to` → `400 invalid_request`.
- Items include `confidence` because they carry the whole §11 object. The History table does not
  show it (HANDOFF §10).

Status codes: `200`; `400`; `401`/`403`; `500`.

### 4.5 `GET /analyses/{request_id}`

Response `200`: one stored result (section 5). Used by S8, and by S6 if the page is reloaded.

- An ID that does not have the request-ID form (section 2) also answers `404 not_found`.

Status codes: `200`; `404 not_found` (unknown ID: S8 "unknown request ID"); `401`/`403`; `500`.

### 4.6 `PUT /analyses/{request_id}/note`

Request body: `{ "text": "Called back on the known number; caller confirmed." }`

Response `200`: `{ "note": { "text": "…", "updated_at": "2026-10-30T09:20:11Z", "updated_by": "analyst01" } }`

- One note per analysis; a new `PUT` replaces it.
- Empty text (after trimming spaces) removes the note; the response is `{ "note": null }`.
- Text longer than `note_max_chars` → `400 invalid_request`.
- Spaces and line breaks at the start and end are removed before the length check and before storing;
  length is counted in characters (Unicode code points).
- The body is exactly `{ "text": <string> }`, sent as `application/json` and at most
  12 × `note_max_chars` + 1 KiB bytes; a missing or non-string `text`, any other field, or a larger body
  → `400 invalid_request`.
- `updated_at` is set by the server at every save. Removing a note that does not exist also answers
  `{ "note": null }`.
- The note records the account that last saved it: `updated_by` is that account's username (section
  4.8). A new `PUT` by another account replaces the text and the author. Removing the note keeps no
  author. `updated_by` is `null` for notes saved before sign-in existed. The note text is never logged.
- The note is stored in its own table and never changes the analysis record (FR-11).

Status codes: `200`; `400`; `404 not_found`; `401`/`403`; `500`.

### 4.7 `GET /stats`

Query parameters: `from`, `to` (inclusive days). Without them: the last 7 days, including today.

Response `200`:

```json
{
  "from": "2026-10-24",
  "to": "2026-10-30",
  "time_zone": "UTC",
  "generated_at": "2026-10-30T09:14:05Z",
  "today": { "date": "2026-10-30", "total": 12 },
  "total": 214,
  "counts": { "synthetic": 41, "uncertain": 23, "authentic": 150 },
  "uncertain_share": 0.1075,
  "median_processing_ms": 840,
  "awaiting_note": { "total": 18, "recent": [ { "analysis": { "...": "" }, "created_at": "…", "note": null } ] },
  "days": [ { "date": "2026-10-24", "synthetic": 5, "uncertain": 3, "authentic": 20, "total": 28 } ],
  "monitoring_marker": { "deferral_on_fit": 0.0993, "policy_version": "risk-v1", "mock": true }
}
```

- `awaiting_note`: synthetic or uncertain results with no note; `recent` holds the 6 newest.
- `days` lists every day of the period, with zeros for empty days.
- `uncertain_share` = uncertain ÷ total; `null` when total is 0.
- A figure the service cannot compute is `null`; the app shows "Not available" for that figure
  only (S9 "one figure unavailable").
- `monitoring_marker` comes from the analyzer (section 8), never from a constant in code
  (ADR 0008 §8).
- **Open decision 6:** the time zone used to count days (config; UTC for now).
- No accuracy, EER or other quality metric, no per-analyst figure, nothing from DEEP-VOICE
  (ADR 0008 §8).
- Malformed date or `from` after `to` → `400 invalid_request`.
- Only `from` given: `to` is today. Only `to` given: `from` is 6 days before `to`.
- A period longer than 366 days, or one that would start before 0001-01-01, → `400 invalid_request`.
- `today` is always the current day in `stats_time_zone`, with its total across all verdicts, whatever
  `from` and `to` are. Its `date` is the date of `generated_at`.
- `total`, `counts`, `days`, `uncertain_share`, `median_processing_ms` and `awaiting_note` cover the period
  only. For the same `from` and `to`, `total` and `counts` equal those of `GET /analyses`.
- `uncertain_share` is rounded to 4 decimal places. `median_processing_ms` is the middle `processing_ms` of
  the period, or the mean of the two middle values rounded to the nearest integer (halves up); `null` when
  total is 0. `awaiting_note.recent` is newest first, ordered as in History.
- Stats and History count every stored result, mock or real; `is_mock` is not a filter.
  `monitoring_marker.mock` describes the analyzer now loaded. Before the real ModelBundle is served, the
  store starts empty (a fresh database file) or mock results and their notes are removed (`analysis_note`
  has a foreign key to `analysis_record`), so the figures never mix the two.
- Dates use the §4.4 rules (exactly `YYYY-MM-DD`, a real date). Unknown query parameters are ignored. All
  figures are read in one consistent snapshot. A failed read → `500 internal_error`.

Status codes: `200`; `400`; `401`/`403`; `500`.

### 4.8 Sign-in and sessions

**Decided (D47, [ADR 0011](decisions/0011-analyst-sign-in-sessions.md), team, 2026-11-01):** open
decision 1, Option A: username + password, with a server-side session held in a cookie. Accounts are
created by an administrator; there is no self-signup. Values marked *proposed* in section 9 are for
the team to confirm in review.

#### Endpoints

| Method | Path | Purpose | Needs a session |
|---|---|---|---|
| POST | `/session` | Sign in | no |
| GET | `/session` | Who am I | yes |
| DELETE | `/session` | Sign out | yes |
| GET | `/health` | Liveness for the container health check | no |

**`GET /health`.** Response `200`, always exactly `{ "ok": true }` while the process is up. It needs no
session and no `Origin`, reads no database, calls no analyzer, returns no version, hash or count, and
writes no log line. It is liveness only: readiness stays in `GET /status`, which needs a session. The Step 8
Docker health check calls it.

**`POST /session`.** Request body, `application/json`, exactly:

```json
{ "username": "analyst01", "password": "correct horse battery staple" }
```

Response `200`, with `Set-Cookie` (see Cookie below):

```json
{ "username": "analyst01", "signed_in_at": "2026-11-01T09:00:00Z", "expires_at": "2026-11-01T21:00:00Z" }
```

- `username` is compared in lower case. `password` is never trimmed or changed.
- A missing or non-string field, an empty field, any other field, a `password` longer than
  `password_max_chars` or a body over 4 KiB → `400 invalid_request`. The app checks for empty
  fields first (S1 field errors), so this is a second check.
- Checks run in this order, and the first one that fails gives the answer:
  1. The username is locked (Lockout below) → `429 account_locked`. The password is not checked.
  2. Unknown username or wrong password → `401 invalid_credentials`. For an unknown username,
     the password is still verified against a fixed dummy argon2id hash, so both cases take the
     same time. The response body is identical in both cases.
  3. Correct password, but the account is disabled → `403 access_denied`.
  4. Success: the failure count for this username is reset, and a new session is created. A
     session cookie already sent with the request is deleted first, so the session ID is always
     new (no session fixation).
- The account store cannot be read or written → `503 sign_in_unavailable`.
- `expires_at` is the absolute end of the session. The idle limit can end it earlier.

**`GET /session`.** Response `200`: the same shape as sign-in. The app calls it on load to see whether
a session already exists. No session → `401 session_expired`.

**`DELETE /session`.** Deletes the session row and clears the cookie. Response `204`, no body. No
valid session → `401 session_expired`; the cookie is cleared anyway. Sign-out takes effect at once
and asks for no confirmation. The unsaved-note dialog (D1) belongs to the app.

#### Which routes need a session

Every route under `/api` needs a valid session except `POST /session` and `GET /health`. **This
includes `GET /status` and `GET /limits`.** The sign-in screen (S1) calls only `GET /session` and
`POST /session`. Limits are needed from S2 on, and the status line and S10 from the rail, so both are
read after sign-in. `/status` exposes the model versions and the bundle hash, and nothing calls it
before sign-in. The container health check uses `GET /health`, which carries no data.

Session and Origin checks run in middleware, before the route and before any request body is
read. An upload without a session is therefore refused before it is parsed, and before a request
ID is assigned (`request_id: null`).

#### Cookie

`Set-Cookie: vsd_session=<id>; HttpOnly; Secure; SameSite=Strict; Path=/api`, with no `Domain` and
no `Max-Age` (it is a browser-session cookie; the server enforces the lifetimes).

- The session ID is 32 random bytes from `secrets.token_urlsafe(32)` (43 URL-safe characters,
  256 bits).
- **Stored hashed:** the database keeps only the SHA-256 of the ID (hex). Anyone who reads the
  database file, a backup or the volume gets no usable session. A fast hash is enough because the
  ID is 256 random bits, not a password. Lookup by hash keeps an index.
- The `__Host-` prefix is not used because it requires `Path=/`.
- `Secure` stays on in every environment. Browsers accept it on `http://localhost`, and tests use
  an `https://` base URL.

#### Sessions

Sessions are stored in SQLite (`session` table, section 7).

- **Idle limit** `session_idle_minutes`: a session unused for longer than this is ended.
  `last_seen_at` is updated on every request that has a valid session.
- **Absolute limit** `session_absolute_hours`, counted from sign-in. It is never extended.
- An ended session, an unknown ID or no cookie → `401 session_expired`, and the response clears
  the cookie. Expired rows are deleted when they are found and at each sign-in.
- Sign-out deletes the session row.
- Setting a new password, or disabling an account, deletes **all** sessions of that account.
- Several sessions per account are allowed (for example, two browsers).

#### Accounts

- Accounts are created only with an admin command on the server, `python -m api.admin`.
  Subcommands: `create-account <username>`, `set-password <username>`, `disable <username>`,
  `enable <username>` and `unlock <username>`. The password is typed twice at a hidden prompt
  (`getpass`), never given on the command line or in a file. There is no signup endpoint and no
  password-change endpoint in v1.
- Usernames: 3 to 64 characters from `a–z 0–9 . _ -`, stored in lower case, unique, never renamed
  or deleted. An account that is no longer needed is disabled.
- Passwords: at least `password_min_chars` and at most `password_max_chars` characters (Unicode
  code points), with no other composition rules. They are hashed with **argon2id** (`argon2-cffi`,
  pinned in Step 7b; its RFC 9106 low-memory profile: t = 3, m = 64 MiB, p = 4, 16-byte salt). A
  stored hash with older parameters is rehashed at the next successful sign-in.
- **Recovery:** the analyst asks an administrator, who runs `set-password`. This ends all of the
  account's sessions.

#### Lockout and rate limit

- **Lockout (API):** after `lockout_threshold` consecutive failed sign-ins for one username, that
  username is locked for `lockout_minutes`. While it is locked, every sign-in for it answers
  `429 account_locked`, even with the right password. The count resets on success, when a lock
  ends, or when the last failure is more than `lockout_minutes` old. An admin `unlock` also resets it.
- The count is kept per submitted username, known or not, under the **HMAC-SHA-256** of the
  lower-case username, keyed with a server secret (`sign_in_failure`, section 7). An unknown
  username therefore locks the same way, and lockout does not reveal whether a username exists.
  The typed text is never stored. A plain hash would let anyone holding a leaked database file
  guess offline any password that was typed into the username field. Without the key that is not
  possible.
- The key `VSD_SIGNIN_HMAC_KEY` (at least 32 random bytes, hex) is read from the environment: `.env`
  locally (its name only in `.env.example`) and a secret in deployment. It is never in the repo,
  `configs/` or a log. The API refuses to start without it. Changing the key only resets the failure
  counts.
- **Accepted risk:** anyone who knows a username can lock that analyst out for `lockout_minutes` by
  failing on purpose. nginx's per-IP limit only slows this down. The remedy is an admin `unlock`. The
  team accepts this knowingly (D47, ADR 0011).
- **Rate limit (nginx):** nginx limits `POST /api/session` per client IP (`signin_rate_per_minute`,
  `signin_burst`) and answers over-limit requests itself with `429 too_many_requests` in the
  section 6 shape. The IP stays in nginx memory only; the API never sees or stores it (§1). nginx
  stops floods before they reach argon2, and lockout stops slow guessing against one account. In
  production the API runs only behind nginx. In development there is no IP limit, and lockout
  still applies.

#### CSRF

- Every request that changes data must carry an `Origin` header equal to one of `allowed_origins`:
  `POST /analyses`, `PUT /analyses/{request_id}/note`, `POST /session` and `DELETE /session`. A missing
  or other `Origin` → `403 origin_not_allowed`, checked before the session.
- `SameSite=Strict` is a second line of defence. GET routes change nothing (the `last_seen_at`
  update is not a user action).
- No CORS: the app is same-origin through nginx (section 2).

#### Errors for S1

| S1 state | Answer |
|---|---|
| Invalid credentials | `401 invalid_credentials`: same body for an unknown username and a wrong password |
| Access denied | `403 access_denied`: correct password, account disabled |
| Temporarily locked | `429 account_locked` |
| Sign-in service unavailable | `503 sign_in_unavailable`. nginx's `429 too_many_requests` and a network failure also show this state. |
| Expired session (notice) | `401 session_expired` from any other route |

All of them use the section 6 shape with `request_id: null`, as do `401 session_expired` (no valid
session) and `403 access_denied` (signed in, not allowed; reserved earlier and kept).

#### Privacy and logging

- Never logged, and never in a response body: passwords, session IDs, their hashes, the `Cookie`
  header and the request body of `POST /session`. The session ID leaves the server only in `Set-Cookie`. A failed sign-in is logged without the username, because a
  password is sometimes typed into that field. A successful one is logged with the account ID only.
- The §1 and §7 rules do not change. The `session` table has no IP or user-agent column, and no
  audio, file name or IP is stored anywhere.

## 5. Shapes

| Shape | Fields | Notes |
|---|---|---|
| `AnalysisResponse` | `prediction`, `confidence`, `risk_level`, `model_version`, `profile_version`, `policy_version`, `request_id`, `notice` | Exactly §11. Mirrored by the web app's TypeScript type (ADR 0008 §3). |
| `StoredAnalysis` | `analysis` (an `AnalysisResponse`), `created_at`, `note` (`{text, updated_at, updated_by}` or `null`) | The §11 object stays whole; stored-only fields sit beside it. |
| `Error` | `error.code`, `error.message`, `error.request_id` | Section 6. |

`processing_ms` and the mock flag are stored but not returned per item; they feed `GET /stats`
and `GET /status`.

## 6. Errors

Every error uses one shape:

```json
{ "error": { "code": "corrupt_audio", "message": "The recording could not be decoded.", "request_id": "req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10" } }
```

| Code | HTTP | When | Prototype state |
|---|---|---|---|
| `unsupported_format` | 415 | Extension not in `formats` | S3 check (browser), repeated by the API |
| `file_too_large` | 413 | Body passes `max_size_bytes` | S3 check, repeated by the API |
| `recording_too_long` | 422 | Duration passes `max_duration_s` | S3 check, repeated by the API |
| `empty_file` | 422 | 0 bytes, or decodes to 0 frames | S3 check, repeated by the API |
| `corrupt_audio` | 422 | `soundfile` cannot decode it | S5 Corrupt audio (input) |
| `analysis_unavailable` | 503 | Analyzer not loaded or refusing work | S5 Analysis service unavailable |
| `analysis_timeout` | 504 | Analyzer did not finish in `analysis_timeout_s` | S5 Request timeout |
| `session_expired` | 401 | No cookie, unknown, idle or expired session (section 4.8) | S5 / S1 Expired session |
| `access_denied` | 403 | Correct password but account disabled; or signed in, not allowed (section 4.8) | S1 Access denied |
| `invalid_credentials` | 401 | Unknown username or wrong password, same body for both (section 4.8) | S1 Invalid credentials |
| `account_locked` | 429 | Username locked after failed sign-ins (section 4.8) | S1 Temporarily locked |
| `too_many_requests` | 429 | nginx per-IP rate limit on `POST /api/session` (section 4.8) | S1 Sign-in service unavailable |
| `sign_in_unavailable` | 503 | Account store cannot be read or written | S1 Sign-in service unavailable |
| `origin_not_allowed` | 403 | Missing or other `Origin` on a request that changes data (section 4.8) | none (the app is same-origin) |
| `not_found` | 404 | Unknown request ID or route | S8 Unknown request ID |
| `internal_error` | 500 | Anything unexpected, including a failed store | S5 Unexpected service error |
| `invalid_request` | 400 | Bad query or body (**proposal beyond the agreed list**) | none (the app prevents these) |
| `invalid_request` | 405 | Method not allowed on a known path | none (the app never sends these) |

Rules:

- No error contains `prediction`, `confidence`, `risk_level` or any partial result.
- `request_id` is the upload's ID for errors from `POST /analyses` after the ID is assigned;
  otherwise `null`. A network failure in the browser has no request ID at all (HANDOFF §12).
- `message` is one plain sentence for logs and support. The web app shows its own wording per
  `code` (HANDOFF §8) and never displays raw server text.
- FastAPI's default error bodies (validation `422`, `HTTPException`, unhandled `500`) are replaced
  by this shape. No stack trace or internal path ever reaches the client.
- nginx must not answer with its own error pages for `/api`: its body-size limit is set above
  `max_size_bytes` and its proxy timeout above `analysis_timeout_s`, so the API's codes reach the app.
  The one answer nginx gives itself, its `429 too_many_requests` for `POST /api/session`, uses this
  JSON shape with `request_id: null`.

## 7. Storage

Five tables (SQLite on a volume first; architecture §7, open):

| Table | Columns |
|---|---|
| `analysis_record` | `request_id` (key), `created_at` (UTC), `prediction`, `confidence`, `risk_level`, `model_version`, `profile_version`, `policy_version`, `notice`, `processing_ms`, `is_mock` |
| `analysis_note` | `request_id` (key, refers to `analysis_record`), `text`, `updated_at` (UTC), `updated_by_account_id` (refers to `account`; `null` for notes saved before sign-in existed) |
| `account` | `account_id` (key), `username` (unique, lower case), `password_hash` (argon2id PHC string), `disabled`, `created_at`, `password_changed_at` |
| `session` | `session_hash` (key, SHA-256 hex of the session ID), `account_id` (refers to `account`), `created_at`, `last_seen_at`, `expires_at` (UTC) |
| `sign_in_failure` | `username_hmac` (key, HMAC-SHA-256 hex of the lower-case username, key `VSD_SIGNIN_HMAC_KEY`), `failures`, `last_failure_at`, `locked_until` (UTC) |

- `processing_ms`: from upload received to result ready.
- `is_mock`: true for MockAnalyzer results, so they can be told apart from real ones.
- **Never stored:** audio, the original file name, the client IP address, passwords in clear,
  session IDs in clear, the typed username of a failed sign-in. The file name is not logged either.
- Existing database files gain `analysis_note.updated_by_account_id` through a migration in Step 7b.
- The operational log (FR-20, `service_log` in architecture §4) is not part of this contract.

## 8. Analyzer seam

One interface is the only place the API calls model code:

```text
analyze(path) -> AnalysisOutcome   # prediction, confidence, risk_level,
                                   # model_version, profile_version, policy_version
info() -> AnalyzerInfo             # ready, versions, deferral_on_fit,
                                   # bundle_manifest_sha256, mock
```

- **Now:** `MockAnalyzer`, with `model_version: "mock"` and `mock: true`. It returns fixed,
  plausible outcomes that follow `risk-v1` behaviour (as the prototype's mock data does) and reads
  its `deferral_on_fit` from config, labelled mock.
- **Later:** the team's frozen ModelBundle (Stage 12), loaded and verified by `src/bundle/`, which
  supplies `deferral_on_fit` from its risk policy. The API code does not change.
- The API adds `request_id` and `notice` to the outcome to make the §11 response.
- Analyzer not ready → `503 analysis_unavailable`; over `analysis_timeout_s` → `504`; any other
  exception → `500 internal_error`.

## 9. Configuration

All values come from config (`configs/api.yaml`, written in Stage 17), never from code.

| Key | Meaning | Value now | Status |
|---|---|---|---|
| `formats` | Accepted extensions | `wav`, `flac`, `mp3` | Mock; **open decision 2** |
| `max_size_bytes` | Upload size cap | 20,971,520 (20 MB) | Mock; **open decision 2** |
| `max_duration_s` | Recording duration cap | 60 | Mock; **open decision 2** |
| `audio_deletion` | Deletion statement shown on S2 | "Deleted when processing finishes" | Depends on **open decision 4** |
| `analysis_timeout_s` | How long the API waits for the analyzer | 30 | Proposal |
| `page_size_max` | Largest `page_size` | 50 | Proposal |
| `note_max_chars` | Longest note | 2,000 | Proposal |
| `stats_time_zone` | Time zone for counting days | `UTC` | **Open decision 6** |
| `notice_text` | The §11 `notice` text | prototype wording | Proposal |
| `mock_deferral_on_fit` | Monitoring marker for the MockAnalyzer only | 0.0993 | Mock; replaced by the bundle's value |
| `temp_dir` | Folder for temp uploads | — | Deployment setting |
| `database_path` | SQLite file holding the section 7 tables | `var/api.sqlite` | Deployment setting |
| `session_idle_minutes` | Idle limit of a session | 30 | **Proposed, team to confirm in review** |
| `session_absolute_hours` | Absolute limit, from sign-in | 12 | **Proposed, team to confirm in review** |
| `lockout_threshold` | Consecutive failed sign-ins before a lock | 5 | **Proposed, team to confirm in review** |
| `lockout_minutes` | Lock length, and how long a failure counts | 15 | **Proposed, team to confirm in review** |
| `password_min_chars` | Shortest password | 15 | **Proposed, team to confirm in review** |
| `password_max_chars` | Longest password | 128 | **Proposed, team to confirm in review** |
| `allowed_origins` | Origins accepted on requests that change data | — | Deployment setting |
| `signin_rate_per_minute` / `signin_burst` | nginx `limit_req` on `POST /api/session`, per client IP (nginx config, not `api.yaml`) | 10 / 5 | **Proposed, team to confirm in review** |
| session ID length | Random bytes per session ID (fixed in code, not a config key) | 32 | **Proposed, team to confirm in review** |

`analysis_timeout_s` is not a latency target: **open decision 5** (latency target) stays open.

- `session_idle_minutes`: a shorter limit closes an unattended console sooner. But an analyst away on a call
  for longer than the limit loses the session before saving a note, and must sign in again before the
  save works. A longer limit avoids that and leaves an unattended signed-in console open longer.
- `password_min_chars`: 15 follows NIST SP 800-63B-4 §3.1.1.2 (Password Verifiers): "Verifiers and
  CSPs SHALL require passwords that are used as a single-factor authentication mechanism to be a
  minimum of 15 characters in length". Here the password is the only factor. The same section asks for a
  maximum of at least 64 (128 meets it) and no composition rules (none here). It also requires a
  check against a blocklist of common or compromised passwords. That check is not in this contract;
  it is an open item for the team.
- `VSD_SIGNIN_HMAC_KEY` is a secret, not a config value: environment only (section 4.8).

## 10. Open decisions and questions

From HANDOFF §11, as they touch the API:

1. Authentication and access control: **decided** (D47,
   [ADR 0011](decisions/0011-analyst-sign-in-sessions.md), 2026-11-01); see section 4.8.
2. Real upload limits (section 9).
3. Recordings longer than 2 s (inside the analyzer).
4. Fallback if temp-file deletion fails.
5. Latency target.
6. Time zone for statistics.
7. Definition of "Degraded" (section 4.1).
9. Cancel analysis: no API change. Cancel stops the browser waiting; a request already accepted
   may still finish and be stored, and then shows in History.

For the reviewers of this note:

- **`invalid_request` (400)** is added beyond the agreed error list, for bad queries and bodies.
- **Decision record:** once approved, does this contract need a D-ID in `docs/decisions/README.md`
  (HANDOFF §11 item 10)? This note does not add one.

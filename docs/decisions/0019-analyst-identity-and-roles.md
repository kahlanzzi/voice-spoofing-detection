# 0019. Analyst identity, roles and member management

- **Status:** Approved 2026-11-25 (Iheb, Malak and Khalil in person, user-reported; reviews on PR #167 by Iheb (`kahlanzzi`) and Khalil (`dridi-khalil`); recorded in Entry 220); designed, not shipped: cut line applied by D69 (approved 2026-11-26, PR #170)
- **Date:** 2026-11-25
- **Register:** D68 (amends D47)
- **Author:** Donia Mabrouk
- **Deciders:** Iheb Zemzemi, Malak Ben Salem, Khalil Dridi (approvers); Donia Mabrouk (author)

## Context

- D47 ([ADR 0011](0011-analyst-sign-in-sessions.md)) uses username + password with server-side sessions;
  accounts are created only by an admin command (no self-signup).
- The adopted direction (D67, [ADR 0018](0018-ui-direction-enterprise.md)) has an Administration screen and
  roles.

## Decision

This decision amends D47 and `docs/api_design.md` §4.8 "Accounts". It supersedes the identity and accounts
parts of [ADR 0011](0011-analyst-sign-in-sessions.md); sessions unchanged.

1. Sign-in identifier = **email address** (lower-cased, unique, never reused; replaces the username);
   passwords, argon2id, server-side sessions, the `HttpOnly; Secure; SameSite=Strict; Path=/api` cookie, the
   Origin check, the lockout per submitted identifier (now the email, whether or not an account has it, so
   lockout does not reveal which accounts exist, §4.8) and the nginx per-IP rate limit are unchanged.
2. Still **no self-signup**. Accounts are created by an Administrator, in the Members screen or with `python
   -m api.admin`. **No email is sent** (no free mail service in scope). Passwords stay as D47 v1 (§4.8
   "Accounts", "Recovery"): the Administrator sets the password (on creation and on reset), typed twice
   (Members screen, or the hidden prompt of `api.admin`), and gives it to the person out of band; there is
   still no self-service password-change endpoint. Setting a password ends all of that account's sessions
   (unchanged).
3. Roles, enforced **on the server** for every route (the UI only hides what the role cannot do): **Viewer**
   reads Overview, History, Results, Flagged reviews and notes; **Analyst** = Viewer + upload, live analysis,
   and write notes (any note may be replaced or removed, as today; `updated_by` records the last account that
   saved it, §4.6); **Administrator** = Analyst + Members (create, disable, enable, unlock, change role, set
   password).
4. Safeguards: the API refuses to disable or demote the last enabled Administrator; an Administrator cannot
   change their own role or disable themselves; accounts are disabled, never deleted (notes keep their
   author); a role change ends all of that account's sessions, as disabling and setting a password already do
   (§4.8 "Sessions").
5. Every member change (who, what, when, old → new role) is written to a membership audit table and the
   service log, without passwords or hashes.
6. Migration: in production, before the switch, an Administrator enters each existing account's real email; on
   staging, accounts stay test accounts with test addresses (D54 (8), SRS v4.1 NFR-20); the first
   Administrator is created by the admin command. The production emails are personal data stored on the
   hosting machine: the D54 hosting amendment (in progress, Iheb) must cover them in its data section (who can
   access them, deletion).
7. The threat model (§10.1) and the §4.8 contract are updated in this decision's implementation PR, reviewed
   before merge; the threat model records that the Administrator knows each password they set; new endpoints:
   `GET/POST /api/members`, `PATCH /api/members/{id}` (role, enabled), `POST /api/members/{id}/password`,
   `POST /api/members/{id}/unlock`.

### Reason

The adopted direction (D67) has an Administration screen and roles; doing it on the server with the existing
session and password design keeps D47's security properties and adds no endpoint beyond Members. D54 (8) is
unchanged: staging never holds real accounts.

### Cut line

If D68 is not **merged and running on staging by Fri 18 Dec 2026** (the D68 backend, the migration and the
Sign-in and Members screens, all three), it is not shipped: sign-in stays D47 (username), accounts stay with
`python -m api.admin`, and Administration shows the System tab plus a Members tab stating that accounts are
created by an administrator on the server (D47); D68 is reported as designed, not shipped. Iheb's load to Wed
23 Dec is about 20–27 working days against 20 available (25 Nov), so the cut line is likely to apply. Order
(Iheb): staging → D63 → L1 → F5 swap → production → D68.

### Options rejected

1. email invitations (no free mail service).
2. a one-time password the user must change at the next sign-in (new endpoint, +0.5–1 day against the cut
   line).
3. client-side role checks only (not a security control).

## Consequences

**Easier:**

- Roles are enforced on the server for every route; the UI only hides what the role cannot do.
- D47's security properties are kept, with no endpoint beyond Members and no self-service password-change
  endpoint.

**Harder or committed:**

- The Administrator knows each password they set; the threat model records this.
- The production emails are personal data stored on the hosting machine; the D54 hosting amendment (in
  progress, Iheb) must cover them. On staging, accounts stay test accounts with test addresses.
- The cut line is likely to apply; in that case sign-in stays D47 (username) and D68 is reported as designed,
  not shipped.

## Links

D47, D54 and D67 in [README.md](README.md) · [ADR 0011](0011-analyst-sign-in-sessions.md) · [ADR
0018](0018-ui-direction-enterprise.md) · `docs/api_design.md` §4.8 · `docs/threat_model.md` §10.1

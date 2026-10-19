# 0008. Analyst interface: React web app instead of Streamlit

- **Status:** Proposed
- **Date:** 2026-10-19
- **Register:** D38 in `README.md` (recheck the next free D-ID against `main` before merge)
- **Deciders:** Iheb Zemzemi (author, Part E owner), Donia Mabrouk, Malak Ben Salem, Khalil Dridi
- **AI assistance:** Claude (claude.ai chat) drafted this ADR and the UI mockup; Iheb reviewed both.

## Context

- The online tier has two containers: an API (FastAPI) and an analyst dashboard. The dashboard
  calls the API only and contains no detector logic (`docs/architecture.md` §1–3; NFR-08). [doc]
- ADR 0002 chose Streamlit for the dashboard and notes that "the Execution Plan also allows React;
  the roadmap fixes Streamlit". It also states that changing a component of the stack requires a
  new ADR: this is that ADR. Streamlit was never added to `requirements.txt`, so no code depends
  on it yet. [doc]
- The dashboard must let an analyst upload a recording and read the prediction, confidence,
  risk level and the decision-support notice (FR-01, FR-05–FR-08, NFR-05), browse history
  (FR-10) and add notes stored beside results (FR-11, Could). It must be deployable with Docker
  (NFR-10) and tested in CI (NFR-11). [doc]
- The API response is frozen in Protocol v1.0 §11 (C→E): `prediction`, `confidence`,
  `risk_level`, `model_version`, `profile_version`, `policy_version`, `request_id`, `notice`.
  `docs/risk_policy.md` (d) requires the dashboard to show the prediction next to the
  confidence. [doc]
- Feedback from the course professor (2026-10-19): Streamlit reads as a prototype; the project
  deserves a more professional analyst interface. [stated by Iheb]
- A clickable mockup of the full interface (sign-in, overview dashboard, analyze with all result
  and error states, history, system status) was built and accepted by Iheb on 2026-10-19. It
  needs layouts Streamlit does not offer (a persistent app shell, a combined verdict line,
  per-state error screens, a filterable history table, an overview page with charts).
  [verified: mockup review]
- The SRS requires history (FR-10) and an operational log (FR-20), but no overview dashboard. The
  overview page in the mockup is therefore **new scope**, proposed here so the team approves it
  knowingly. [doc]
- Part E is owned by one person (D26). The frontend code will be written mostly with AI
  assistance (Claude, Claude Code). AGENTS.md requires every AI-produced artifact to be reviewed
  by a team member, and D32 requires each owner to explain their stage without the author.
  [doc]

## Decision

Replace the Streamlit dashboard with a **React + TypeScript web app**. Only the analyst interface
changes: FastAPI, `src/`, the ModelBundle, the risk policy and the §11 contract are unchanged.

1. **Stack:** React with TypeScript in strict mode, built with Vite, in a top-level `web/`
   folder. Styling with plain CSS and the design tokens of the accepted mockup; no UI component
   framework in v1 (fewer dependencies to review and keep secure). Node LTS pinned in
   `web/.nvmrc`; `package-lock.json` committed.
2. **Deployment:** the app is built to static files and served by an `nginx` container named
   `web`. `docker-compose.yml` keeps two containers: `api` and `web` (NFR-10).
3. **Contract:** one TypeScript type, `AnalysisResponse`, mirrors the §11 response exactly; the
   app reads nothing else from the API's success response. A change to §11 needs a decision
   (D26), and this type changes with it.
4. **Same validation rules (NFR-08):** the web app does not hard-code upload limits. The API
   exposes them (e.g. `GET /limits`: accepted formats, maximum size, maximum duration) and the
   app reads them, so the rule exists in one place.
5. **Tests in CI (NFR-11):** a separate CI job for `web/` runs type checking, linting, unit tests
   (Vitest + React Testing Library) and end-to-end tests of the upload → result flow
   (Playwright) against the API with a mock bundle.
6. **Review guardrails for AI-written code:** small PRs (one screen or one concern each); every
   PR description explains what the code does in plain words; a line the owner cannot explain is
   not merged. The owner learns the concepts needed to review (components, state, API calls,
   types, tests) as each part is built.
7. **Pages (v1):** sign-in, Overview, Analyze (upload, result, errors, analyst note), History,
   System status, as in the accepted mockup. Nothing else is added without a decision.
8. **Overview page (new scope), with limits:** it shows only figures computed from the stored
   analysis records and the operational log: analyses per day by verdict, the share of uncertain
   results compared with the deferral rate the frozen policy reached on val-B
   (`deferral_on_fit`, read from the loaded bundle, never hard-coded), processing time, recent
   synthetic and uncertain results without a note, and service and bundle status. The API
   computes these aggregates (e.g. `GET /stats`); the web app does not. It never shows accuracy,
   EER or any other quality metric (there is no ground truth in production; those belong in the
   report, Stages 11–16), no explanation visuals (scope lock, AGENTS.md hard rule 3), no
   per-analyst figures, and nothing from DEEP-VOICE. It is built after the Analyze flow and
   History, and is the first page dropped if the schedule slips.
9. **Fallback:** if the Analyze flow (upload, result, errors) is not merged with green CI by the
   end of week 10, the team may return to Streamlit through a new decision, reusing the same API.

## Consequences

- Easier: a professional interface that matches the accepted mockup; the §11 contract is
  checked by the TypeScript compiler; the end-to-end tests are evidence for NFR-11; the analyst
  interface can be demonstrated on its own.
- Harder: a second language (TypeScript) and toolchain (Node, npm) in the repo, CI and Docker;
  more code for one owner to review; frontend dependencies to keep updated.
- Unchanged: the API, the shared core, the research tier, every other stage and owner. The
  governance test scans `web/` like any other folder, so DEEP-VOICE paths stay forbidden there.
- New scope: the Overview page and its `GET /stats` endpoint add work to Stages 17–18; they come
  last in the build order so they cannot delay a required feature.
- Committed to: the decision-support notice on every screen; no audio and no original file
  names stored by the interface (NFR-04); errors never show a prediction (NFR-06).
- Supersedes the Dashboard row of ADR 0002 (and the matching part of the SDD); the rest of
  ADR 0002 stands. Documents to update in a follow-up PR once this is approved:
  `docs/architecture.md` (component diagram and module table), ADR 0002 (a pointer to this ADR
  on its Dashboard row) and the `AGENTS.md` roadmap row for Stages 17–19.
- Verify later: Stage 18 PRs are small and reviewed by the owner with explanations; the web CI job
  is green before Stage 19; the fallback date is checked in week 10.

## Alternatives considered

- **Keep Streamlit, with a custom theme.** Least work and pure Python, but limited layout
  control; the accepted mockup cannot be built with it, and it does not address the feedback.
- **Server-rendered pages from FastAPI (Jinja2 templates + HTMX).** Python only and simpler
  tooling, but it moves the interface into the API container and blurs the API / dashboard
  separation the architecture relies on (NFR-08).
- **Next.js.** Server-side rendering and routing the app does not need, and more framework to
  review.

## Links

- D26, D32; ADR 0002 (tech stack); ADR 0006 (AI tooling; Playwright planned for Stage 17).
- Protocol v1.0 §11; `docs/risk_policy.md`; `docs/architecture.md`; SRS FR-01, FR-05–FR-11,
  NFR-04–NFR-11.
- Mockup: linked in the PR description.

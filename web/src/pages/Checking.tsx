// Shown while GET /api/session has not answered on load: neither S1 nor the page, so a signed-in analyst sees
// no sign-in flash. No h1: it is not a page (HANDOFF §6 uses "Checking…" for a pending check).
export default function Checking() {
  return (
    <main>
      <p role="status">Checking your session…</p>
    </main>
  )
}

// S10 System status (HANDOFF §1). Stub (Stage 18 PR 3c): no API call, no content yet. The shell (PR 6a) owns main;
// the h1 takes focus on a page change (HANDOFF line 70).
export default function Status() {
  return (
    <>
      <h1 tabIndex={-1}>System status</h1>
      <p>Stub page.</p>
    </>
  )
}

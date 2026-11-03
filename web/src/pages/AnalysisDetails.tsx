// S8 Analysis details (HANDOFF §1). Stub (Stage 18 PR 3c): no API call, no content yet. The route admits only
// a full request ID (routing/routes.ts). The shell (PR 6a) owns main; the h1 takes focus on a page change.
import { useParams } from 'react-router'

export default function AnalysisDetails() {
  const { requestId } = useParams()
  return (
    <>
      <h1 tabIndex={-1}>Analysis details</h1>
      <p>
        Stub page for <code>{requestId}</code>.
      </p>
    </>
  )
}

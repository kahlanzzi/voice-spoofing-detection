// S8 Analysis details (HANDOFF §1). Stub (Stage 18 PR 3c): no API call, no content yet. The route admits only
// a full request ID (routing/routes.ts).
import { useParams } from 'react-router'

export default function AnalysisDetails() {
  const { requestId } = useParams()
  return (
    <main>
      <h1>Analysis details</h1>
      <p>
        Stub page for <code>{requestId}</code>.
      </p>
    </main>
  )
}

// The rail status line (Stage 18 PR 7a; HANDOFF line 41, prototype lines 139-143 and 803): the service status above
// the analyst in the rail footer, linking to System status. It reads the status store and never fetches; the shell
// starts the one check when it mounts.
//
// Ready with a notice (degraded, team decision 2026-11-01; the answer to Q1 in Entry 139): "Ready for analysis" with
// the alert icon, and ", with a notice" in the name; S10 names what fails (PR 7b). "(mock)" is visible when the API
// says mock (Q2). The accessible name is the content, not an aria-label, so it holds the visible text (WCAG 2.5.3):
// the prototype's aria-label hid its "(mock)". Not a live region: it would speak on every page. Icon + text, never
// colour alone (HANDOFF line 185); while a new check runs the last result stays. The text is one inline span, not
// several flex items, and its spaces are text nodes between the spans, so the name keeps them.
import { Link } from 'react-router'
import Icon from '../components/Icon'
import { useStatus } from '../status/useStatus'

export default function RailStatus() {
  const { result } = useStatus()

  if (result === null) {
    return (
      <Link to="/status" className="svc svc-checking">
        <span className="spin" aria-hidden="true" />
        <span>
          <span className="sr">Service status:</span> Checking…
        </span>
      </Link>
    )
  }

  const notice = result.kind === 'ready' && result.notice
  const kind = result.kind === 'unavailable' ? 'down' : notice ? 'notice' : 'ready'
  const mock = 'mock' in result && result.mock
  return (
    <Link to="/status" className={`svc svc-${kind}`}>
      <Icon name={kind === 'ready' ? 'dot' : kind === 'notice' ? 'alert' : 'x'} size={14} />
      <span>
        <span className="sr">Service status:</span> {result.kind === 'ready' ? 'Ready for analysis' : 'Unavailable'}
        {mock && (
          <>
            {' '}
            <span className="svc-mock">(mock)</span>
          </>
        )}
        {notice && <span className="sr">, with a notice</span>}
      </span>
    </Link>
  )
}

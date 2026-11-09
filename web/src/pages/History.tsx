// S7 History (Stage 18 PR C; HANDOFF §1, lines 147, 164, prototype lines 1038-1079), the MVP cut for FR-10: the newest
// page of stored results, marked by verdict. Entry 147 lists what waits: filters, the pager, the row link, scroll restore.
//
// One GET per entry to the page: the effect runs when the page mounts, so a new analysis appears when the analyst next
// opens History, and a query change or the rail link on /history sends nothing. No timer, no polling. The state is the
// page's (Q34): each entry starts on the skeleton, never on old rows (HANDOFF line 68), and an answer that lands after
// leaving is ignored. A null answer is a dropped one (history/history.ts): the guard is already moving to S1.
//
// ?verdict= is ignored and left in the URL (Q36): the GET never forwards it, and the count line claims no filter.
//
// Words: the count and the announcement give what is stored (total; prototype lines 1065, 1081), and "Showing" what is
// on screen (line 1075). The failure is one alert, not also an announcement (Q22). Focus moves only on Retry, which
// unmounts the button that holds it (HANDOFF §7). The shell (PR 6a) owns main; PageChange moves focus to the h1.
import { useEffect, useRef, useState } from 'react'
import Button from '../components/Button'
import Icon from '../components/Icon'
import { isMockResult } from '../components/resultText'
import VerdictPill from '../components/VerdictPill'
import { formatAnalyzedAt, timeZoneLabel } from '../history/analyzedAt'
import { loadHistory, type HistoryAnswer } from '../history/history'
import { useNewAnalysis } from '../shell/useNewAnalysis'
import './History.css'

type HistoryState = { readonly status: 'loading' } | HistoryAnswer

const LOADING: HistoryState = Object.freeze({ status: 'loading' })

// Prototype line 1058.
const SKELETON_WIDTHS = [72, 88, 64, 80, 70, 90, 76, 84]

function resultCount(total: number): string {
  return `${total} ${total === 1 ? 'result' : 'results'}`
}

function Rows({ state }: { state: Extract<HistoryState, { status: 'ready' }> }) {
  const { items, total } = state
  return (
    <>
      <p className="count-line" id="h-count">
        <strong className="num">{resultCount(total)}</strong>
        <span className="muted">all stored results</span>
      </p>
      <div className="panel">
        <div className="table-wrap">
          <table className="data stack-sm">
            <caption className="sr">Stored analysis results, newest first.</caption>
            <thead>
              <tr>
                <th scope="col">{`Analyzed (${timeZoneLabel()})`}</th>
                <th scope="col">Request ID</th>
                <th scope="col">Verdict · risk</th>
                <th scope="col">Analyst note</th>
              </tr>
            </thead>
            <tbody>
              {items.map(({ analysis, created_at: createdAt, note }) => (
                <tr key={analysis.request_id}>
                  <td data-label="Analyzed" className="num">
                    {formatAnalyzedAt(createdAt)}
                  </td>
                  <td data-label="Request ID">
                    <code className="mono history-id">{analysis.request_id}</code>
                  </td>
                  <td data-label="Verdict · risk">
                    <span className="history-verdict">
                      <VerdictPill prediction={analysis.prediction} risk={analysis.risk_level} />
                      {isMockResult(analysis) && <span className="tag tag-demo">Mock result</span>}
                    </span>
                  </td>
                  <td data-label="Analyst note">{note === null ? <span className="muted">None</span> : 'Added'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="history-showing num">{`Showing 1–${items.length} of ${total}`}</p>
      </div>
    </>
  )
}

export default function History() {
  const [state, setState] = useState<HistoryState>(LOADING)
  const [attempt, setAttempt] = useState(0)
  const heading = useRef<HTMLHeadingElement>(null)
  const newAnalysis = useNewAnalysis()

  useEffect(() => {
    let here = true
    void loadHistory().then((answer) => {
      if (here && answer !== null) setState(answer)
    })
    return () => {
      here = false
    }
  }, [attempt])

  function retry() {
    heading.current?.focus()
    setState(LOADING)
    setAttempt((count) => count + 1)
  }

  return (
    <>
      <div className="sr" aria-live="polite" aria-atomic="true" id="history-announcer">
        {state.status === 'ready' && `History loaded. ${resultCount(state.total)}.`}
      </div>
      <div className="page-head">
        <div>
          <h1 tabIndex={-1} ref={heading}>
            History
          </h1>
          <p className="page-sub">Stored analysis results, newest first.</p>
        </div>
      </div>
      <p className="reminder">
        <Icon name="info" size={16} />
        <span>
          <strong>Decision support only:</strong> results do not prove fraud or verify identity.
        </span>
      </p>
      {state.status === 'loading' && (
        <div className="panel" aria-busy="true">
          <p className="sr" role="status">
            Loading history…
          </p>
          <div className="history-skel">
            {SKELETON_WIDTHS.map((width) => (
              <div key={width} className="skel" style={{ width: `${width}%` }} />
            ))}
          </div>
        </div>
      )}
      {state.status === 'failed' && (
        <div className="panel state">
          <div className="state-icon err">
            <Icon name="alert" size={22} />
          </div>
          <div className="state-text" role="alert">
            <h2>History could not be loaded</h2>
            <p className="muted small">The service did not return the stored results. No results were changed.</p>
          </div>
          <Button variant="primary" icon={<Icon name="refresh" size={16} />} onClick={retry}>
            Retry
          </Button>
        </div>
      )}
      {state.status === 'ready' && state.total === 0 && (
        <div className="panel state">
          <div className="state-icon">
            <Icon name="history" size={22} />
          </div>
          <div className="state-text">
            <h2>No analyses yet</h2>
            <p className="muted small">Results appear here after the first analysis. Only results are stored: no audio and no original filenames.</p>
          </div>
          <Button variant="primary" icon={<Icon name="plus" size={16} />} onClick={newAnalysis}>
            New analysis
          </Button>
        </div>
      )}
      {state.status === 'ready' && state.total > 0 && <Rows state={state} />}
      <p className="small muted">Only results, request IDs and analyst notes are stored. No audio, no original filenames, no per-analyst figures.</p>
    </>
  )
}

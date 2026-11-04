// S2 Analyze (HANDOFF §1; Stage 18 PR A). The limits come from the store the shell fills, so entering this page sends
// nothing. The shell (PR 6a) owns main; the h1 takes focus on a page change (HANDOFF line 70).
//
// One hidden file input for the whole page, while limits are ready: never focusable, opened by the visible buttons
// (HANDOFF line 177). Its value resets after a pick, so the same file can be chosen again; a change without a file (a
// cancelled picker) keeps the draft. The File goes to the draft store, never into component state (A23).
import { useRef, type ChangeEvent } from 'react'
import Icon from '../components/Icon'
import { setFileDraft } from '../upload/fileDraft'
import { useLimits } from '../upload/useLimits'
import AnalyzeUpload from './AnalyzeUpload'
import './Analyze.css'

export default function Analyze() {
  const state = useLimits()
  const input = useRef<HTMLInputElement>(null)
  const choose = useRef<HTMLButtonElement>(null)

  function pick(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0]
    event.target.value = ''
    if (chosen !== undefined) setFileDraft(chosen)
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 tabIndex={-1}>Analyze a recording</h1>
          <p className="page-sub">Upload the recorded segment of a high-risk request before acting on it.</p>
        </div>
      </div>
      <div className="notice">
        <Icon name="info" />
        <span>
          <strong>Decision support only.</strong> A result estimates whether a recording is synthetic. It does not prove
          fraud, verify identity, or replace your judgment.
        </span>
      </div>
      {state.status === 'ready' && (
        <input
          ref={input}
          type="file"
          id="file-input"
          className="sr"
          accept={`${state.limits.formats.map((format) => `.${format}`).join(',')},audio/*`}
          tabIndex={-1}
          aria-hidden="true"
          onChange={pick}
        />
      )}
      <AnalyzeUpload state={state} chooseRef={choose} onChoose={() => input.current?.click()} />
    </>
  )
}

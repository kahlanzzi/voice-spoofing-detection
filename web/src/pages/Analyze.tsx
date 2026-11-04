// S2 and S3 Analyze (HANDOFF §1; Stage 18 PR A). The limits come from the store the shell fills, so entering this page sends
// nothing. The shell (PR 6a) owns main; the h1 takes focus on a page change (HANDOFF line 70).
//
// One hidden file input for the whole page, while limits are ready: never focusable, opened by the visible buttons
// (HANDOFF line 177). Its value resets after a pick, so the same file can be chosen again; a change without a file (a
// cancelled picker) keeps the draft. The File goes to the draft store, never into component state (A23).
//
// A kept draft while limits are not ready shows only the limits block (prototype line 919): it can happen after an
// expiry, since the File outlives the session and the limits do not. A pick moves focus to the file card, and its
// result goes to this page's own polite region as a new node (S10's technique, no timer) once the checks have settled.
// Only a pick arms that, so coming back to a kept draft announces nothing. Failures are spoken by the alert (Q22).
//
// Analyze recording moves focus to the file heading: the prototype's target, the button, is disabled at that moment and
// cannot hold focus (line 1318). An answer that arrives on this page moves focus to its title (HANDOFF line 178); one
// that arrived elsewhere does neither, since the page change then focuses the h1. Only a change from submitting seen
// here counts, so a remount speaks nothing. The result goes to an assertive region, not an alert (Q22).
import { useEffect, useRef, type ChangeEvent } from 'react'
import Icon from '../components/Icon'
import { resultAnnouncement } from '../components/resultText'
import { resetAnalysis, startAnalysis } from '../upload/analysis'
import { clearFileDraft, setFileDraft } from '../upload/fileDraft'
import { useAnalysis } from '../upload/useAnalysis'
import { useFileDraft } from '../upload/useFileDraft'
import { useLimits } from '../upload/useLimits'
import AnalyzeFile from './AnalyzeFile'
import AnalyzeLimits from './AnalyzeLimits'
import AnalyzeResult from './AnalyzeResult'
import AnalyzeUpload from './AnalyzeUpload'
import './Analyze.css'

function speak(region: HTMLElement | null, text: string) {
  const message = document.createElement('p')
  message.textContent = text
  region?.replaceChildren(message)
}

export default function Analyze() {
  const state = useLimits()
  const file = useFileDraft()
  const analysis = useAnalysis()
  const input = useRef<HTMLInputElement>(null)
  const choose = useRef<HTMLButtonElement>(null)
  const fileHeading = useRef<HTMLHeadingElement>(null)
  const limitsPanel = useRef<HTMLDivElement>(null)
  const resultHeading = useRef<HTMLHeadingElement>(null)
  const live = useRef<HTMLDivElement>(null)
  const assertive = useRef<HTMLDivElement>(null)
  const lastStatus = useRef(analysis.status)
  const pendingFocus = useRef<'file' | 'choose' | null>(null)
  const armed = useRef(false)

  // Focus once the new state is on screen (the SignIn.tsx pending-focus pattern).
  useEffect(() => {
    const target = pendingFocus.current
    if (target === null) return
    pendingFocus.current = null
    if (target === 'file') fileHeading.current?.focus()
    else choose.current?.focus()
  })

  useEffect(() => {
    const was = lastStatus.current
    lastStatus.current = analysis.status
    if (was !== 'submitting' || analysis.status !== 'result') return
    resultHeading.current?.focus()
    speak(assertive.current, resultAnnouncement(analysis.analysis))
  }, [analysis])

  function announce(text: string) {
    speak(live.current, text)
  }

  function pick(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0]
    event.target.value = ''
    if (chosen === undefined) return
    pendingFocus.current = 'file'
    armed.current = true
    resetAnalysis()
    setFileDraft(chosen)
  }

  function analyze(chosen: File) {
    pendingFocus.current = 'file'
    announce('Analysis started.')
    void startAnalysis(chosen)
  }

  function remove() {
    pendingFocus.current = 'choose'
    clearFileDraft()
    announce('File removed.')
  }

  function settled(name: string, allPassed: boolean) {
    if (!armed.current) return
    armed.current = false
    announce(`File selected: ${name}.${allPassed ? ' All checks passed.' : ''}`)
  }

  const openPicker = () => input.current?.click()

  const regions = (
    <>
      <div className="sr" aria-live="polite" aria-atomic="true" id="analyze-announcer" ref={live} />
      <div className="sr" aria-live="assertive" aria-atomic="true" id="analyze-assertive" ref={assertive} />
    </>
  )

  if (analysis.status === 'result' && file !== null) {
    return (
      <>
        <AnalyzeResult analysis={analysis.analysis} fileName={file.name} headingRef={resultHeading} />
        {regions}
      </>
    )
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
      {file === null ? (
        <AnalyzeUpload state={state} chooseRef={choose} onChoose={openPicker} />
      ) : state.status === 'ready' ? (
        <AnalyzeFile
          file={file}
          limits={state.limits}
          headingRef={fileHeading}
          submitting={analysis.status === 'submitting'}
          onAnalyze={() => analyze(file)}
          onReplace={openPicker}
          onRemove={remove}
          onSettled={(allPassed) => settled(file.name, allPassed)}
        />
      ) : (
        <div className="panel panel-pad" tabIndex={-1} ref={limitsPanel}>
          <AnalyzeLimits state={state} retryFocus={limitsPanel} />
        </div>
      )}
      {regions}
    </>
  )
}

// S5 (Stage 18 PR B; HANDOFF §1 and lines 146, 278; prototype lines 943-966): what the analyst sees when no verdict was
// made. Each kind has its own words and actions, and no raw server text ever reaches them (§6).
//
// Two changes to the prototype. The timeout body drops its late-result sentences: a 504 stores nothing (§4.3). And four
// kinds are added, for a check the browser passed and the service refused: an unreadable duration can really end in
// recording_too_long (§6; copy proposals, Entry 146 Q28).
//
// The failure is the one alert (Q22), with the prototype's words (line 1303). Sign in again ends the session here
// (prototype line 1412), and the analysis store then returns to idle, so the analyst comes back to S3 with the file.
import type { JSX, Ref } from 'react'
import { Link } from 'react-router'
import type { RequestId } from '../api/types'
import Button from '../components/Button'
import ErrorPanel, { NO_VERDICT, type ErrorPanelProps } from '../components/ErrorPanel'
import Icon from '../components/Icon'
import { expire } from '../session/session'
import type { AnalysisErrorKind } from '../upload/analysis'

type Action = 'retry' | 'replace' | 'remove' | 'status' | 'signin'

type ErrorText = Pick<ErrorPanelProps, 'tone' | 'icon' | 'badge' | 'title' | 'body'> & { actions: readonly Action[] }

const REFUSED: readonly Action[] = ['replace', 'remove']

const ERRORS: Record<AnalysisErrorKind, ErrorText> = {
  network: {
    tone: 'net',
    icon: 'wifi',
    badge: 'Network interruption',
    title: 'The connection to the service was interrupted',
    body: 'Your browser could not reach the service, so the recording was not analyzed. Check your connection, then retry. Your file is still selected.',
    actions: ['retry', 'replace'],
  },
  service: {
    tone: 'svc',
    icon: 'alert',
    badge: 'Analysis service unavailable',
    title: 'The analysis service is unavailable',
    body: 'The service received the request but cannot analyze recordings right now. Your file is still selected. Try again in a few minutes. If it keeps failing, report the request ID through your usual support channel.',
    actions: ['retry', 'status'],
  },
  timeout: {
    tone: 'svc',
    icon: 'clock',
    badge: 'Request timeout',
    title: 'The analysis did not finish in time',
    body: 'The service did not answer before the console stopped waiting. Your file is still selected. Retrying sends a new request with a new request ID.',
    actions: ['retry', 'replace'],
  },
  corrupt: {
    tone: 'input',
    icon: 'file',
    badge: 'Corrupt audio',
    title: 'The recording could not be decoded',
    body: 'The file passed the checks in your browser, but the service could not decode the audio. It may be damaged, or not really in the format its name suggests. Export the segment again and choose the new file.',
    actions: REFUSED,
  },
  session: {
    tone: 'net',
    icon: 'lock',
    badge: 'Session expired',
    title: 'Your session has expired',
    body: 'Sign in again to continue. Your selected file stays on this screen and nothing was analyzed.',
    actions: ['signin'],
  },
  server: {
    tone: 'svc',
    icon: 'alert',
    badge: 'Unexpected service error',
    title: 'Something went wrong on the service',
    body: 'The request reached the service but failed unexpectedly. Your file is still selected. Retry once. If it fails again, report the request ID through your usual support channel.',
    actions: ['retry', 'replace'],
  },
  format: {
    tone: 'input',
    icon: 'file',
    badge: 'Unsupported format',
    title: 'The service does not accept this format',
    body: 'The file passed the checks in your browser, but the service refused its format. Choose a recording in an accepted format.',
    actions: REFUSED,
  },
  size: {
    tone: 'input',
    icon: 'file',
    badge: 'File too large',
    title: 'The file is over the service’s size limit',
    body: 'The file passed the checks in your browser, but the service refused its size. Export only the segment with the request.',
    actions: REFUSED,
  },
  duration: {
    tone: 'input',
    icon: 'file',
    badge: 'Recording too long',
    title: 'The recording is over the duration limit',
    body: 'The service measured the recording and it is over the limit. Upload only the segment with the request.',
    actions: REFUSED,
  },
  empty: {
    tone: 'input',
    icon: 'file',
    badge: 'Empty file',
    title: 'The recording has no audio',
    body: 'The service found no audio in this file. Export the segment again and choose the new file.',
    actions: REFUSED,
  },
}

interface AnalyzeErrorProps {
  kind: AnalysisErrorKind
  requestId: RequestId | null
  fileName: string
  headingRef: Ref<HTMLHeadingElement>
  onRetry: () => void
  onReplace: () => void
  onRemove: () => void
}

export default function AnalyzeError({ kind, requestId, fileName, headingRef, onRetry, onReplace, onRemove }: AnalyzeErrorProps) {
  const { actions, ...text } = ERRORS[kind]
  const buttons: Record<Action, JSX.Element> = {
    retry: (
      <Button key="retry" variant="primary" icon={<Icon name="refresh" size={16} />} onClick={onRetry}>
        Retry analysis
      </Button>
    ),
    replace: (
      <Button key="replace" variant={actions[0] === 'replace' ? 'primary' : 'quiet'} onClick={onReplace}>
        Choose another file
      </Button>
    ),
    remove: (
      <Button key="remove" variant="quiet" onClick={onRemove}>
        Remove file
      </Button>
    ),
    status: (
      <Link key="status" className="btn btn-quiet" to="/status">
        Check system status
      </Link>
    ),
    signin: (
      <Button key="signin" variant="primary" onClick={expire}>
        Sign in again
      </Button>
    ),
  }
  return (
    <>
      <p className="sr" role="alert">{`${text.title}. ${NO_VERDICT}`}</p>
      <ErrorPanel {...text} fileName={fileName} requestId={requestId} headingRef={headingRef}>
        {actions.map((action) => buttons[action])}
      </ErrorPanel>
    </>
  )
}

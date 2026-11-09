// ErrorPanel (Stage 18 PR B; HANDOFF line 146, prototype lines 960-964): kind badge, title, "No verdict was produced",
// cause, what is kept, request ID or "None", actions. Never a verdict, risk or score (NFR-06). The actions come from the
// caller. Icon + text, never colour alone.
import type { ReactNode, Ref } from 'react'
import type { RequestId } from '../api/types'
import Icon, { type IconName } from './Icon'
import './ErrorPanel.css'

export const NO_VERDICT = 'The analysis did not complete. No verdict was produced.'

export interface ErrorPanelProps {
  tone: 'input' | 'net' | 'svc'
  icon: IconName
  badge: string
  title: string
  body: string
  fileName: string
  requestId: RequestId | null
  headingRef?: Ref<HTMLHeadingElement>
  children: ReactNode
}

export default function ErrorPanel({ tone, icon, badge, title, body, fileName, requestId, headingRef, children }: ErrorPanelProps) {
  return (
    <section className={`alert k-${tone}`} aria-labelledby="err-h">
      <span className={`alert-kind kind-${tone}`}>
        <Icon name={icon} size={14} />
        {badge}
      </span>
      <h2 id="err-h" tabIndex={-1} ref={headingRef}>
        {title}
      </h2>
      <p className="no-pred">{NO_VERDICT}</p>
      <p className="explain">{body}</p>
      <dl className="kv">
        <dt>Recording</dt>
        <dd>
          <span className="mono">{fileName}</span> <span className="muted">(this screen only)</span>
        </dd>
        <dt>Request ID</dt>
        {requestId === null ? (
          <dd className="muted">None: the request was not accepted by the service</dd>
        ) : (
          <dd>
            <code className="mono">{requestId}</code>
          </dd>
        )}
      </dl>
      <div className="actions">{children}</div>
    </section>
  )
}

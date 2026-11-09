// ResultCard (Stage 18 PR B; HANDOFF lines 144, 224-225, 283; prototype lines 981-999): verdict, risk, required action,
// explanation, request, technical details, in that fixed order, with the service's notice at the foot (FR-08, NFR-05).
// Every value comes from the response as it is: the browser computes nothing and has no default (HANDOFF line 270), and
// risk comes from risk_level, not from the verdict. The model output stays inside Technical details (HANDOFF line 199).
//
// The response has no mock field: is_mock is stored, never returned (api/routers/analyses.py). model_version "mock" is
// what the server's MockAnalyzer reports, and the only sign of it in the 8 fields (Entry 146 Q26). The other two
// versions are real names, so they mark nothing.
//
// Left out: the prototype's Analyzed and Processing time rows, which the response does not carry, and its Copy button,
// which needs a timer (Entry 146 Q29). PR 13 adds the prop `fresh` and the stored-result storage line.
import type { Ref } from 'react'
import type { AnalysisResponse, RiskLevel, Verdict } from '../api/types'
import Icon, { type IconName } from './Icon'
import { EXPLANATION, isMockResult, REQUIRED_ACTION, RISK_TEXT, VERDICT_TEXT } from './resultText'
import './ResultCard.css'

const VERDICT_LOOK: Record<Verdict, { className: string; icon: IconName }> = {
  synthetic: { className: 'r-syn', icon: 'alert' },
  uncertain: { className: 'r-unc', icon: 'help' },
  authentic: { className: 'r-auth', icon: 'shieldcheck' },
}

const RISK_ICON: Record<RiskLevel, IconName> = { LOW: 'down', MEDIUM: 'minus', HIGH: 'up' }

interface ResultCardProps {
  analysis: AnalysisResponse
  fileName: string
  headingRef?: Ref<HTMLHeadingElement>
}

export default function ResultCard({ analysis, fileName, headingRef }: ResultCardProps) {
  const { prediction, risk_level: risk } = analysis
  const look = VERDICT_LOOK[prediction]
  const action = REQUIRED_ACTION[prediction]
  const output = analysis.confidence.toFixed(2)
  return (
    <article className={`result ${look.className}`} aria-labelledby="res-h">
      <div className="result-top">
        <p className="kicker">Model verdict</p>
        <h2 id="res-h" tabIndex={-1} ref={headingRef}>
          <Icon name={look.icon} size={26} />
          {VERDICT_TEXT[prediction]}
        </h2>
        <div className="meta-row">
          <span className="risk">
            <Icon name={RISK_ICON[risk]} size={14} />
            {RISK_TEXT[risk].toUpperCase()}
          </span>
          {isMockResult(analysis) && <span className="tag tag-demo">Mock result</span>}
        </div>
      </div>
      <div className="result-body">
        <section className="todo" aria-labelledby="todo-h">
          <h3 id="todo-h">
            <Icon name={prediction === 'uncertain' ? 'alert' : 'check'} size={18} />
            {action.heading}
          </h3>
          <ul>
            {action.items.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
        <section aria-labelledby="why-h">
          <h3 className="section-h" id="why-h">
            What this result means
          </h3>
          <p className="explain">{EXPLANATION[prediction]}</p>
        </section>
        <section aria-labelledby="meta-h">
          <h3 className="section-h" id="meta-h">
            Request
          </h3>
          <dl className="kv">
            <dt>Request ID</dt>
            <dd>
              <code className="mono">{analysis.request_id}</code>
            </dd>
            <dt>Storage</dt>
            <dd>
              <span className="mono">{fileName}</span>{' '}
              <span className="muted">is visible on this screen only and is not stored. The audio is deleted when processing finishes.</span>
            </dd>
          </dl>
        </section>
      </div>
      <details className="more">
        <summary>Technical details</summary>
        <div className="more-body">
          <div className="score-def">
            <p>
              <strong>Calibrated model output:</strong> <span className="score-val">{output}</span>{' '}
              <span className="muted">(range 0.50 to 1.00)</span>
            </p>
            <p>
              It is the model’s calibrated output for whichever outcome it finds more likely. The frozen risk policy, not this
              number alone, determines the verdict and the risk band
              {prediction === 'uncertain' && ', which is why a “Cannot decide” result can show a high value'}.
            </p>
            <p>
              It is <strong>not</strong>:
            </p>
            <ul>
              <li>the probability that fraud occurred;</li>
              <li>the probability that the verdict is correct;</li>
              <li>{`a percentage of certainty (${output} does not mean “${Math.round(analysis.confidence * 100)}% certain”).`}</li>
            </ul>
          </div>
          <dl className="kv">
            <dt>Model version</dt>
            <dd className="mono">{analysis.model_version}</dd>
            <dt>Preprocessing profile</dt>
            <dd className="mono">{analysis.profile_version}</dd>
            <dt>Risk policy</dt>
            <dd className="mono">{analysis.policy_version}</dd>
            <dt>API fields</dt>
            <dd className="mono">{`prediction = ${prediction}, confidence = ${output}, risk_level = ${risk}`}</dd>
          </dl>
        </div>
      </details>
      <p className="api-notice">
        <strong>Notice from the service:</strong> {analysis.notice}
      </p>
    </article>
  )
}

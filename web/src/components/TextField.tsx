// TextField (Stage 18 PR 8a; HANDOFF line 138, prototype lines 267-277): the label above a native <input>, and the
// error below it as an icon + text, linked by aria-describedby, with aria-invalid. `invalid` marks the field without
// text of its own (S1's invalid credentials point at the banner through `describedBy`, which an error replaces).
// A hint, when given, is described first. `action` sits inside the input's box (S1's Show/Hide). No font size here:
// base.css gives inputs 16 px on phones (B4).
import type { ChangeEvent, ReactNode, Ref } from 'react'
import Icon from './Icon'
import './TextField.css'

export interface TextFieldProps {
  id: string
  label: string
  value: string
  onChange: (event: ChangeEvent<HTMLInputElement>) => void
  error?: string | null
  invalid?: boolean
  describedBy?: string
  hint?: string
  action?: ReactNode
  name?: string
  type?: string
  autoComplete?: string
  disabled?: boolean
  ref?: Ref<HTMLInputElement>
}

export default function TextField({ id, label, error, invalid = false, describedBy, hint, action, ref, ...input }: TextFieldProps) {
  const errorId = `${id}-err`
  const hintId = `${id}-hint`
  const described = [hint ? hintId : undefined, error ? errorId : describedBy].filter(Boolean).join(' ')
  const field = (
    <input
      {...input}
      className="input"
      id={id}
      ref={ref}
      aria-invalid={Boolean(error) || invalid}
      aria-describedby={described === '' ? undefined : described}
    />
  )
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {hint && (
        <span className="field-hint" id={hintId}>
          {hint}
        </span>
      )}
      {action === undefined ? (
        field
      ) : (
        <div className="pw">
          {field}
          {action}
        </div>
      )}
      {error && (
        <span className="field-err" id={errorId}>
          <Icon name="alert" size={14} />
          {error}
        </span>
      )}
    </div>
  )
}

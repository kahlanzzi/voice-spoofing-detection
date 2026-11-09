// Button (Stage 18 PR 8a; HANDOFF line 137, prototype lines 103-123): a native <button> with the prototype's class
// names, so context rules such as `.rail .btn-primary` still apply. The caller picks the variant, so one primary
// per area is a choice made where the button is used.
//
// Busy: the spinner replaces the icon (or comes before the label), and the button is disabled with aria-busy, which
// blocks a repeat submit. The label comes from the caller: S1 says "Signing in…" (Entry 134 Q4b), S10 keeps "Check
// now". type defaults to "button", so a button inside a form never submits by accident.
import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'
import Spinner from './Spinner'
import './Button.css'

export type ButtonVariant = 'primary' | 'secondary' | 'quiet' | 'destructive' | 'link'

const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary: 'btn btn-primary',
  secondary: 'btn btn-secondary',
  quiet: 'btn btn-quiet',
  destructive: 'btn btn-danger',
  link: 'btn-link',
}

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'className' | 'type'> {
  variant: ButtonVariant
  size?: 'md' | 'sm'
  busy?: boolean
  icon?: ReactNode
  type?: 'button' | 'submit'
  ref?: Ref<HTMLButtonElement>
}

export default function Button({ variant, size = 'md', busy = false, icon, type = 'button', disabled, children, ...rest }: ButtonProps) {
  const className = size === 'sm' ? `${VARIANT_CLASS[variant]} btn-sm` : VARIANT_CLASS[variant]
  return (
    <button {...rest} type={type} className={className} disabled={disabled || busy} aria-busy={busy || undefined}>
      {busy ? <Spinner /> : icon}
      {children}
    </button>
  )
}

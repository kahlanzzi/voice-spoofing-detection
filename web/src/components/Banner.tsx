// Banner (Stage 18 PR 8a; HANDOFF line 143, prototype lines 261-264): icon + text, never animated. The role comes
// from the caller, with no default: an error that answers an action is an alert, a notice that appears after an
// action is a status, and a notice that is part of the page has none (S10's degraded notice). To announce the same
// message again, the caller changes the key (S1's alertKey).
import type { ReactNode } from 'react'
import Icon, { type IconName } from './Icon'
import './Banner.css'

export interface BannerProps {
  tone: 'info' | 'warn' | 'err'
  icon: IconName
  role?: 'alert' | 'status'
  id?: string
  children: ReactNode
}

export default function Banner({ tone, icon, role, id, children }: BannerProps) {
  return (
    <div className={`banner-${tone}`} role={role} id={id}>
      <Icon name={icon} size={16} />
      <span>{children}</span>
    </div>
  )
}

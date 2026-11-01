// Theme choice (team decision 2026-11-01): System, Light or Dark, kept in localStorage and applied
// as <html data-theme>. "system" removes the attribute so prefers-color-scheme decides (prototype
// applyTheme). Storage can be missing or throw (private mode, blocked site data): every access is
// guarded and falls back to "system". No UI control yet (Stage 18 PR 6).

export type ThemeChoice = 'system' | 'light' | 'dark'

export const THEME_KEY = 'vsd-theme'

const CHOICES: readonly ThemeChoice[] = ['system', 'light', 'dark']

function isThemeChoice(value: unknown): value is ThemeChoice {
  return CHOICES.includes(value as ThemeChoice)
}

export function readTheme(): ThemeChoice {
  try {
    const stored = window.localStorage.getItem(THEME_KEY)
    return isThemeChoice(stored) ? stored : 'system'
  } catch {
    return 'system'
  }
}

export function writeTheme(choice: ThemeChoice): void {
  try {
    window.localStorage.setItem(THEME_KEY, choice)
  } catch {
    // Not stored; the choice still applies for this page view.
  }
}

export function applyTheme(choice: ThemeChoice, root: HTMLElement = document.documentElement): void {
  if (choice === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', choice)
}

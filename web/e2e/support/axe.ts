// axe-core on a page, in light and dark (HANDOFF l.186, l.297: WCAG 2.0/2.1/2.2 A and AA, 0 violations). Only
// { id, impact, target } of a violation is ever reported: a node's HTML could hold typed text, so it never reaches
// an assertion, the JSON report or the console. No rule is disabled and no part of the page is left out: an exclusion
// is one exact rule + target with its reason, passed in the spec and listed in the project log, and one that matches
// nothing fails the scan. No bypassCSP: if the page's CSP blocks axe, the scan fails (Entry 135 decision).
import AxeBuilder from '@axe-core/playwright'
import { expect, type Page } from '@playwright/test'

// axe-core 4.14.0 tags no rule wcag22a (its 2.2 rule is AA); kept so the list matches HANDOFF l.297.
export const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'] as const

export const SCHEMES = ['light', 'dark'] as const

type AxeResults = Awaited<ReturnType<AxeBuilder['analyze']>>

export interface AxeFinding {
  id: string
  impact: string | null
  target: string
}

export interface AxeExclusion {
  rule: string
  target: string
  reason: string
  /** Only in this scheme; both when absent. */
  scheme?: (typeof SCHEMES)[number]
}

/** One finding per violating node: the rule, its impact and the node's selector, nothing else. */
export function axeFindings(results: Pick<AxeResults, 'violations'>): AxeFinding[] {
  return results.violations.flatMap((violation) => violation.nodes.map((node) => ({
    id: violation.id,
    impact: node.impact ?? violation.impact ?? null,
    target: node.target.map((part) => (Array.isArray(part) ? part.join(' >>> ') : String(part))).join(' | '),
  })))
}

/** The findings no exclusion covers, and the exclusions that covered nothing (stale, so they fail too). */
export function applyExclusions(findings: AxeFinding[], exclusions: readonly AxeExclusion[]) {
  const covers = (e: AxeExclusion, f: AxeFinding) => e.rule === f.id && e.target === f.target
  return {
    left: findings.filter((f) => !exclusions.some((e) => covers(e, f))),
    unused: exclusions.filter((e) => !findings.some((f) => covers(e, f))).map(({ rule, target }) => ({ rule, target })),
  }
}

/**
 * Scans the page as it is, in light then dark (prefers-color-scheme: the app's default theme is "system"), with
 * reduced motion so no colour is read mid-transition (base.css turns transitions off). Restores the emulation after.
 */
export async function expectNoAxeViolations(page: Page, state: string, exclusions: readonly AxeExclusion[] = []) {
  for (const colorScheme of SCHEMES) {
    await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' })
    const results = await new AxeBuilder({ page }).withTags([...WCAG_TAGS]).analyze()
    const { left, unused } = applyExclusions(axeFindings(results), exclusions.filter((e) => (e.scheme ?? colorScheme) === colorScheme))
    expect(left, `axe violations: ${state} (${colorScheme})`).toEqual([])
    expect(unused, `axe exclusions that matched nothing: ${state} (${colorScheme})`).toEqual([])
  }
  await page.emulateMedia({ colorScheme: null, reducedMotion: null })
}

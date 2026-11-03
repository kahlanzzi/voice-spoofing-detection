import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    files: ['**/*.{ts,tsx,js}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { globals: globals.browser },
    plugins: { 'react-hooks': reactHooks, 'react-refresh': reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    },
  },
  // No polling (team decision 2026-11-01): no timers in the API layer, session state, routing, the sign-in
  // screen (no lockout countdown, PR 4), the shell (PR 6a), the status store (PR 7a) or S10 (PR 7b: no polling, no
  // relative time), whether called bare or through window or globalThis. Tests included.
  {
    files: ['src/api/**', 'src/session/**', 'src/status/**', 'src/routing/**', 'src/shell/**', 'src/pages/SignIn*.{ts,tsx}', 'src/pages/signInOutcome*.{ts,tsx}', 'src/pages/Status*.{ts,tsx}', 'src/pages/statusView*.{ts,tsx}'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Identifier[name=/^(setTimeout|setInterval)$/]',
          message: 'No timers in src/api, src/session, src/status, src/routing, src/shell or the sign-in screen (no polling).',
        },
      ],
    },
  },
  // Session state lives in memory only and the username is never logged (Stage 18 PR 3b); likewise the
  // sign-in `next` and any route state live only in the URL (PR 3c): no Web Storage, no document.cookie (the
  // session cookie is the server's HttpOnly cookie), no console. Tests are exempt, they spy on these. Do not
  // add no-restricted-syntax here: it would replace the timer rule above. The sign-in screen joins (PR 4): the
  // password and Analyst ID live in its state only. The shell joins (PR 6a): the theme is stored by theme.ts only.
  // The status store joins (PR 7a): in memory only. S10 joins (PR 7b): it shows the store, never stores or logs it.
  {
    files: ['src/session/**', 'src/status/**', 'src/routing/**', 'src/shell/**', 'src/pages/SignIn*.{ts,tsx}', 'src/pages/signInOutcome*.{ts,tsx}', 'src/pages/Status*.{ts,tsx}', 'src/pages/statusView*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}'],
    rules: {
      'no-restricted-globals': [
        'error',
        ...['localStorage', 'sessionStorage', 'console'].map((name) => ({
          name,
          message: 'No storage or console in src/session, src/routing or the sign-in screen (memory or URL only, never logged).',
        })),
      ],
      'no-restricted-properties': [
        'error',
        ...['window', 'globalThis', 'self'].flatMap((object) =>
          ['localStorage', 'sessionStorage', 'console'].map((property) => ({
            object,
            property,
            message: 'No storage or console in src/session, src/routing or the sign-in screen (memory or URL only, never logged).',
          })),
        ),
        { object: 'document', property: 'cookie', message: 'No document.cookie in src/session, src/routing or the sign-in screen (HttpOnly cookie).' },
      ],
    },
  },
  // End-to-end tests (Stage 18 PR 5a) run in Node. No fixed waits: Playwright's auto-wait and expect polling only.
  // The password and cookie values never go to fill/type/pressSequentially/insertText (Playwright prints typed text
  // in step titles, kept in the JSON report) nor to expect (a failure prints what it was given): typeSecret() types
  // the password, assertions take booleans. rules-of-hooks is off: fixtures call Playwright's use(), which the
  // plugin mistakes for React's `use`; this is not React code.
  {
    files: ['e2e/**', 'playwright.config.ts'],
    languageOptions: { globals: globals.node },
    rules: {
      'react-hooks/rules-of-hooks': 'off',
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Identifier[name=/^(setTimeout|setInterval)$/]',
          message: 'No fixed waits in e2e: use auto-wait or expect polling.',
        },
        {
          selector: 'MemberExpression[property.name="waitForTimeout"]',
          message: 'No fixed waits in e2e: use auto-wait or expect polling.',
        },
        ...['fill', 'type', 'pressSequentially', 'insertText'].flatMap((method) => [
          `CallExpression[callee.property.name="${method}"] > CallExpression.arguments[callee.name="accountPassword"]`,
          `CallExpression[callee.property.name="${method}"] > Identifier.arguments[name=/^(password|cookie)/i]`,
        ]).map((selector) => ({ selector, message: 'Never type a secret with Playwright input APIs: use typeSecret().' })),
        ...[
          'CallExpression[callee.name="expect"] > CallExpression.arguments[callee.name="accountPassword"]',
          'CallExpression[callee.name="expect"] > Identifier.arguments[name=/^(password|cookie)/i]',
        ].map((selector) => ({ selector, message: 'Never pass a secret to expect: assert a boolean.' })),
        {
          selector: 'MemberExpression[property.name=/^(disableRules|exclude|include|withRules|options)$/]',
          message: 'No blanket axe exclusion: pass an exact { rule, target, reason } to expectNoAxeViolations().',
        },
        {
          selector: 'CallExpression[callee.property.name="withTags"]:not(:has(Identifier[name="WCAG_TAGS"]))',
          message: 'axe runs with WCAG_TAGS only (HANDOFF l.297).',
        },
        {
          selector: 'Identifier[name="bypassCSP"]',
          message: 'No bypassCSP: a CSP block of axe is reported, not bypassed (Entry 135).',
        },
      ],
    },
  },
)

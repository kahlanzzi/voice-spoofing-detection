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
  // No polling (team decision 2026-11-01): no timers in the API layer, session state, routing or the sign-in
  // screen (no lockout countdown, PR 4), whether called bare or through window or globalThis. Tests included.
  {
    files: ['src/api/**', 'src/session/**', 'src/routing/**', 'src/pages/SignIn*.{ts,tsx}', 'src/pages/signInOutcome*.{ts,tsx}'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Identifier[name=/^(setTimeout|setInterval)$/]',
          message: 'No timers in src/api, src/session, src/routing or the sign-in screen (no polling).',
        },
      ],
    },
  },
  // Session state lives in memory only and the username is never logged (Stage 18 PR 3b); likewise the
  // sign-in `next` and any route state live only in the URL (PR 3c): no Web Storage, no document.cookie (the
  // session cookie is the server's HttpOnly cookie), no console. Tests are exempt, they spy on these. Do not
  // add no-restricted-syntax here: it would replace the timer rule above. The sign-in screen joins (PR 4): the
  // password and Analyst ID live in its state only.
  {
    files: ['src/session/**', 'src/routing/**', 'src/pages/SignIn*.{ts,tsx}', 'src/pages/signInOutcome*.{ts,tsx}'],
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
)

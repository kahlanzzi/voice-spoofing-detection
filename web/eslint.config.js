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
  // No polling (team decision 2026-11-01): no timers in the API layer, whether called bare or through
  // window or globalThis.
  {
    files: ['src/api/**'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Identifier[name=/^(setTimeout|setInterval)$/]',
          message: 'No timers in src/api (no polling).',
        },
      ],
    },
  },
)

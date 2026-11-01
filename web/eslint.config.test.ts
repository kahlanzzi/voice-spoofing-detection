// @vitest-environment node
import { ESLint } from 'eslint'
import { beforeAll, describe, expect, it } from 'vitest'

// The no-timer rule in eslint.config.js: an error in src/api, not elsewhere.

const eslint = new ESLint()

// The first lint loads the config and its plugins, which can take longer than the 5 s default.
beforeAll(() => eslint.lintText('', { filePath: 'src/api/warm-up.ts' }), 60_000)

async function ruleIds(code: string, filePath: string): Promise<(string | null)[]> {
  const [result] = await eslint.lintText(code, { filePath })
  return result!.messages.map((message) => message.ruleId)
}

describe('no timers in src/api', () => {
  it.each([
    'setTimeout(() => undefined, 1)\n',
    'setInterval(() => undefined, 1)\n',
    'window.setTimeout(() => undefined, 1)\n',
    'globalThis.setInterval(() => undefined, 1)\n',
  ])('%s is an error in src/api', async (code) => {
    expect(await ruleIds(code, 'src/api/planted.ts')).toContain('no-restricted-syntax')
  })

  it('the same call outside src/api is allowed', async () => {
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/theme.ts')).not.toContain('no-restricted-syntax')
  })
})

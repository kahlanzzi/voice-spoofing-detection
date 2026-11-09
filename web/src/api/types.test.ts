import { describe, expect, expectTypeOf, it } from 'vitest'
import apiDesign from '../../../docs/api_design.md?raw'
import schemas from '../../../api/schemas.py?raw'
import { ANALYSIS_RESPONSE_FIELDS, ERROR_CODES, type AnalysisResponse, type ErrorCode } from './types'

// The web types against the contract: the AnalysisResponse fields in api/schemas.py and the codes in the
// docs/api_design.md §6 table. Both files are only read.

function pydanticFields(model: string): string[] {
  const body = new RegExp(`^class ${model}\\(_Strict\\):\\n((?: {4}.*\\n)+)`, 'm').exec(schemas)?.[1]
  if (body === undefined) throw new Error(`api/schemas.py: no class ${model}`)
  return [...body.matchAll(/^ {4}(\w+):/gm)].map((match) => match[1]!)
}

function section6Codes(): string[] {
  const section = /^## 6\. Errors\n([\s\S]*?)^## 7\./m.exec(apiDesign)?.[1]
  if (section === undefined) throw new Error('docs/api_design.md: no section 6')
  return [...new Set([...section.matchAll(/^\| `(\w+)` \| \d{3} \|/gm)].map((match) => match[1]!))]
}

describe('AnalysisResponse', () => {
  it('has exactly the fields of the Pydantic model, in order', () => {
    expect([...ANALYSIS_RESPONSE_FIELDS]).toEqual(pydanticFields('AnalysisResponse'))
  })

  it('the field list and the type have the same keys', () => {
    expectTypeOf<(typeof ANALYSIS_RESPONSE_FIELDS)[number]>().toEqualTypeOf<keyof AnalysisResponse>()
  })
})

describe('ErrorCode', () => {
  it('lists every §6 code exactly once', () => {
    const codes = section6Codes()
    expect(codes).toHaveLength(17)
    expect([...ERROR_CODES].sort()).toEqual(codes.sort())
  })

  it('the code list and the type agree', () => {
    expectTypeOf<(typeof ERROR_CODES)[number]>().toEqualTypeOf<ErrorCode>()
  })
})

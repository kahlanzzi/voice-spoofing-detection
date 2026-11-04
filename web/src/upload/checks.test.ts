import { describe, expect, it } from 'vitest'
import { fileExtension, formatDuration, formatSize, runChecks } from './checks'

// The pre-upload checks (prototype lines 898-913) against the server's rules (api/intake.py). The fileExtension table
// was checked against PurePath(name).suffix.lower()[1:] on Python 3.13 and 3.14 (Entry 143).

const LIMITS = { formats: ['wav', 'flac', 'mp3'], max_size_bytes: 20971520, max_duration_s: 60 }
const MB = 1048576

const FORMAT_WAV = { id: 'format', ok: true, wait: false, text: 'Format WAV is accepted' }
const NOT_CHECKED = { id: 'duration', ok: null, wait: false, text: 'Duration not checked: the file cannot be read as audio' }
const EMPTY = 'Empty file: it contains no data. Export the segment again and choose the new file.'
const UNREADABLE = {
  id: 'duration',
  ok: null,
  wait: false,
  text: 'Duration could not be read in the browser; the service will check it',
}

const wav = (size: number) => ({ name: 'call.wav', size })

describe('fileExtension matches the server', () => {
  it.each([
    ['a.wav', 'wav'],
    ['A.WAV', 'wav'],
    ['a.WaV', 'wav'],
    ['a b.wav', 'wav'],
    ['a..wav', 'wav'],
    ['.a.wav', 'wav'],
    ['a.exe.wav', 'wav'],
    ['a.wav.exe', 'exe'],
    ['x.tar.gz', 'gz'],
    ['a.wav ', 'wav '],
    ['wav', ''],
    ['.wav', ''],
    ['..wav', ''], // Python 3.13 says wav, 3.14 says nothing: the stricter answer (Entry 143 Q12)
    ['a.', ''],
    ['a.wav.', ''],
    ['noext', ''],
    ['', ''],
    ['.', ''],
    ['..', ''],
  ])('%j → %j', (name, extension) => {
    expect(fileExtension(name)).toBe(extension)
  })
})

describe('format', () => {
  it('all checks pass: three rows in order, valid', () => {
    expect(runChecks(wav(3 * MB), LIMITS, 12.5)).toEqual({
      checks: [
        FORMAT_WAV,
        { id: 'size', ok: true, wait: false, text: 'Size 3.0 MB is within 20 MB' },
        { id: 'duration', ok: true, wait: false, text: 'Duration 12.5 s is within 60 s' },
      ],
      valid: true,
    })
  })

  it('an uppercase extension is accepted', () => {
    const result = runChecks({ name: 'CALL.WAV', size: MB }, LIMITS, 10)
    expect(result.checks[0]).toEqual(FORMAT_WAV)
    expect(result.valid).toBe(true)
  })

  it('an unsupported format fails, in lower case, and the duration is not checked', () => {
    const result = runChecks({ name: 'call.OGG', size: MB }, LIMITS, 10)
    expect(result.checks[0]).toEqual({
      id: 'format',
      ok: false,
      wait: false,
      text: 'Unsupported format: .ogg files cannot be analyzed. Use WAV, FLAC, MP3.',
    })
    expect(result.checks[2]).toEqual(NOT_CHECKED)
    expect(result.valid).toBe(false)
  })

  it.each(['wav', '.wav', '..wav', 'noext', 'call.', ''])('%j has no extension: refused', (name) => {
    const result = runChecks({ name, size: MB }, LIMITS, 10)
    expect(result.checks[0]).toEqual({
      id: 'format',
      ok: false,
      wait: false,
      text: 'Unsupported format: this file has no extension. Use WAV, FLAC, MP3.',
    })
    expect(result.checks[2]).toEqual(NOT_CHECKED)
    expect(result.valid).toBe(false)
  })

  it.each([
    ['a.wav.exe', false],
    ['a.exe.wav', true],
    ['a.mp3.flac', true],
  ])('double extension %s: valid %s', (name, valid) => {
    expect(runChecks({ name, size: MB }, LIMITS, 10).valid).toBe(valid)
  })

  it.each(['', 'application/x-evil', 'audio/x-wav', 'text/plain'])('the MIME type %j changes nothing', (type) => {
    const accepted = new File(['x'], 'call.wav', { type })
    const refused = new File(['x'], 'call.ogg', { type })
    expect(runChecks(accepted, LIMITS, 10)).toEqual(runChecks(new File(['x'], 'call.wav'), LIMITS, 10))
    expect(runChecks(accepted, LIMITS, 10).valid).toBe(true)
    expect(runChecks(refused, LIMITS, 10).valid).toBe(false)
  })
})

describe('size', () => {
  it('exactly at the limit passes (the server refuses only above it)', () => {
    const result = runChecks(wav(LIMITS.max_size_bytes), LIMITS, 10)
    expect(result.checks[1]).toEqual({ id: 'size', ok: true, wait: false, text: 'Size 20.0 MB is within 20 MB' })
    expect(result.valid).toBe(true)
  })

  it('one byte under passes', () => {
    expect(runChecks(wav(LIMITS.max_size_bytes - 1), LIMITS, 10).valid).toBe(true)
  })

  it('one byte over fails, and the duration row stays', () => {
    const result = runChecks(wav(LIMITS.max_size_bytes + 1), LIMITS, 10)
    expect(result.checks[1]).toEqual({
      id: 'size',
      ok: false,
      wait: false,
      text: 'File too large: 20.0 MB is over the 20 MB limit. Export only the segment with the request.',
    })
    expect(result.checks[2]).toEqual({ id: 'duration', ok: true, wait: false, text: 'Duration 10 s is within 60 s' })
    expect(result.valid).toBe(false)
  })

  it('0 bytes: the Empty row replaces the size, and the duration is not checked', () => {
    expect(runChecks(wav(0), LIMITS, 10)).toEqual({
      checks: [FORMAT_WAV, { id: 'size', ok: false, wait: false, text: EMPTY }, NOT_CHECKED],
      valid: false,
    })
  })

  it('0 bytes with a bad format: both fail, still three rows', () => {
    const { checks, valid } = runChecks({ name: 'call.ogg', size: 0 }, LIMITS, 'unreadable')
    expect(checks.map((check) => check.ok)).toEqual([false, false, null])
    expect(checks[2]).toEqual(NOT_CHECKED)
    expect(valid).toBe(false)
  })

  it('a limit that is not a whole number of MB shows one decimal', () => {
    const { checks } = runChecks(wav(MB), { ...LIMITS, max_size_bytes: 20_000_000 }, 10)
    expect(checks[1]?.text).toBe('Size 1.0 MB is within 19.1 MB')
  })
})

describe('duration', () => {
  it('reading: waiting, not valid yet', () => {
    const result = runChecks(wav(MB), LIMITS, 'reading')
    expect(result.checks[2]).toEqual({ id: 'duration', ok: null, wait: true, text: 'Reading duration…' })
    expect(result.valid).toBe(false)
  })

  it('reading with a bad format: "not checked" wins', () => {
    const result = runChecks({ name: 'call.ogg', size: MB }, LIMITS, 'reading')
    expect(result.checks[2]).toEqual(NOT_CHECKED)
  })

  it('reading with 0 bytes: "not checked" wins', () => {
    expect(runChecks(wav(0), LIMITS, 'reading').checks[2]).toEqual(NOT_CHECKED)
  })

  it('unreadable: neither passed nor failed, and valid (the service will check it)', () => {
    const result = runChecks(wav(MB), LIMITS, 'unreadable')
    expect(result.checks[2]).toEqual(UNREADABLE)
    expect(result.valid).toBe(true)
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('%d counts as unreadable (prototype line 1292)', (duration) => {
    const result = runChecks(wav(MB), LIMITS, duration)
    expect(result.checks[2]).toEqual(UNREADABLE)
    expect(result.valid).toBe(true)
  })

  it('exactly at the limit passes (the server refuses only above it)', () => {
    const result = runChecks(wav(MB), LIMITS, 60)
    expect(result.checks[2]).toEqual({ id: 'duration', ok: true, wait: false, text: 'Duration 1 min 0 s is within 60 s' })
    expect(result.valid).toBe(true)
  })

  it('just over the limit fails', () => {
    const result = runChecks(wav(MB), LIMITS, 60.5)
    expect(result.checks[2]).toEqual({
      id: 'duration',
      ok: false,
      wait: false,
      text: 'Recording too long: 1 min 1 s is over the 60 s limit. Upload only the segment with the request.',
    })
    expect(result.valid).toBe(false)
  })

  it('0 fails with the Empty text: the server answers empty_file for 0 frames', () => {
    const result = runChecks(wav(MB), LIMITS, 0)
    expect(result.checks[2]).toEqual({ id: 'duration', ok: false, wait: false, text: EMPTY })
    expect(result.valid).toBe(false)
  })
})

describe('limits come from the argument', () => {
  const limits = { formats: ['ogg'], max_size_bytes: 5, max_duration_s: 2 }

  it('what passes under the default limits fails here', () => {
    const { checks, valid } = runChecks({ name: 'call.wav', size: 6 }, limits, 3)
    expect(checks[0]?.text).toBe('Unsupported format: .wav files cannot be analyzed. Use OGG.')
    expect(checks[1]?.ok).toBe(false)
    expect(valid).toBe(false)
  })

  it('the format, size and duration all follow them', () => {
    const { checks, valid } = runChecks({ name: 'call.ogg', size: 5 }, limits, 2)
    expect(checks[0]?.text).toBe('Format OGG is accepted')
    expect(checks[1]?.ok).toBe(true)
    expect(checks[2]?.text).toBe('Duration 2 s is within 2 s')
    expect(valid).toBe(true)
  })

  it('a duration over this limit fails', () => {
    expect(runChecks({ name: 'call.ogg', size: 5 }, limits, 2.1).checks[2]?.text).toBe(
      'Recording too long: 2.1 s is over the 2 s limit. Upload only the segment with the request.',
    )
  })
})

describe('formatSize (prototype line 656)', () => {
  it.each([
    [0, '0 bytes'],
    [1, '1 KB'],
    [1536, '2 KB'],
    [MB - 1, '1024 KB'],
    [MB, '1.0 MB'],
    [3.45 * MB, '3.5 MB'],
  ])('%d → %s', (bytes, text) => {
    expect(formatSize(bytes)).toBe(text)
  })
})

describe('formatDuration (prototype line 657, carry fixed)', () => {
  it.each([
    [0.04, '0 s'],
    [12.34, '12.3 s'],
    [59.96, '60 s'],
    [60, '1 min 0 s'],
    [60.4, '1 min 0 s'],
    [119.6, '2 min 0 s'],
    [125, '2 min 5 s'],
  ])('%d → %s', (seconds, text) => {
    expect(formatDuration(seconds)).toBe(text)
  })
})

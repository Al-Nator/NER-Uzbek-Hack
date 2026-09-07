import { describe, expect, it } from 'vitest'
import {
  characterCount,
  entityText,
  parseDocuments,
  segmentText,
  validateDocuments,
  validateResponse,
} from './ner'

describe('Unicode offsets and highlighting', () => {
  it('matches Python indices after emoji and preserves the entire original text', () => {
    const text = '👩🏽‍💻 Ali Тошкентда.'
    const entity = { label: 'NAME' as const, start: 5, end: 8 }
    expect(entityText(text, entity)).toBe('Ali')
    expect(characterCount('😀a')).toBe(2)
    const segments = segmentText(text, [entity])
    expect(segments.map((segment) => segment.text).join('')).toBe(text)
    expect(segments.find((segment) => segment.entities.length)?.text).toBe('Ali')
  })
  it('retains nested and crossing annotations without duplicating or dropping text', () => {
    const segments = segmentText('abcdefgh', [
      { label: 'ORG', start: 0, end: 5 },
      { label: 'GEO', start: 2, end: 7 },
      { label: 'NAME', start: 3, end: 4 },
    ])
    expect(segments.map((segment) => segment.text).join('')).toBe('abcdefgh')
    expect(segments.find((segment) => segment.text === 'd')?.entities).toHaveLength(3)
  })
})

describe('file import', () => {
  it('reads UTF-8 BOM, ignores empty JSONL lines, preserves original text and discards extra fields', () => {
    expect(
      parseDocuments(
        '\uFEFF{"hash":"a","text":"  Ali\\n😀","extra":1}\r\n\r\n{"hash":"b","text":""}',
        'batch.JSONL',
      ),
    ).toEqual([
      { hash: 'a', text: '  Ali\n😀' },
      { hash: 'b', text: '' },
    ])
  })
  it('accepts TXT as a single document without trimming or normalizing', () => {
    expect(parseDocuments(' A\r\nB ', 'sample.txt')).toEqual([{ hash: 'sample', text: ' A\r\nB ' }])
  })
  it('reports invalid JSONL line numbers and rejects unsupported formats', () => {
    expect(() => parseDocuments('{"hash":"1","text":"a"}\n\n{', 'bad.jsonl')).toThrow('строке 3')
    expect(() => parseDocuments('{}', 'data.csv')).toThrow('Поддерживаются')
  })
  it('rejects wrong shapes, null documents, duplicate hashes and malformed Unicode', () => {
    for (const contents of [
      '{}',
      '[]',
      '[null]',
      '[{"hash":1,"text":"a"}]',
      '[{"hash":"x","text":2}]',
      '[{"hash":"x","text":"a"},{"hash":"x","text":"b"}]',
      '[{"hash":"x","text":"\\ud800"}]',
    ]) {
      expect(() => parseDocuments(contents, 'invalid.json')).toThrow()
    }
  })
  it('validates document, total and per-text limits by code points', () => {
    expect(() => validateDocuments([{ hash: '1', text: '😀'.repeat(50000) }])).not.toThrow()
    expect(() => validateDocuments([{ hash: '1', text: 'a'.repeat(50001) }])).toThrow('50 000')
    expect(() =>
      validateDocuments(
        Array.from({ length: 65 }, (_, index) => ({ hash: String(index), text: '' })),
      ),
    ).toThrow('64')
    expect(() =>
      validateDocuments(
        Array.from({ length: 11 }, (_, index) => ({
          hash: String(index),
          text: 'a'.repeat(50000),
        })),
      ),
    ).toThrow('500 000')
  })
})

describe('response contract', () => {
  const documents = [{ hash: 'one', text: '😀Ali' }]
  it('accepts original API payloads with Python offsets', () => {
    const response = { data: [{ hash: 'one', entities: [{ label: 'NAME', start: 1, end: 4 }] }] }
    expect(validateResponse(response, documents)).toEqual(response)
  })
  it('rejects incomplete batches, mismatched hashes, invalid spans and duplicates', () => {
    const entity = { label: 'NAME', start: 1, end: 4 }
    const badResponses = [
      null,
      {},
      { data: [] },
      { data: [{ hash: 'wrong', entities: [] }] },
      ...[
        { ...entity, start: -1 },
        { ...entity, end: 5 },
        { ...entity, end: 1 },
        { ...entity, label: 'PER' },
        { ...entity, start: 1.5 },
        { ...entity, start: '1' },
        null,
      ].map((span) => ({ data: [{ hash: 'one', entities: [span] }] })),
      { data: [{ hash: 'one', entities: [entity, entity] }] },
    ]
    badResponses.forEach((response) =>
      expect(() => validateResponse(response, documents)).toThrow('некорректный результат'),
    )
  })
})

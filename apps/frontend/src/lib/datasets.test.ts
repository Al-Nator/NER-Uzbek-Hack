import { afterEach, describe, expect, it, vi } from 'vitest'
import { chooseRecord, loadRandomDocument, parseManifest } from './datasets'
import type { DatasetInfo, DatasetSample } from './datasets'

const train: DatasetInfo = {
  id: 'train',
  title: 'Обучающая',
  count: 13,
  source: 'train.jsonl',
  revision: 'abcdef012345',
}
const dev: DatasetInfo = {
  id: 'dev',
  title: 'Валидационная',
  count: 2,
  source: 'dev.jsonl',
  revision: 'fedcba543210',
}
afterEach(() => vi.unstubAllGlobals())

describe('dataset selection', () => {
  it('weights selection by record counts and reaches both datasets', () => {
    expect(chooseRecord([train, dev], null, () => 0)).toMatchObject({
      dataset: train,
      index: 0,
      total: 15,
    })
    expect(chooseRecord([train, dev], null, () => 12 / 15)).toMatchObject({
      dataset: train,
      index: 12,
    })
    expect(chooseRecord([train, dev], null, () => 13 / 15)).toMatchObject({
      dataset: dev,
      index: 0,
    })
    expect(chooseRecord([train, dev], null, () => 0.99999)).toMatchObject({
      dataset: dev,
      index: 1,
    })
  })
  it('never repeats the previous record and handles a one-record corpus', () => {
    const previous: DatasetSample = {
      dataset: train,
      index: 4,
      total: 15,
      document: { hash: '4', text: 'Ali' },
    }
    expect(chooseRecord([train], previous, () => 4 / 12).index).toBe(5)
    expect(chooseRecord([{ ...train, count: 1 }], { ...previous, index: 0 }, () => 0).index).toBe(0)
    expect(chooseRecord([dev], previous, () => 0)).toMatchObject({ dataset: dev, index: 0 })
    expect(() => chooseRecord([], null)).toThrow('нет текстов')
  })
  it('rejects malformed catalogs and unsafe asset paths', () => {
    for (const value of [
      null,
      {},
      { version: 1, chunkSize: 0, datasets: [train] },
      { version: 1, chunkSize: 128, datasets: [train, train] },
      { version: 1, chunkSize: 128, datasets: [{ ...train, id: '../secret' }] },
      { version: 1, chunkSize: 128, datasets: [{ ...train, revision: '../../' }] },
    ]) {
      expect(() => parseManifest(value)).toThrow()
    }
  })
})

describe('dataset loading', () => {
  it('loads only one chunk and preserves source text/hash without gold annotations', async () => {
    const record = {
      hash: 'source-hash',
      text: ' 😀Ali\n',
      entities: [{ label: 'NAME', start: 2, end: 5 }],
    }
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ version: 1, chunkSize: 128, datasets: [{ ...dev, count: 1 }] }),
        ),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify([record])))
    vi.stubGlobal('fetch', fetchMock)
    const sample = await loadRandomDocument('dev', null, new AbortController().signal)
    expect(sample.document).toEqual({ hash: record.hash, text: record.text })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe('/datasets/dev/fedcba543210/0.json')
  })
  it('rejects missing, malformed and incomplete chunks', async () => {
    for (const response of [
      new Response('', { status: 404 }),
      new Response('<html>SPA</html>'),
      new Response('[]'),
    ]) {
      vi.stubGlobal(
        'fetch',
        vi
          .fn()
          .mockResolvedValueOnce(
            new Response(
              JSON.stringify({ version: 1, chunkSize: 128, datasets: [{ ...dev, count: 1 }] }),
            ),
          )
          .mockResolvedValueOnce(response),
      )
      await expect(loadRandomDocument('dev', null, new AbortController().signal)).rejects.toThrow()
    }
  })
  it('propagates cancellation so an old request cannot become the new source', async () => {
    const controller = new AbortController()
    controller.abort()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError')))
    await expect(loadRandomDocument('all', null, controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    })
  })
})

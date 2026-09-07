import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkHealth, predict } from './api'

afterEach(() => vi.unstubAllGlobals())
const documents = [{ hash: 'example', text: '😀Ali Toshkent' }]
const signal = () => new AbortController().signal

describe('API client', () => {
  it('sends an array and preserves hashes and original text', async () => {
    const response = {
      data: [{ hash: 'example', entities: [{ label: 'NAME', start: 1, end: 4 }] }],
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response)))
    vi.stubGlobal('fetch', fetchMock)
    expect(await predict(documents, signal())).toEqual(response)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/predict')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual(documents)
  })
  it('reports model unavailability without inventing predictions', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: { code: 'model_unavailable', message: 'Model is not connected.', details: [] },
          }),
          { status: 503 },
        ),
      ),
    )
    await expect(predict(documents, signal())).rejects.toMatchObject({
      status: 503,
      message: expect.stringContaining('Модель пока не готова'),
    })
  })
  it('preserves backend validation details', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: {
              code: 'validation_error',
              details: [{ location: ['body', 0, 'hash'], message: 'Field required' }],
            },
          }),
          { status: 422 },
        ),
      ),
    )
    await expect(predict(documents, signal())).rejects.toMatchObject({
      status: 422,
      details: ['body → 0 → hash: Field required'],
    })
  })
  it('handles HTML proxy errors, invalid JSON successes and lost connections', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('<html>Bad Gateway</html>', { status: 502 })),
    )
    await expect(predict(documents, signal())).rejects.toMatchObject({ status: 502 })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>SPA fallback</html>')))
    await expect(predict(documents, signal())).rejects.toThrow('некорректный результат')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(predict(documents, signal())).rejects.toThrow('Нет соединения')
  })
  it('propagates cancellation', async () => {
    const controller = new AbortController()
    controller.abort()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError')))
    await expect(predict(documents, controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    })
  })
})

describe('health checks', () => {
  it('requires the correct payload, not just HTTP 200', async () => {
    for (const [body, status, expected] of [
      [{ status: 'ok' }, 200, 'ready'],
      ['<html>index</html>', 200, 'offline'],
      [{ error: { code: 'model_unavailable' } }, 503, 'unavailable'],
      [{}, 503, 'offline'],
    ] as const) {
      vi.stubGlobal(
        'fetch',
        vi
          .fn()
          .mockResolvedValue(
            new Response(typeof body === 'string' ? body : JSON.stringify(body), { status }),
          ),
      )
      expect(await checkHealth(signal())).toBe(expected)
    }
  })
})

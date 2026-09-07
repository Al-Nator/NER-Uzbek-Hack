import { validateResponse, type Document, type PredictResponse } from './ner'

export class ApiError extends Error {
  status: number
  details: string[]
  constructor(message: string, status = 0, details: string[] = []) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

const messages: Record<number, string> = {
  400: 'Не удалось прочитать запрос. Проверьте формат текста.',
  413: 'Превышен лимит сервера. Уменьшите текст или количество документов.',
  422: 'Сервис отклонил документы. Проверьте текст и уникальность идентификаторов.',
  500: 'Не удалось выполнить анализ. Попробуйте ещё раз.',
  502: 'Нет соединения с сервисом анализа. Убедитесь, что backend запущен.',
  503: 'Модель пока не готова. Дождитесь её загрузки и повторите анализ.',
  504: 'Сервис не успел ответить. Попробуйте отправить меньше документов.',
}

export async function predict(
  documents: Document[],
  signal: AbortSignal,
): Promise<PredictResponse> {
  let response: Response
  try {
    response = await fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(documents),
      signal,
    })
  } catch (error) {
    if (signal.aborted) throw error
    throw new ApiError('Нет соединения с сервисом анализа. Проверьте, запущен ли backend.')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const error = body && typeof body === 'object' && 'error' in body ? body.error : null
    const details =
      error && typeof error === 'object' && 'details' in error && Array.isArray(error.details)
        ? error.details.flatMap((detail: { location?: unknown; message?: unknown }) =>
            typeof detail?.message === 'string'
              ? [
                  `${Array.isArray(detail.location) ? detail.location.join(' → ') + ': ' : ''}${detail.message}`,
                ]
              : [],
          )
        : []
    throw new ApiError(
      messages[response.status] || `Ошибка сервиса (${response.status}). Попробуйте ещё раз.`,
      response.status,
      details,
    )
  }
  return validateResponse(body, documents)
}

export type Health = 'checking' | 'ready' | 'unavailable' | 'offline'
export async function checkHealth(signal: AbortSignal): Promise<Health> {
  try {
    const response = await fetch('/healthz', { signal, cache: 'no-store' })
    const body = await response.json().catch(() => null)
    if (response.ok && body?.status === 'ok') return 'ready'
    if (response.status === 503 && body?.error?.code === 'model_unavailable') return 'unavailable'
    return 'offline'
  } catch {
    return 'offline'
  }
}

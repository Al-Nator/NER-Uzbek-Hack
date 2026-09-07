export const LABELS = ['NAME', 'ORG', 'GEO'] as const
export type Label = (typeof LABELS)[number]
export type Document = { hash: string; text: string }
export type Entity = { label: Label; start: number; end: number }
export type Prediction = { hash: string; entities: Entity[] }
export type PredictResponse = { data: Prediction[] }
export type Analysis = { documents: Document[]; response: PredictResponse; duration: number }

export const labelInfo = {
  NAME: { title: 'Люди', singular: 'Человек', description: 'Имена и фамилии', className: 'person' },
  ORG: {
    title: 'Организации',
    singular: 'Организация',
    description: 'Компании и учреждения',
    className: 'organization',
  },
  GEO: {
    title: 'География',
    singular: 'Место',
    description: 'Страны, города и регионы',
    className: 'location',
  },
} as const

// Offsets in the API are Python Unicode code points, not JavaScript UTF-16 units.
export const characterCount = (text: string) => Array.from(text).length
export const entityText = (text: string, entity: Entity) =>
  Array.from(text).slice(entity.start, entity.end).join('')
export const LIMITS = { documents: 64, text: 50_000, total: 500_000, fileBytes: 8 * 1024 * 1024 }

export function validateDocuments(documents: Document[]): void {
  if (!documents.length) throw new Error('Добавьте хотя бы один документ.')
  if (documents.length > LIMITS.documents)
    throw new Error(`В одном пакете может быть не больше ${LIMITS.documents} документов.`)
  const hashes = new Set<string>()
  let total = 0
  for (const document of documents) {
    if (
      !document ||
      typeof document.hash !== 'string' ||
      !document.hash ||
      typeof document.text !== 'string'
    ) {
      throw new Error('Каждый документ должен содержать непустой строковый hash и строковый text.')
    }
    if (
      [document.text, document.hash].some((value) =>
        /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value),
      )
    ) {
      throw new Error('В документе есть некорректный символ Unicode.')
    }
    if (hashes.has(document.hash))
      throw new Error(`Идентификатор «${document.hash}» повторяется. Hash должен быть уникальным.`)
    hashes.add(document.hash)
    const length = characterCount(document.text)
    if (length > LIMITS.text)
      throw new Error(`Документ «${document.hash}» превышает лимит в 50 000 символов.`)
    total += length
  }
  if (total > LIMITS.total) throw new Error('Общий объём пакета превышает 500 000 символов.')
}

export function parseDocuments(content: string, filename: string): Document[] {
  const clean = content.replace(/^\uFEFF/, '')
  let documents: Document[]
  if (/\.txt$/i.test(filename)) {
    documents = [{ hash: filename.replace(/\.txt$/i, '') || 'document', text: clean }]
  } else if (/\.jsonl$/i.test(filename)) {
    documents = clean.split(/\r?\n/).flatMap((line, index) => {
      if (!line.trim()) return []
      try {
        return [JSON.parse(line)]
      } catch {
        throw new Error(`Ошибка JSON в строке ${index + 1}. Проверьте содержимое файла.`)
      }
    })
  } else if (/\.json$/i.test(filename)) {
    try {
      documents = JSON.parse(clean)
    } catch {
      throw new Error('Не удалось прочитать JSON. Проверьте синтаксис файла.')
    }
    if (!Array.isArray(documents))
      throw new Error('JSON должен содержать массив документов: [{ "hash": "…", "text": "…" }].')
  } else {
    throw new Error('Поддерживаются файлы .txt, .json и .jsonl.')
  }
  validateDocuments(documents)
  return documents.map(({ hash, text }) => ({ hash, text }))
}

export function validateResponse(body: unknown, documents: Document[]): PredictResponse {
  const invalid = () =>
    new Error('Сервис вернул некорректный результат. Повторите анализ или проверьте API.')
  if (
    !body ||
    typeof body !== 'object' ||
    !('data' in body) ||
    !Array.isArray(body.data) ||
    body.data.length !== documents.length
  )
    throw invalid()
  body.data.forEach((prediction: unknown, index: number) => {
    if (
      !prediction ||
      typeof prediction !== 'object' ||
      !('hash' in prediction) ||
      prediction.hash !== documents[index].hash ||
      !('entities' in prediction) ||
      !Array.isArray(prediction.entities)
    )
      throw invalid()
    const length = characterCount(documents[index].text)
    const seen = new Set<string>()
    for (const entity of prediction.entities) {
      if (
        !entity ||
        !LABELS.includes(entity.label) ||
        !Number.isInteger(entity.start) ||
        !Number.isInteger(entity.end) ||
        entity.start < 0 ||
        entity.end <= entity.start ||
        entity.end > length
      )
        throw invalid()
      const key = `${entity.label}:${entity.start}:${entity.end}`
      if (seen.has(key)) throw invalid()
      seen.add(key)
    }
  })
  return body as PredictResponse
}

export type TextSegment = { start: number; end: number; text: string; entities: Entity[] }

// Split at all boundaries: nested and overlapping spans remain visible and inspectable.
export function segmentText(text: string, entities: Entity[]): TextSegment[] {
  const characters = Array.from(text)
  const events = new Map<number, { starting: Entity[]; ending: Entity[] }>()
  const at = (position: number) => {
    let event = events.get(position)
    if (!event) {
      event = { starting: [], ending: [] }
      events.set(position, event)
    }
    return event
  }
  for (const entity of entities) {
    at(entity.start).starting.push(entity)
    at(entity.end).ending.push(entity)
  }
  const points = [...new Set([0, characters.length, ...events.keys()])].sort((a, b) => a - b)
  const active = new Set<Entity>()
  return points.slice(0, -1).map((start, index) => {
    const end = points[index + 1]
    const event = events.get(start)
    event?.ending.forEach((entity) => active.delete(entity))
    event?.starting.forEach((entity) => active.add(entity))
    return { start, end, text: characters.slice(start, end).join(''), entities: [...active] }
  })
}

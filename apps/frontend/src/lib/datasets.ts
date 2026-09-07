import type { Document } from './ner'
import { validateDocuments } from './ner'

export type DatasetId = 'train' | 'dev'
export type DatasetSelection = 'all' | DatasetId
export type DatasetInfo = {
  id: DatasetId
  title: string
  count: number
  source: string
  revision: string
}
export type DatasetManifest = { version: 1; chunkSize: number; datasets: DatasetInfo[] }
export type DatasetSample = {
  document: Document
  dataset: DatasetInfo
  index: number
  total: number
}

export function parseManifest(value: unknown): DatasetManifest {
  if (
    !value ||
    typeof value !== 'object' ||
    !('version' in value) ||
    value.version !== 1 ||
    !('chunkSize' in value) ||
    !Number.isInteger(value.chunkSize) ||
    Number(value.chunkSize) < 1 ||
    Number(value.chunkSize) > 1024 ||
    !('datasets' in value) ||
    !Array.isArray(value.datasets) ||
    !value.datasets.length
  )
    throw new Error('Не удалось прочитать каталог датасетов.')
  const ids = new Set<string>()
  for (const entry of value.datasets) {
    if (
      !entry ||
      !['train', 'dev'].includes(entry.id) ||
      ids.has(entry.id) ||
      typeof entry.title !== 'string' ||
      !Number.isSafeInteger(entry.count) ||
      entry.count < 1 ||
      typeof entry.source !== 'string' ||
      typeof entry.revision !== 'string' ||
      !/^[a-f0-9]{12}$/.test(entry.revision)
    )
      throw new Error('Каталог датасетов имеет неверный формат.')
    ids.add(entry.id)
  }
  return value as DatasetManifest
}

// Choose uniformly over records, not over datasets. Avoid the previous record without retry loops.
export function chooseRecord(
  datasets: DatasetInfo[],
  previous?: DatasetSample | null,
  random = Math.random,
) {
  const total = datasets.reduce((sum, item) => sum + item.count, 0)
  if (!Number.isSafeInteger(total) || total < 1)
    throw new Error('В выбранном датасете нет текстов.')
  let excluded = -1
  let offset = 0
  for (const dataset of datasets) {
    if (
      previous?.dataset.id === dataset.id &&
      previous.dataset.revision === dataset.revision &&
      previous.index >= 0 &&
      previous.index < dataset.count
    )
      excluded = offset + previous.index
    offset += dataset.count
  }
  const canExclude = excluded >= 0 && total > 1
  let position = Math.floor(random() * (total - Number(canExclude)))
  if (canExclude && position >= excluded) position += 1
  for (const dataset of datasets) {
    if (position < dataset.count) return { dataset, index: position, total }
    position -= dataset.count
  }
  throw new Error('Не удалось выбрать текст.')
}

async function readJson(url: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(url, {
    signal,
    cache: url.endsWith('/manifest.json') ? 'no-cache' : 'default',
  })
  if (!response.ok) throw new Error('Датасет недоступен. Попробуйте загрузить текст ещё раз.')
  return response.json().catch(() => {
    throw new Error('Не удалось прочитать датасет. Проверьте его подготовку.')
  })
}

export async function loadRandomDocument(
  selection: DatasetSelection,
  previous: DatasetSample | null,
  signal: AbortSignal,
): Promise<DatasetSample> {
  try {
    const manifest = parseManifest(await readJson('/datasets/manifest.json', signal))
    const candidates = manifest.datasets.filter(
      (dataset) => selection === 'all' || dataset.id === selection,
    )
    const chosen = chooseRecord(candidates, previous)
    const chunk = Math.floor(chosen.index / manifest.chunkSize)
    const records = await readJson(
      `/datasets/${chosen.dataset.id}/${chosen.dataset.revision}/${chunk}.json`,
      signal,
    )
    const expected = Math.min(manifest.chunkSize, chosen.dataset.count - chunk * manifest.chunkSize)
    if (!Array.isArray(records) || records.length !== expected)
      throw new Error('Фрагмент датасета имеет неверный формат.')
    const document = records[chosen.index % manifest.chunkSize] as Document
    validateDocuments([document])
    return { ...chosen, document: { hash: document.hash, text: document.text } }
  } catch (error) {
    if (signal.aborted) throw error
    if (error instanceof TypeError)
      throw new Error('Не удалось загрузить текст. Проверьте соединение и попробуйте ещё раз.')
    throw error
  }
}

import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('../', import.meta.url))
const source = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve(root, '../../ner_uz_hackathon_participant/data')
const destination = path.join(root, 'public/datasets')
const chunkSize = 128
const manifest = { version: 1, chunkSize, datasets: [] }

for (const [id, title] of [
  ['train', 'Обучающая выборка'],
  ['dev', 'Валидационная выборка'],
]) {
  const input = await readFile(path.join(source, `${id}.jsonl`), 'utf8')
  const hashes = new Set()
  const records = input
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      const { hash, text } = JSON.parse(line)
      if (typeof hash !== 'string' || !hash || typeof text !== 'string' || hashes.has(hash)) {
        throw new Error(`Invalid record in ${id}.jsonl at line ${index + 1}`)
      }
      hashes.add(hash)
      // Эталонная разметка не попадает в корпус браузера и запрос предсказания.
      return { hash, text }
    })
  if (!records.length) throw new Error(`Empty dataset: ${id}`)
  const digest = createHash('sha256').update(input).digest('hex')
  const revision = digest.slice(0, 12)
  await mkdir(path.join(destination, id, revision), { recursive: true })
  for (let offset = 0; offset < records.length; offset += chunkSize) {
    await writeFile(
      path.join(destination, id, revision, `${offset / chunkSize}.json`),
      JSON.stringify(records.slice(offset, offset + chunkSize)),
    )
  }
  manifest.datasets.push({
    id,
    title,
    count: records.length,
    source: `${id}.jsonl`,
    revision,
    sha256: digest,
  })
}
// Публикуем манифест после записи всех фрагментов.
await writeFile(path.join(destination, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
console.log(manifest.datasets.map(({ id, count }) => `${id}: ${count} texts`).join(', '))

import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/healthz', (route) => route.fulfill({ json: { status: 'ok' } }))
})

test('cancelling a slow request does not resubmit the form or restore late results', async ({
  page,
}) => {
  let calls = 0
  let delivered = 0
  let releaseResponse!: () => void
  const responseGate = new Promise<void>((resolve) => {
    releaseResponse = resolve
  })
  await page.route('**/api/v1/predict', async (route) => {
    calls += 1
    await responseGate
    await route
      .fulfill({
        json: { data: [{ hash: 'text-001', entities: [{ label: 'NAME', start: 0, end: 3 }] }] },
      })
      .catch(() => {})
    delivered += 1
  })
  await page.goto('/')
  await page.getByLabel('Текст на узбекском языке').fill('Ali')
  await page.getByRole('button', { name: 'Анализировать', exact: true }).click()
  await expect.poll(() => calls).toBe(1)
  await page.getByRole('button', { name: 'Отменить', exact: true }).click()
  releaseResponse()
  await expect.poll(() => delivered).toBe(1)
  expect(calls).toBe(1)
  await expect(page.locator('.entity-mark')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Анализировать', exact: true })).toBeEnabled()
})

test('editing input clears previous predictions and export', async ({ page }) => {
  await page.route('**/api/v1/predict', (route) =>
    route.fulfill({
      json: { data: [{ hash: 'text-001', entities: [{ label: 'NAME', start: 1, end: 4 }] }] },
    }),
  )
  await page.goto('/')
  await page.getByLabel('Текст на узбекском языке').fill('😀Ali')
  await page.getByRole('button', { name: 'Анализировать', exact: true }).click()
  await expect(page.locator('.entity-mark')).toContainText('Ali')
  await expect(page.getByRole('button', { name: 'Экспорт JSON' })).toBeVisible()
  await page.getByLabel('Текст на узбекском языке').fill('Другой текст')
  await expect(page.locator('.entity-mark')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Экспорт JSON' })).toHaveCount(0)
})

test('an unavailable model shows an error without a fabricated result', async ({ page }) => {
  await page.route('**/api/v1/predict', (route) =>
    route.fulfill({
      status: 503,
      json: {
        error: { code: 'model_unavailable', message: 'Model is not connected.', details: [] },
      },
    }),
  )
  await page.goto('/')
  await page.getByLabel('Текст на узбекском языке').fill('Ali')
  await page.getByRole('button', { name: 'Анализировать', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Модель пока не готова')
  await expect(page.locator('.entity-mark')).toHaveCount(0)
})

test('mobile controls fit and help remains keyboard-accessible', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 })
  await page.goto('/')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  for (const control of [
    page.getByRole('button', { name: 'Анализировать', exact: true }),
    page.getByRole('tab', { name: 'Файл', exact: true }),
  ]) {
    const box = await control.boundingBox()
    expect(box).not.toBeNull()
    expect(box!.x).toBeGreaterThanOrEqual(0)
    expect(box!.x + box!.width).toBeLessThanOrEqual(320)
  }
  await page.getByRole('button', { name: 'Как это работает' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('three modes use real corpus texts, preserve source hashes, and retain a custom draft', async ({
  page,
}) => {
  let submitted: unknown
  await page.route('**/api/v1/predict', (route) => {
    submitted = route.request().postDataJSON()
    return route.fulfill({
      json: { data: (submitted as { hash: string }[]).map(({ hash }) => ({ hash, entities: [] })) },
    })
  })
  await page.goto('/')
  await expect(page.getByRole('tab')).toHaveCount(3)
  await expect(page.getByLabel('Текст на узбекском языке')).toHaveValue('')
  await expect(page.getByText('Примеры', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Анализировать', exact: true })).toBeDisabled()
  await page.getByLabel('Текст на узбекском языке').fill('Мой черновик')
  await page.getByRole('tab', { name: 'Случайный текст', exact: true }).click()
  await page.getByLabel('Выборка датасета').selectOption('dev')
  await expect(page.locator('.dataset-meta')).toContainText('dev.jsonl')
  const hash = await page.locator('.dataset-hash code').innerText()
  const text = await page.getByLabel('Случайный текст из датасета').inputValue()
  const manifest = await (await page.request.get('/datasets/manifest.json')).json()
  const metadata = await page.locator('.dataset-meta').innerText()
  const position = Number(metadata.match(/Текст ([\d\s]+) из/)![1].replace(/\s/g, '')) - 1
  const dev = manifest.datasets.find((entry: { id: string }) => entry.id === 'dev')
  const records = await (
    await page.request.get(
      `/datasets/dev/${dev.revision}/${Math.floor(position / manifest.chunkSize)}.json`,
    )
  ).json()
  expect(records[position % manifest.chunkSize]).toEqual({ hash, text })
  await page.getByRole('button', { name: 'Анализировать', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Экспорт JSON' })).toBeVisible()
  expect(submitted).toEqual([{ hash, text }])
  await page.getByRole('button', { name: 'Другой текст', exact: true }).click()
  await expect(page.locator('.dataset-hash code')).not.toHaveText(hash)
  await expect(page.getByRole('button', { name: 'Экспорт JSON' })).toHaveCount(0)
  await page.getByRole('tab', { name: 'Свой текст', exact: true }).click()
  await expect(page.getByLabel('Текст на узбекском языке')).toHaveValue('Мой черновик')
  await page.getByRole('tab', { name: 'Файл', exact: true }).click()
  await page.locator('input[type=file]').setInputFiles({
    name: 'batch.json',
    mimeType: 'application/json',
    buffer: Buffer.from('[{"hash":"file-1","text":"Ali Toshkent"}]'),
  })
  await page.getByRole('button', { name: 'Анализировать', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Экспорт JSON' })).toBeVisible()
  expect(submitted).toEqual([{ hash: 'file-1', text: 'Ali Toshkent' }])
})

test('switching away from a loading dataset cannot overwrite the custom text', async ({ page }) => {
  const manifest = await (await page.request.get('/datasets/manifest.json')).json()
  let release!: () => void
  let requested = false
  let delivered = false
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/datasets/manifest.json', async (route) => {
    requested = true
    await gate
    await route.fulfill({ json: manifest }).catch(() => {})
    delivered = true
  })
  await page.goto('/')
  await page.getByRole('tab', { name: 'Случайный текст', exact: true }).click()
  await expect.poll(() => requested).toBe(true)
  await page.getByRole('tab', { name: 'Свой текст', exact: true }).click()
  await page.getByLabel('Текст на узбекском языке').fill('Сохранить этот текст')
  release()
  await expect.poll(() => delivered).toBe(true)
  await expect(page.getByLabel('Текст на узбекском языке')).toHaveValue('Сохранить этот текст')
  await expect(page.getByLabel('Случайный текст из датасета')).toHaveCount(0)
})

test('dataset failures remain retryable and three tabs support keyboard navigation', async ({
  page,
}) => {
  await page.route('**/datasets/manifest.json', (route) => route.fulfill({ status: 503, body: '' }))
  await page.goto('/')
  await page.getByRole('tab', { name: 'Свой текст', exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Случайный текст', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await expect(page.getByRole('alert')).toContainText('Датасет недоступен')
  await expect(page.getByRole('button', { name: 'Получить текст' })).toBeEnabled()
  await page.keyboard.press('End')
  await expect(page.getByRole('tab', { name: 'Файл', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
})

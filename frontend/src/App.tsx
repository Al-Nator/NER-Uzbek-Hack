import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import type { DragEvent, FormEvent, ReactNode } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Braces,
  Building2,
  Check,
  ChevronDown,
  CircleHelp,
  Clipboard,
  FileJson,
  FileText,
  LoaderCircle,
  MapPin,
  Maximize2,
  RefreshCw,
  ScanText,
  Shuffle,
  Database,
  Trash2,
  Upload,
  UserRound,
  X,
} from 'lucide-react'
import { ApiError, checkHealth, predict } from './lib/api'
import type { Health } from './lib/api'
import { loadRandomDocument } from './lib/datasets'
import type { DatasetSample, DatasetSelection } from './lib/datasets'
import {
  characterCount,
  entityText,
  LABELS,
  labelInfo,
  LIMITS,
  parseDocuments,
  segmentText,
  validateDocuments,
} from './lib/ner'
import type { Analysis, Document, Entity, Label } from './lib/ner'

const formatNumber = (number: number) => new Intl.NumberFormat('ru-RU').format(number)
const entityKey = (entity: Entity) => `${entity.label}-${entity.start}-${entity.end}`
const healthLabels: Record<Health, string> = {
  checking: 'Проверяем соединение',
  ready: 'Модель готова к анализу',
  unavailable: 'Модель ещё не готова',
  offline: 'Сервис недоступен',
}
const typeIcons = { NAME: UserRound, ORG: Building2, GEO: MapPin }
const INPUT_MODES = [
  { id: 'text', title: 'Свой текст', icon: FileText },
  { id: 'dataset', title: 'Случайный текст', icon: Shuffle },
  { id: 'batch', title: 'Файл', icon: Upload },
] as const
type InputMode = (typeof INPUT_MODES)[number]['id']

function Modal({
  title,
  children,
  close,
}: {
  title: string
  children: ReactNode
  close: () => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  useEffect(() => {
    const element = dialog.current
    element?.showModal()
    return () => element?.close()
  }, [])
  return (
    <dialog
      ref={dialog}
      className="modal"
      aria-labelledby={titleId}
      onCancel={close}
      onClick={(event) => {
        if (event.target === event.currentTarget) close()
      }}
    >
      <div className="modal-heading">
        <h2 id={titleId}>{title}</h2>
        <button className="icon-button" onClick={close} aria-label="Закрыть">
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  )
}

function App() {
  const [mode, setMode] = useState<InputMode>('text')
  const [text, setText] = useState('')
  const [datasetSource, setDatasetSource] = useState<DatasetSelection>('all')
  const [sample, setSample] = useState<DatasetSample | null>(null)
  const [sampleBusy, setSampleBusy] = useState(false)
  const sampleRequestRef = useRef<AbortController | null>(null)
  const [batch, setBatch] = useState<Document[]>([])
  const [filename, setFilename] = useState('')
  const [inputIndex, setInputIndex] = useState(0)
  const [uploadBusy, setUploadBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<Error | null>(null)
  const [health, setHealth] = useState<Health>('checking')
  const [healthTick, setHealthTick] = useState(0)
  const [outputIndex, setOutputIndex] = useState(0)
  const [outputMode, setOutputMode] = useState<'text' | 'json'>('text')
  const [enabledLabels, setEnabledLabels] = useState<Label[]>([...LABELS])
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)
  const [modal, setModal] = useState<'help' | 'expanded' | null>(null)
  const [toast, setToast] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const resultRef = useRef<HTMLDivElement>(null)
  const requestRef = useRef<AbortController | null>(null)
  const uploadVersion = useRef(0)
  const runRef = useRef<() => void>(() => {})

  useEffect(() => {
    let disposed = false
    let controller: AbortController | null = null
    const check = async () => {
      controller?.abort()
      controller = new AbortController()
      const timeout = window.setTimeout(() => controller?.abort(), 5000)
      const next = await checkHealth(controller.signal)
      window.clearTimeout(timeout)
      if (!disposed) setHealth(next)
    }
    void check()
    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') void check()
    }, 30000)
    return () => {
      disposed = true
      controller?.abort()
      window.clearInterval(interval)
    }
  }, [healthTick])

  useEffect(
    () => () => {
      requestRef.current?.abort()
      sampleRequestRef.current?.abort()
      uploadVersion.current += 1
    },
    [],
  )
  useEffect(() => {
    if (!busy) return
    const start = performance.now()
    const timer = window.setInterval(
      () => setElapsed(Math.floor((performance.now() - start) / 1000)),
      1000,
    )
    return () => window.clearInterval(timer)
  }, [busy])
  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(''), 3500)
    return () => window.clearTimeout(timer)
  }, [toast])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        (event.metaKey || event.ctrlKey) &&
        event.key === 'Enter' &&
        !document.querySelector('dialog[open]')
      ) {
        event.preventDefault()
        runRef.current()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const invalidate = useCallback(() => {
    requestRef.current?.abort()
    requestRef.current = null
    setBusy(false)
    setAnalysis(null)
    setError(null)
    setSelectedEntity(null)
    setOutputIndex(0)
  }, [])

  const changeText = (value: string) => {
    invalidate()
    setText(value)
  }
  const loadSample = async (selection: DatasetSelection) => {
    sampleRequestRef.current?.abort()
    const controller = new AbortController()
    sampleRequestRef.current = controller
    const previous = sample
    invalidate()
    setSample(null)
    setSampleBusy(true)
    let timedOut = false
    const timeout = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, 15000)
    try {
      const next = await loadRandomDocument(selection, previous, controller.signal)
      if (sampleRequestRef.current === controller && !controller.signal.aborted) setSample(next)
    } catch (cause) {
      if (sampleRequestRef.current !== controller) return
      if (timedOut) setError(new Error('Датасет загружается слишком долго. Попробуйте ещё раз.'))
      else if (!controller.signal.aborted)
        setError(
          cause instanceof Error ? cause : new Error('Не удалось загрузить текст из датасета.'),
        )
    } finally {
      window.clearTimeout(timeout)
      if (sampleRequestRef.current === controller) {
        sampleRequestRef.current = null
        setSampleBusy(false)
      }
    }
  }
  const switchMode = (value: InputMode) => {
    if (mode === value) return
    invalidate()
    uploadVersion.current += 1
    setUploadBusy(false)
    sampleRequestRef.current?.abort()
    sampleRequestRef.current = null
    setSampleBusy(false)
    setMode(value)
    if (value === 'dataset' && !sample) void loadSample(datasetSource)
  }
  const uploadFile = async (file: File) => {
    const version = ++uploadVersion.current
    invalidate()
    setUploadBusy(true)
    try {
      if (file.size > LIMITS.fileBytes)
        throw new Error('Файл слишком большой. Максимальный размер — 8 МБ.')
      const buffer = await file.arrayBuffer()
      let content: string
      try {
        content = new TextDecoder('utf-8', { fatal: true }).decode(buffer)
      } catch {
        throw new Error('Сохраните файл в кодировке UTF-8 и загрузите его ещё раз.')
      }
      const documents = parseDocuments(content, file.name)
      if (version !== uploadVersion.current) return
      setBatch(documents)
      setFilename(file.name)
      setInputIndex(0)
      setToast(`Загружено документов: ${documents.length}`)
    } catch (cause) {
      if (version === uploadVersion.current) {
        setBatch([])
        setFilename('')
        setError(cause instanceof Error ? cause : new Error('Не удалось прочитать файл.'))
      }
    } finally {
      if (version === uploadVersion.current) setUploadBusy(false)
    }
  }

  const handleDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    setDragging(false)
    if (event.dataTransfer.files.length > 1) {
      setError(
        new Error('Загрузите один файл. Несколько документов можно объединить в JSON или JSONL.'),
      )
      return
    }
    const file = event.dataTransfer.files[0]
    if (file) void uploadFile(file)
  }

  const run = async (event?: FormEvent) => {
    event?.preventDefault()
    if (busy || uploadBusy || sampleBusy || !inputReady) return
    const documents =
      mode === 'text'
        ? [{ hash: 'text-001', text }]
        : mode === 'dataset' && sample
          ? [{ ...sample.document }]
          : batch.map((document) => ({ ...document }))
    try {
      validateDocuments(documents)
    } catch (cause) {
      setError(cause as Error)
      return
    }
    invalidate()
    const controller = new AbortController()
    requestRef.current = controller
    setBusy(true)
    setElapsed(0)
    setEnabledLabels([...LABELS])
    setOutputMode('text')
    const start = performance.now()
    let timedOut = false
    const timeout = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, 180000)
    try {
      const response = await predict(documents, controller.signal)
      if (requestRef.current !== controller) return
      setAnalysis({ documents, response, duration: performance.now() - start })
      setHealth('ready')
    } catch (cause) {
      if (requestRef.current !== controller) return
      if (timedOut)
        setError(new Error('Анализ занял больше трёх минут. Уменьшите пакет и попробуйте ещё раз.'))
      else if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause : new Error('Не удалось выполнить анализ.'))
        setHealthTick((tick) => tick + 1)
      }
    } finally {
      window.clearTimeout(timeout)
      if (requestRef.current === controller) {
        requestRef.current = null
        setBusy(false)
      }
    }
  }
  runRef.current = () => {
    void run()
  }

  const activeDocument = analysis?.documents[outputIndex]
  const entities = useMemo(
    () =>
      analysis?.response.data[outputIndex]?.entities
        .slice()
        .sort((a, b) => a.start - b.start || a.end - b.end) ?? [],
    [analysis, outputIndex],
  )
  const visibleEntities = useMemo(
    () => entities.filter((entity) => enabledLabels.includes(entity.label)),
    [entities, enabledLabels],
  )
  const segments = useMemo(
    () => segmentText(activeDocument?.text ?? '', visibleEntities),
    [activeDocument, visibleEntities],
  )
  const totalEntities =
    analysis?.response.data.reduce((sum, item) => sum + item.entities.length, 0) ?? 0
  const counts = useMemo(
    () =>
      Object.fromEntries(
        LABELS.map((label) => [label, entities.filter((entity) => entity.label === label).length]),
      ) as Record<Label, number>,
    [entities],
  )
  const inputText =
    mode === 'text'
      ? text
      : mode === 'dataset'
        ? (sample?.document.text ?? '')
        : (batch[inputIndex]?.text ?? '')
  const inputReady =
    mode === 'text' ? Boolean(text.trim()) : mode === 'dataset' ? Boolean(sample) : batch.length > 0
  const length = characterCount(inputText)
  const tooLong = length > LIMITS.text
  const totalCharacters = batch.reduce((sum, item) => sum + characterCount(item.text), 0)

  const toggleLabel = (label: Label) => {
    setEnabledLabels((current) =>
      current.includes(label) ? current.filter((item) => item !== label) : [...current, label],
    )
    setSelectedEntity(null)
  }

  const selectEntity = (entity: Entity) => {
    const key = entityKey(entity)
    setSelectedEntity((previous) => (previous === key ? null : key))
    const target = resultRef.current?.querySelector(`[data-entities~="${key}"]`)
    target?.scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 'instant'
        : 'smooth',
      block: 'nearest',
    })
  }

  const download = (content: string, name: string) => {
    const url = URL.createObjectURL(new Blob([content], { type: 'application/json;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = name
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const copy = async () => {
    if (!analysis) return
    try {
      await navigator.clipboard.writeText(JSON.stringify(analysis.response, null, 2))
      setToast('JSON скопирован в буфер обмена')
    } catch {
      setToast('Копирование недоступно. Скачайте результат кнопкой «Экспорт JSON».')
    }
  }

  const renderAnnotated = () => (
    <div className="annotated-text" lang="uz" dir="auto">
      {segments.map((segment) =>
        segment.entities.length ? (
          <button
            type="button"
            key={segment.start}
            className={`entity-mark ${labelInfo[segment.entities[0].label].className} ${segment.entities.length > 1 ? 'overlapping' : ''} ${segment.entities.some((entity) => entityKey(entity) === selectedEntity) ? 'selected' : ''}`}
            data-entities={segment.entities.map(entityKey).join(' ')}
            aria-pressed={segment.entities.some((entity) => entityKey(entity) === selectedEntity)}
            title={segment.entities
              .map(
                (entity) => `${labelInfo[entity.label].singular} · ${entity.start}–${entity.end}`,
              )
              .join('\n')}
            onClick={() => {
              const current = segment.entities.findIndex(
                (entity) => entityKey(entity) === selectedEntity,
              )
              selectEntity(segment.entities[(current + 1) % segment.entities.length])
            }}
          >
            {segment.text}
            <span className="inline-label">
              {segment.entities.map((entity) => entity.label).join(' / ')}
            </span>
          </button>
        ) : (
          <span key={segment.start}>{segment.text}</span>
        ),
      )}
      {!activeDocument?.text && <span className="muted">Пустой документ — сущностей нет.</span>}
    </div>
  )

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Перейти к анализу
      </a>
      <div className="main-shell">
        <header className="topbar">
          <a className="app-title" href="#workspace">
            <span className="identity-mark" aria-hidden="true" /> Uzbek NER
          </a>
          <nav className="topbar-actions" aria-label="Навигация">
            <a href="/docs" target="_blank" rel="noreferrer">
              API <ArrowUpRight size={14} />
            </a>
            <button
              className="icon-button"
              aria-label="Как это работает"
              onClick={() => setModal('help')}
            >
              <CircleHelp size={18} />
            </button>
          </nav>
        </header>
        <main id="workspace" className="workspace">
          <section className="page-intro">
            <div className="intro-copy">
              <span className="intro-kicker">O‘ZBEK TILI</span>
              <h1>Анализ текста</h1>
              <p>
                Люди, организации и места
                <br />в узбекском тексте.
              </p>
            </div>
            <img
              className="heritage-image"
              src="/uzbek-still-life.png"
              alt="Узбекский плов в расписной керамике на узорном ковре"
              width="1536"
              height="1024"
              fetchPriority="high"
            />
          </section>
          <div className="studio-grid">
            <form
              className="panel input-panel"
              onSubmit={(event) => {
                void run(event)
              }}
            >
              <div className="panel-heading">
                <div className="panel-title">
                  <h2>Исходный текст</h2>
                </div>
                <div className="panel-tools">
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => {
                      invalidate()
                      if (mode === 'text') {
                        setText('')
                        inputRef.current?.focus()
                      } else if (mode === 'dataset') {
                        sampleRequestRef.current?.abort()
                        sampleRequestRef.current = null
                        setSampleBusy(false)
                        setSample(null)
                      } else {
                        uploadVersion.current += 1
                        setUploadBusy(false)
                        setBatch([])
                        setFilename('')
                      }
                    }}
                    disabled={
                      mode === 'text'
                        ? !text
                        : mode === 'dataset'
                          ? !sample && !sampleBusy
                          : !batch.length && !uploadBusy
                    }
                    aria-label="Очистить исходный текст"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              <div className="input-tabs" role="tablist" aria-label="Источник документов">
                {INPUT_MODES.map((tab, index) => {
                  const Icon = tab.icon
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      role="tab"
                      aria-selected={mode === tab.id}
                      aria-controls="input-content"
                      id={`${tab.id}-tab`}
                      tabIndex={mode === tab.id ? 0 : -1}
                      className={mode === tab.id ? 'current' : ''}
                      onClick={() => switchMode(tab.id)}
                      onKeyDown={(event) => {
                        let next = index
                        if (event.key === 'ArrowRight') next = (index + 1) % INPUT_MODES.length
                        else if (event.key === 'ArrowLeft')
                          next = (index + INPUT_MODES.length - 1) % INPUT_MODES.length
                        else if (event.key === 'Home') next = 0
                        else if (event.key === 'End') next = INPUT_MODES.length - 1
                        else return
                        event.preventDefault()
                        switchMode(INPUT_MODES[next].id)
                        document.getElementById(`${INPUT_MODES[next].id}-tab`)?.focus()
                      }}
                    >
                      <Icon size={16} />
                      <span>{tab.title}</span>
                    </button>
                  )
                })}
              </div>
              <div
                id="input-content"
                role="tabpanel"
                aria-labelledby={`${mode}-tab`}
                className="input-content"
              >
                {mode === 'text' ? (
                  <>
                    <label htmlFor="source-text" className="sr-only">
                      Текст на узбекском языке
                    </label>
                    <textarea
                      id="source-text"
                      ref={inputRef}
                      value={text}
                      onChange={(event) => changeText(event.target.value)}
                      placeholder="Вставьте текст на узбекском языке…"
                      spellCheck={false}
                      lang="uz"
                      dir="auto"
                      aria-describedby="character-limit"
                      aria-invalid={tooLong}
                    />
                  </>
                ) : mode === 'dataset' ? (
                  <div className="dataset-panel" aria-busy={sampleBusy}>
                    <div className="dataset-controls">
                      <label className="dataset-select">
                        <span>Выборка</span>
                        <select
                          aria-label="Выборка датасета"
                          value={datasetSource}
                          onChange={(event) => {
                            const selection = event.target.value as DatasetSelection
                            setDatasetSource(selection)
                            void loadSample(selection)
                          }}
                        >
                          <option value="all">Все датасеты</option>
                          <option value="train">Обучающая · train</option>
                          <option value="dev">Валидационная · dev</option>
                        </select>
                        <ChevronDown size={14} />
                      </label>
                      <button
                        type="button"
                        className="button dataset-shuffle"
                        onClick={() => {
                          void loadSample(datasetSource)
                        }}
                        disabled={sampleBusy}
                      >
                        {sampleBusy ? (
                          <LoaderCircle size={15} className="spin" />
                        ) : (
                          <Shuffle size={15} />
                        )}
                        {sample ? 'Другой текст' : 'Получить текст'}
                      </button>
                    </div>
                    {sampleBusy ? (
                      <div className="dataset-empty" role="status">
                        <LoaderCircle size={22} className="spin" />
                        <span>Загружаем случайный текст…</span>
                      </div>
                    ) : sample ? (
                      <>
                        <div className="dataset-meta">
                          <span>
                            <Database size={13} />
                            {sample.dataset.source}
                          </span>
                          <span>
                            Текст {formatNumber(sample.index + 1)} из{' '}
                            {formatNumber(sample.dataset.count)}
                          </span>
                        </div>
                        <label htmlFor="dataset-text" className="sr-only">
                          Случайный текст из датасета
                        </label>
                        <textarea
                          id="dataset-text"
                          className="dataset-text"
                          value={sample.document.text}
                          readOnly
                          lang="uz"
                          dir="auto"
                        />
                        <div className="dataset-hash" title={sample.document.hash}>
                          <span>HASH</span>
                          <code>{sample.document.hash}</code>
                        </div>
                      </>
                    ) : (
                      <div className="dataset-empty">
                        <Database size={22} strokeWidth={1.4} />
                        <span>Получите случайный текст из выбранной выборки.</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <input
                      className="sr-only"
                      ref={fileRef}
                      type="file"
                      accept=".txt,.json,.jsonl"
                      tabIndex={-1}
                      aria-label="Файл с документами"
                      onChange={(event) => {
                        const file = event.target.files?.[0]
                        event.target.value = ''
                        if (file) void uploadFile(file)
                      }}
                    />
                    {batch.length ? (
                      <div className="loaded-file">
                        <div className="file-summary">
                          <span className="file-icon">
                            <FileJson size={21} />
                          </span>
                          <div>
                            <strong title={filename}>{filename}</strong>
                            <span>
                              {batch.length} док. · {formatNumber(totalCharacters)} символов
                            </span>
                          </div>
                          <button
                            type="button"
                            className="icon-button"
                            onClick={() => fileRef.current?.click()}
                            aria-label="Заменить файл"
                          >
                            <RefreshCw size={16} />
                          </button>
                        </div>
                        <label className="document-select">
                          <span>Документ</span>
                          <select
                            aria-label="Исходный документ"
                            value={inputIndex}
                            onChange={(event) => setInputIndex(Number(event.target.value))}
                          >
                            {batch.map((document, index) => (
                              <option key={document.hash} value={index}>
                                {index + 1}. {document.hash}
                              </option>
                            ))}
                          </select>
                          <ChevronDown size={15} />
                        </label>
                        <div className="batch-text" lang="uz" dir="auto">
                          {inputText || <span className="muted">Пустой документ</span>}
                        </div>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className={`dropzone ${dragging ? 'dragging' : ''}`}
                        onClick={() => fileRef.current?.click()}
                        onDragOver={(event) => {
                          event.preventDefault()
                          setDragging(true)
                        }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={handleDrop}
                        disabled={uploadBusy}
                      >
                        <span className="upload-symbol">
                          {uploadBusy ? (
                            <LoaderCircle size={26} className="spin" />
                          ) : (
                            <Upload size={26} strokeWidth={1.5} />
                          )}
                        </span>
                        <strong>{uploadBusy ? 'Читаем документы…' : 'Перетащите файл сюда'}</strong>
                        <span>
                          или <u>выберите на компьютере</u>
                        </span>
                        <small>TXT, JSON, JSONL · до 8 МБ</small>
                      </button>
                    )}
                    <div className="upload-hint">
                      <span>До 64 документов в одном пакете</span>
                      <span>Кодировка UTF-8</span>
                    </div>
                  </>
                )}
              </div>
              <div className="editor-meta">
                <span>Латиница и кириллица</span>
                <span id="character-limit" className={tooLong ? 'limit-exceeded' : ''}>
                  {formatNumber(length)} <span>/ 50 000</span>
                </span>
              </div>
              <div className="input-footer">
                {busy ? (
                  <>
                    <span className="processing-label">
                      <LoaderCircle size={15} className="spin" /> Анализируем · {elapsed} с
                    </span>
                    <button
                      key="cancel"
                      type="button"
                      className="button secondary"
                      onClick={(event) => {
                        event.preventDefault()
                        invalidate()
                        setToast('Анализ отменён')
                      }}
                    >
                      <X size={16} /> Отменить
                    </button>
                  </>
                ) : (
                  <>
                    <span className="shortcut">
                      <span>Ctrl / ⌘ + Enter</span>
                    </span>
                    <button
                      key="submit"
                      type="submit"
                      className="button primary"
                      disabled={uploadBusy || sampleBusy || tooLong || !inputReady}
                    >
                      Анализировать <ArrowRight size={17} />
                    </button>
                  </>
                )}
              </div>
            </form>

            <section
              className={`panel result-panel ${analysis ? 'has-results' : ''}`}
              aria-label="Результаты анализа"
              aria-busy={busy}
            >
              <div className="panel-heading">
                <div className="panel-title">
                  <h2>Результат анализа</h2>
                </div>
                <div className="panel-tools">
                  {analysis && <span className="result-count">{totalEntities}</span>}
                  <button
                    className="icon-button"
                    onClick={() => setModal('expanded')}
                    disabled={!analysis}
                    aria-label="Развернуть результат"
                  >
                    <Maximize2 size={16} />
                  </button>
                </div>
              </div>
              {analysis ? (
                <>
                  <div className="result-toolbar">
                    <div className="view-switch" role="group" aria-label="Представление результата">
                      <button
                        aria-pressed={outputMode === 'text'}
                        className={outputMode === 'text' ? 'current' : ''}
                        onClick={() => setOutputMode('text')}
                      >
                        <ScanText size={15} /> Разметка
                      </button>
                      <button
                        aria-pressed={outputMode === 'json'}
                        className={outputMode === 'json' ? 'current' : ''}
                        onClick={() => setOutputMode('json')}
                      >
                        <Braces size={15} /> JSON
                      </button>
                    </div>
                    <button
                      className="icon-button"
                      onClick={() => {
                        void copy()
                      }}
                      aria-label="Скопировать JSON"
                    >
                      <Clipboard size={16} />
                    </button>
                  </div>
                  {analysis.documents.length > 1 && (
                    <label className="document-select result-document">
                      <span>Документ</span>
                      <select
                        aria-label="Документ результата"
                        value={outputIndex}
                        onChange={(event) => {
                          setOutputIndex(Number(event.target.value))
                          setSelectedEntity(null)
                        }}
                      >
                        {analysis.documents.map((document, index) => (
                          <option key={document.hash} value={index}>
                            {index + 1}. {document.hash}
                          </option>
                        ))}
                      </select>
                      <ChevronDown size={15} />
                    </label>
                  )}
                  <div className="result-content" ref={resultRef}>
                    {outputMode === 'json' ? (
                      <pre className="json-output" tabIndex={0}>
                        {JSON.stringify(analysis.response, null, 2)}
                      </pre>
                    ) : (
                      renderAnnotated()
                    )}
                  </div>
                  <div className="result-footer">
                    <span>
                      <Check size={14} /> {analysis.documents.length} док. ·{' '}
                      {(analysis.duration / 1000).toLocaleString('ru-RU', {
                        maximumFractionDigits: 2,
                      })}{' '}
                      с
                    </span>
                    <button
                      className="text-button"
                      onClick={() =>
                        download(
                          JSON.stringify(analysis.response, null, 2),
                          `ner-results-${new Date().toISOString().slice(0, 10)}.json`,
                        )
                      }
                    >
                      <ArrowDownToLine size={15} /> Экспорт JSON
                    </button>
                  </div>
                </>
              ) : (
                <div className="empty-result" aria-live="polite">
                  {busy ? (
                    <LoaderCircle size={24} className="spin" />
                  ) : (
                    <ScanText size={24} strokeWidth={1.4} aria-hidden="true" />
                  )}
                  <h3>{busy ? 'Анализируем текст…' : 'Результат появится здесь'}</h3>
                  <p>
                    {busy
                      ? `Обрабатываем ${mode === 'batch' ? batch.length : 1} док. · ${elapsed} с`
                      : 'Введите текст и запустите анализ.'}
                  </p>
                </div>
              )}
            </section>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <span className="error-icon">!</span>
              <div>
                <strong>Не удалось выполнить действие</strong>
                <p>{error.message}</p>
                {error instanceof ApiError && error.details.length > 0 && (
                  <ul>
                    {error.details.slice(0, 5).map((detail, index) => (
                      <li key={index}>{detail}</li>
                    ))}
                  </ul>
                )}
              </div>
              <button
                className="icon-button"
                onClick={() => setError(null)}
                aria-label="Закрыть сообщение об ошибке"
              >
                <X size={18} />
              </button>
            </div>
          )}

          {analysis && (
            <section className="panel entities-panel" aria-labelledby="entities-heading">
              <div className="entities-heading">
                <div className="panel-title">
                  <h2 id="entities-heading">Найденные сущности</h2>
                  <span className="neutral-count">{entities.length}</span>
                </div>
              </div>
              <div className="entity-filters" role="group" aria-label="Фильтр по типу сущности">
                {LABELS.map((label) => {
                  const Icon = typeIcons[label]
                  return (
                    <button
                      key={label}
                      aria-pressed={enabledLabels.includes(label)}
                      className={`filter-button ${labelInfo[label].className} ${enabledLabels.includes(label) ? 'enabled' : ''}`}
                      onClick={() => toggleLabel(label)}
                      disabled={!analysis}
                    >
                      <Icon size={14} />
                      <span>{labelInfo[label].title}</span>
                      <span className="filter-count">{counts[label]}</span>
                    </button>
                  )
                })}
                {analysis && (
                  <span className="shown-count">
                    Показано: {visibleEntities.length} из {entities.length}
                  </span>
                )}
              </div>
              {visibleEntities.length ? (
                <div className="entity-list">
                  {visibleEntities.map((entity) => {
                    const Icon = typeIcons[entity.label]
                    return (
                      <button
                        key={entityKey(entity)}
                        className={`entity-card ${labelInfo[entity.label].className} ${selectedEntity === entityKey(entity) ? 'selected' : ''}`}
                        aria-pressed={selectedEntity === entityKey(entity)}
                        onClick={() => {
                          setOutputMode('text')
                          selectEntity(entity)
                        }}
                      >
                        <span className="entity-card-icon">
                          <Icon size={17} />
                        </span>
                        <span className="entity-card-content">
                          <strong lang="uz">
                            {entityText(activeDocument?.text ?? '', entity)}
                          </strong>
                          <span>
                            {labelInfo[entity.label].singular}{' '}
                            <span className="entity-offset">
                              {entity.start}–{entity.end}
                            </span>
                          </span>
                        </span>
                        <ArrowUpRight size={14} />
                      </button>
                    )
                  })}
                </div>
              ) : (
                <div className="no-entities">
                  <ScanText size={19} />
                  <span>
                    {entities.length
                      ? 'Все типы скрыты. Включите фильтр, чтобы увидеть сущности.'
                      : 'Именованные сущности в этом документе не найдены.'}
                  </span>
                </div>
              )}
            </section>
          )}

          <footer className="workspace-footer">
            <span className="footer-status">
              <span className={`status-dot ${health}`} />
              {healthLabels[health]}
              <button
                onClick={() => {
                  setHealth('checking')
                  setHealthTick((tick) => tick + 1)
                }}
                aria-label="Обновить статус модели"
              >
                <RefreshCw size={12} />
              </button>
            </span>
          </footer>
        </main>
      </div>

      <div className="toast-container" role="status" aria-live="polite">
        {toast && (
          <div className="toast">
            <Check size={16} />
            {toast}
            <button onClick={() => setToast('')} aria-label="Закрыть уведомление">
              <X size={14} />
            </button>
          </div>
        )}
      </div>
      {modal === 'help' && (
        <Modal title="От текста к сущностям" close={() => setModal(null)}>
          <p className="modal-intro">
            Сервис выделяет именованные сущности в узбекском тексте, сохраняя исходное написание.
          </p>
          <div className="help-steps">
            <div>
              <span>01</span>
              <section>
                <h3>Добавьте исходный текст</h3>
                <p>
                  Введите свой текст, получите случайную запись из датасета или загрузите файл. В
                  случайном режиме доступны обучающая и валидационная выборки; исходная разметка не
                  используется при анализе. Для нескольких документов загрузите JSON или JSONL с
                  полями <code>hash</code> и <code>text</code>. TXT загружается как один документ.
                </p>
              </section>
            </div>
            <div>
              <span>02</span>
              <section>
                <h3>Запустите анализ</h3>
                <p>
                  Нажмите «Анализировать» или Ctrl / ⌘ + Enter. Обработка выполняется подключённым
                  backend. Если модель недоступна, вы увидите сообщение об ошибке.
                </p>
              </section>
            </div>
            <div>
              <span>03</span>
              <section>
                <h3>Исследуйте результат</h3>
                <p>
                  Выбирайте сущности, включайте фильтры и переключайтесь между документами. Экспорт
                  JSON содержит результат всего пакета в формате API, независимо от фильтров.
                </p>
              </section>
            </div>
          </div>
          <div className="help-limits">
            <strong>Лимиты по умолчанию</strong>
            <span>64 документа · 50 000 символов на текст · 500 000 на пакет</span>
            <p>
              Сервер может использовать другие лимиты. Координаты сущностей — позиции символов
              Unicode: начало включительно, конец не включается.
            </p>
          </div>
          <a className="button secondary" href="/docs" target="_blank" rel="noreferrer">
            Открыть документацию API <ArrowUpRight size={16} />
          </a>
        </Modal>
      )}
      {modal === 'expanded' && analysis && (
        <Modal title={`Разметка · ${activeDocument?.hash ?? ''}`} close={() => setModal(null)}>
          {renderAnnotated()}
        </Modal>
      )}
    </div>
  )
}

export default App

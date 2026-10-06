export interface AppConfig {
  app_title: string
  model: string
  embedding_model: string
  use_multi_query: boolean
  max_iterations: number
  retrieval_top_k: number
  reranker_top_n: number
  history_window: number
}

export interface QueryResult {
  final_answer: string
  rag_context: string
  web_context: string
  critique: string
  iterations: number
  route: string
}

export type StreamStep = "route" | "rag" | "web" | "both" | "synthesis"

export interface StreamEvent {
  type: "start" | "status" | "critique" | "final" | "error" | "token"
  thread_id?: string
  step?: StreamStep
  detail?: string
  passed?: boolean
  result?: QueryResult
  text?: string
}

export interface CollectionStats {
  points: number | null
  sources: Record<string, number>
  content_types: Record<string, number>
  collection: string
  error?: string
}

export interface EvalSampleDetail {
  query: string
  ground_truth: string
  source?: string
  answer: string
  recall_hit: number
  faithfulness?: number | null
  answer_relevancy?: number | null
}

export interface EvalResult {
  metrics: Record<string, number | null>
  sample_count: number
  summary: string
  report_path?: string
  samples?: EvalSampleDetail[]
  error?: string
}

async function jsonFetch<T>(url: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
  // 默认不超时；评估等长任务通过 timeoutMs 显式给长超时
  const controller = timeoutMs ? new AbortController() : undefined
  const timer = controller
    ? setTimeout(() => controller.abort(), timeoutMs)
    : undefined
  try {
    const res = await fetch(url, {
      ...init,
      signal: controller ? controller.signal : init?.signal,
    })
    if (!res.ok) {
      const body = await res.json().catch(() => null)
      throw new Error(body?.detail || `请求失败 (${res.status})`)
    }
    return res.json() as Promise<T>
  } catch (e: any) {
    if (e?.name === "AbortError") {
      throw new Error(`请求超时（${Math.round((timeoutMs || 0) / 1000)}s）`)
    }
    throw e
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export function getConfig(): Promise<AppConfig> {
  return jsonFetch("/api/config")
}

export function getStats(): Promise<CollectionStats> {
  return jsonFetch("/api/stats")
}

export function ingestText(text: string, source: string) {
  return jsonFetch<{ chunks: number; message: string }>("/api/ingest/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  })
}

export function ingestFile(file: File) {
  const form = new FormData()
  form.append("file", file)
  return jsonFetch<{ chunks: number; message: string }>("/api/ingest/file", {
    method: "POST",
    body: form,
  })
}

export function deleteConversation(threadId: string): Promise<{ message: string }> {
  return jsonFetch(`/api/conversation/${threadId}`, { method: "DELETE" })
}

export function runEval(k?: number, threshold?: number): Promise<EvalResult> {
  // 评估耗时长（每样本多次 LLM 调用），给 10 分钟超时避免浏览器/Vite 代理提前切断
  return jsonFetch(
    "/api/evaluate",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k: k ?? null, threshold: threshold ?? null }),
    },
    600000,
  )
}

export async function streamQuery(
  query: string,
  threadId: string | null,
  imagePath: string,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, thread_id: threadId , image_path: imagePath }),
  })
  if (!res.ok || !res.body) {
    throw new Error(`流式请求失败 (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "))
      if (!dataLine) continue
      const payload = dataLine.slice(6).trim()
      if (payload === "[DONE]") return
      try {
        onEvent(JSON.parse(payload) as StreamEvent)
      } catch {
        // 忽略无法解析的帧
      }
    }
  }
}

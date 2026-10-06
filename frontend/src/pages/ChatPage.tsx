import { useEffect, useRef, useState } from "react"
import {
  ArrowUp,
  BookOpen,
  GitMerge,
  Globe,
  Loader2,
  MessageSquare,
  Navigation,
  PenLine,
  Plus,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Timer,
  Trash2,
  Image as ImageIcon,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Markdown } from "@/components/markdown"
import { deleteConversation, streamQuery, type AppConfig, type QueryResult, type StreamStep } from "@/lib/api"

const EXAMPLES = [
  "上传报告中的关键发现是什么？",
  "总结最新的 AI 进展",
  "将我的文档信息与当前新闻对比",
  "知识库中关于收入趋势的数据有哪些？",
]

const ROUTE_META: Record<string, { label: string; cls: string }> = {
  rag_agent: { label: "知识库", cls: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" },
  web_agent: { label: "网络", cls: "border-sky-500/30 bg-sky-500/10 text-sky-400" },
  both: { label: "双源", cls: "border-violet-500/30 bg-violet-500/10 text-violet-400" },
  synthesis: { label: "直接", cls: "border-amber-500/30 bg-amber-500/10 text-amber-400" },
}

const STEP_META: Record<StreamStep, { label: string; icon: typeof Navigation }> = {
  route: { label: "监督者路由", icon: Navigation },
  rag: { label: "知识库检索", icon: BookOpen },
  web: { label: "网络搜索", icon: Globe },
  both: { label: "双路并行", icon: GitMerge },
  synthesis: { label: "综合生成", icon: PenLine },
}

interface StatusStep {
  step: StreamStep
  detail: string
}

interface Turn {
  id: string
  role: "user" | "assistant"
  query?: string
  answer?: string
  result?: QueryResult
  elapsed?: number
  statusSteps: StatusStep[]
  critique?: { passed: boolean; detail: string }
  threadId?: string | null
  error?: string
  streaming: boolean
}

interface Conversation {
  id: string
  threadId: string
  title: string
  turns: Turn[]
  createdAt: number
  updatedAt: number
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem("smartrag_conversations")
    if (!raw) return []
    return JSON.parse(raw)
  } catch {
    return []
  }
}

function saveConversations(convs: Conversation[]) {
  localStorage.setItem("smartrag_conversations", JSON.stringify(convs))
}

function RouteBadge({ route }: { route?: string }) {
  if (!route) return null
  const meta = ROUTE_META[route] ?? ROUTE_META.both
  return (
    <Badge variant="outline" className={cn("border text-[11px] font-medium", meta.cls)}>
      {meta.label}
    </Badge>
  )
}

function StatusTimeline({ steps, running }: { steps: StatusStep[]; running: boolean }) {
  if (steps.length === 0 && running) {
    return (
      <div className="flex items-center gap-2 py-1 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        监督者正在分析查询…
      </div>
    )
  }
  return (
    <div className="mb-3 space-y-1.5 rounded-xl border border-border/50 bg-card/50 p-3">
      {steps.map((s, i) => {
        const meta = STEP_META[s.step]
        const Icon = meta.icon
        const isLast = i === steps.length - 1
        return (
          <div key={`${s.step}-${i}`} className="flex items-center gap-2.5 text-[13px]">
            <span
              className={cn(
                "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border",
                isLast && running
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-muted/60 text-muted-foreground",
              )}
            >
              {isLast && running ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Icon className="h-3.5 w-3.5" />
              )}
            </span>
            <span className={cn("font-medium", isLast && running ? "text-foreground" : "text-muted-foreground")}>
              {meta.label}
            </span>
            {s.detail && <span className="truncate text-xs text-muted-foreground/80">{s.detail}</span>}
          </div>
        )
      })}
    </div>
  )
}

function SourcePanel({ title, icon: Icon, content }: { title: string; icon: typeof BookOpen; content?: string }) {
  if (!content || !content.trim()) return null
  return (
    <Collapsible className="mt-2">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 rounded-lg border border-border/60 bg-card/40 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:text-foreground">
        <Icon className="h-3.5 w-3.5 text-primary/80" />
        {title}
        <span className="ml-auto text-primary/70 transition-transform group-data-[state=open]:rotate-180">▾</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1.5">
        <div className="max-h-64 overflow-auto rounded-lg border border-border/40 bg-black/25 p-3 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
          {content}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function AssistantCard({ turn }: { turn: Turn }) {
  const { result, streaming } = turn
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-cyan-400 shadow-md shadow-primary/25">
        <Sparkles className="h-4 w-4 text-white" />
      </div>
      <Card className="min-w-0 flex-1 border-border/60 bg-card/70 p-4 shadow-lg shadow-black/10 backdrop-blur-sm">
        {streaming && !turn.answer && (
          <>
            <StatusTimeline steps={turn.statusSteps} running />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="mt-2 h-3 w-4/5" />
            <Skeleton className="mt-2 h-3 w-3/5" />
          </>
        )}

        {turn.error && <p className="text-sm text-red-400">⚠️ {turn.error}</p>}

        {turn.answer && (
          <>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <RouteBadge route={result?.route} />
              {result?.iterations ? (
                <Badge variant="outline" className="border-border/60 text-[11px] text-muted-foreground">
                  <RefreshCw className="mr-1 h-3 w-3" />
                  {result.iterations} 轮评审
                </Badge>
              ) : null}
              {turn.elapsed !== undefined && (
                <Badge variant="outline" className="border-border/60 text-[11px] text-muted-foreground">
                  <Timer className="mr-1 h-3 w-3" />
                  {turn.elapsed.toFixed(1)}s
                </Badge>
              )}
              {turn.critique?.passed && (
                <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-[11px] text-emerald-400">
                  <ShieldCheck className="mr-1 h-3 w-3" />
                  评审通过
                </Badge>
              )}
            </div>

            <Markdown content={turn.answer} />

            {turn.critique && !turn.critique.passed && (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-300">
                <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <div className="font-medium">评审要求修订，已携带反馈重试</div>
                  <div className="mt-1 text-amber-200/70">{turn.critique.detail}</div>
                </div>
              </div>
            )}

            <SourcePanel title="知识库检索上下文" icon={BookOpen} content={result?.rag_context} />
            <SourcePanel title="网络搜索上下文" icon={Globe} content={result?.web_context} />
          </>
        )}
      </Card>
    </div>
  )
}

function formatTime(ts: number): string {
  const diff = Date.now() - ts
  if (diff < 60000) return "刚刚"
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  const d = new Date(ts)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`
}

export function ChatPage({ config }: { config: AppConfig | null }) {
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations())
  const [activeId, setActiveId] = useState<string | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)

  const [imagePath, setImagePath] = useState("")              // 上传后的图片路径
  const [imagePreview, setImagePreview] = useState("")          // 上传图片的预览
  const [uploading, setUploading] = useState(false)           // 上传中标记
  const fileInputRef = useRef<HTMLInputElement>(null)     // 隐藏的文件选择框引用

  const threadIdRef = useRef<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // 加载对话列表时，自动选中最近的对话
  useEffect(() => {
    if (conversations.length > 0 && !activeId) {
      const latest = conversations[0]
      setActiveId(latest.id)
      setTurns(latest.turns)
      threadIdRef.current = latest.threadId
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // turns 变化时保存到 conversations
  useEffect(() => {
    if (!activeId) return
    setConversations((prev) => {
      const updated = prev.map((c) =>
        c.id === activeId ? { ...c, turns, updatedAt: Date.now() } : c,
      )
      saveConversations(updated)
      return updated
    })
  }, [turns, activeId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [turns])

  function newConversation() {
    threadIdRef.current = null
    setTurns([])
    setActiveId(null)
  }

  function switchConversation(conv: Conversation) {
    threadIdRef.current = conv.threadId
    setTurns(conv.turns)
    setActiveId(conv.id)
  }

  async function deleteConversationById(id: string, threadId: string) {
    try {
      await deleteConversation(threadId)
    } catch {
      // 后端删除失败不阻塞前端删除
    }
    setConversations((prev) => {
      const updated = prev.filter((c) => c.id !== id)
      saveConversations(updated)
      if (activeId === id) {
        if (updated.length > 0) {
          const latest = updated[0]
          setTurns(latest.turns)
          setActiveId(latest.id)
          threadIdRef.current = latest.threadId
        } else {
          setTurns([])
          setActiveId(null)
          threadIdRef.current = null
        }
      }
      return updated
    })
  }

    async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)

    // 本地预览
    const reader = new FileReader()
    reader.onload = (ev) => setImagePreview(ev.target?.result as string)
    reader.readAsDataURL(file)

    // 上传到后端
    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData })
      const data = await res.json()
      if (data.image_path) {
        setImagePath(data.image_path)
      } else {
        alert("上传失败: " + (data.error || "未知错误"))
        setImagePreview("")
      }
    } catch (err) {
      alert("上传出错: " + String(err))
      setImagePreview("")
    } finally {
      setUploading(false)
      // 清空 input，让同一个文件也能再次上传
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  async function handleSend(text: string) {
    const query = text.trim()
    if (!query && !imagePath || streaming) return
    setInput("")

    const assistantId = crypto.randomUUID()
    const userTurn: Turn = { id: crypto.randomUUID(), role: "user", query, statusSteps: [], streaming: false }
    const assistantTurn: Turn = { id: assistantId, role: "assistant", statusSteps: [], streaming: true }
    setTurns((prev) => [...prev, userTurn, assistantTurn])
    setStreaming(true)

    // 如果没有活跃对话，创建一个
    let convId = activeId
    if (!convId) {
      convId = crypto.randomUUID()
      const newConv: Conversation = {
        id: convId,
        threadId: "",
        title: query.slice(0, 30),
        turns: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
      setConversations((prev) => {
        const updated = [newConv, ...prev]
        saveConversations(updated)
        return updated
      })
      setActiveId(convId)
    }

    const started = performance.now()
    try {
      await streamQuery(query, threadIdRef.current,imagePath, (ev) => {
        if (ev.type === "start" && ev.thread_id) {
          threadIdRef.current = ev.thread_id
          // 更新对话的 threadId
          setConversations((prev) => {
            const updated = prev.map((c) =>
              c.id === convId ? { ...c, threadId: ev.thread_id! } : c,
            )
            saveConversations(updated)
            return updated
          })
        } else if (ev.type === "status" && ev.step && ev.detail) {
          const step: StatusStep = { step: ev.step, detail: ev.detail }
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, statusSteps: [...t.statusSteps, step] } : t)),
          )
        } else if (ev.type === "critique") {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId ? { ...t, critique: { passed: !!ev.passed, detail: ev.detail ?? "" } } : t,
            ),
          )
        } else if (ev.type === "token" && ev.text) {
          // token 级流式：逐字追加答案，用户约 1s 内看到首个字符
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, answer: (t.answer ?? "") + ev.text! } : t)),
          )
        } else if (ev.type === "final" && ev.result) {
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, result: ev.result, answer: ev.result!.final_answer } : t)),
          )
        } else if (ev.type === "error") {
          setTurns((prev) =>
            prev.map((t) => (t.id === assistantId ? { ...t, error: ev.detail ?? "未知错误", streaming: false } : t)),
          )
        }
      })
    } catch (e) {
      setTurns((prev) =>
        prev.map((t) => (t.id === assistantId ? { ...t, error: String(e), streaming: false } : t)),
      )
    } finally {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === assistantId ? { ...t, streaming: false, elapsed: (performance.now() - started) / 1000 } : t,
        ),
      )
      setImagePath("")
      setImagePreview("")
      setStreaming(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* 侧边栏：对话历史 */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-border/50 bg-card/30 backdrop-blur-sm">
        <div className="p-3">
          <Button variant="outline" size="sm" className="w-full justify-start" onClick={newConversation}>
            <Plus className="h-4 w-4" />
            新会话
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-1 px-2 pb-4">
            {conversations.length === 0 && (
              <p className="px-2 py-8 text-center text-xs text-muted-foreground/60">
                还没有对话
              </p>
            )}
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={cn(
                  "group flex cursor-pointer items-start gap-2 rounded-lg px-2.5 py-2 text-left transition-colors",
                  activeId === conv.id
                    ? "bg-primary/10 text-foreground"
                    : "hover:bg-muted/50 text-muted-foreground",
                )}
                onClick={() => switchConversation(conv)}
              >
                <MessageSquare className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", activeId === conv.id ? "text-primary" : "text-muted-foreground/60")} />
                <div className="min-w-0 flex-1">
                  <p className={cn("truncate text-xs font-medium", activeId === conv.id ? "text-foreground" : "text-muted-foreground")}>
                    {conv.title || "新对话"}
                  </p>
                  <p className="text-[10px] text-muted-foreground/50">
                    {formatTime(conv.updatedAt)}
                  </p>
                </div>
                <button
                  className="shrink-0 rounded p-0.5 text-muted-foreground/40 opacity-0 transition-all hover:text-red-400 group-hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation()
                    void deleteConversationById(conv.id, conv.threadId)
                  }}
                  title="删除对话"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </aside>

      {/* 主区域 */}
      <div className="flex h-full flex-1 flex-col">
        {/* 顶部栏 */}
        <header className="flex items-center justify-between border-b border-border/50 px-6 py-4 backdrop-blur-sm">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">智能对话</h1>
            <p className="text-xs text-muted-foreground">
              监督者路由 → 混合检索 → 综合 → 评审修订
            </p>
          </div>
        </header>

        {/* 消息区 */}
        <ScrollArea className="min-h-0 flex-1">
          <div className="mx-auto max-w-3xl space-y-5 px-6 py-6">
            {turns.length === 0 && (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-gradient-to-br from-primary via-indigo-500 to-cyan-400 shadow-xl shadow-primary/30">
                  <Sparkles className="h-7 w-7 text-white" />
                </div>
                <h2 className="text-xl font-semibold">SmartRAG 混合研究助手</h2>
                <p className="mt-2 max-w-md text-sm text-muted-foreground">
                  支持文档知识库 + 实时网络双源检索,答案自动带来源引用,并经评审智能体迭代修订。
                </p>
                <div className="mt-7 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleSend(q)}
                      className="rounded-xl border border-border/60 bg-card/50 px-4 py-3 text-left text-[13px] text-muted-foreground transition-all hover:border-primary/40 hover:bg-card hover:text-foreground"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn) =>
              turn.role === "user" ? (
                <div key={turn.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary/15 px-4 py-3 text-sm leading-relaxed">
                    {turn.query}
                  </div>
                </div>
              ) : (
                <AssistantCard key={turn.id} turn={turn} />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        {/* 输入区 */}
        <div className="border-t border-border/50 px-6 py-4 backdrop-blur-sm">
          {/* 图片预览区 */}
          {imagePreview && (
            <div className="mx-auto mb-2 flex max-w-3xl items-center gap-3 rounded-xl border border-border/60 bg-card/60 p-2">
              <img
                src={imagePreview}
                alt="预览"
                className="h-16 w-16 rounded-lg object-cover"
              />
              <span className="flex-1 truncate text-xs text-muted-foreground">
                {uploading ? "上传中..." : "图片已就绪,发送问题即可识别"}
              </span>
              <button
                onClick={() => {
                  setImagePath("")
                  setImagePreview("")
                }}
                className="rounded p-1 text-muted-foreground transition-colors hover:text-red-400"
                aria-label="移除图片"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-border/60 bg-card/80 p-2 shadow-xl shadow-black/10 backdrop-blur-sm focus-within:border-primary/50">
            {/* 隐藏的文件选择框 */}
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
            />

            {/* 图片上传按钮 */}
            <Button
              size="icon"
              variant="ghost"
              onClick={() => fileInputRef.current?.click()}
              disabled={streaming || uploading}
              className="h-10 w-10 shrink-0 rounded-xl text-muted-foreground hover:text-primary"
              aria-label="上传图片"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImageIcon className="h-4 w-4" />}
            </Button>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  void handleSend(input)
                }
              }}
              rows={1}
              placeholder="输入问题,Enter 发送,Shift+Enter 换行…"
              className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground/60"
            />
            <Button
              size="icon"
              onClick={() => void handleSend(input)}
              disabled={streaming || (!input.trim() && !imagePath)}
              className="h-10 w-10 shrink-0 rounded-xl bg-gradient-to-br from-primary to-cyan-500 shadow-md shadow-primary/30"
              aria-label="发送"
            >
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
            </Button>
          </div>
          <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground/60">
            {config ? `模型 ${config.model} · 检索 Top-${config.retrieval_top_k} → 重排 Top-${config.reranker_top_n}` : ""}
            {config?.use_multi_query ? " · 多查询扩展已开启" : ""}
            {" · 答案由 AI 生成,请核验关键信息"}
          </p>
        </div>
      </div>
    </div>
  )
}

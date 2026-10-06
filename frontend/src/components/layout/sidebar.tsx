import { useTheme } from "next-themes"
import {
  Activity,
  BrainCircuit,
  Database,
  MessageSquareText,
  Moon,
  Sparkles,
  Sun,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import type { AppConfig } from "@/lib/api"
import type { Page } from "@/App"

const NAV: { id: Page; label: string; icon: typeof MessageSquareText }[] = [
  { id: "chat", label: "智能对话", icon: MessageSquareText },
  { id: "data", label: "知识库", icon: Database },
  { id: "eval", label: "质量评估", icon: Activity },
]

export function Sidebar({
  page,
  onNavigate,
  config,
  apiOk,
}: {
  page: Page
  onNavigate: (p: Page) => void
  config: AppConfig | null
  apiOk: boolean | null
}) {
  const { resolvedTheme, setTheme } = useTheme()

  return (
    <aside className="relative z-20 flex w-64 shrink-0 flex-col border-r border-border/60 bg-sidebar/70 backdrop-blur-xl">
      {/* 品牌区 */}
      <div className="flex items-center gap-3 px-5 pb-5 pt-6">
        <div className="relative flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary via-indigo-500 to-cyan-400 shadow-lg shadow-primary/30">
          <BrainCircuit className="h-5 w-5 text-white" />
        </div>
        <div>
          <div className="text-[15px] font-semibold tracking-tight">SmartRAG</div>
          <div className="text-[11px] text-muted-foreground">Multi-Agent Hybrid RAG</div>
        </div>
      </div>

      {/* 导航 */}
      <nav className="flex-1 space-y-1 px-3">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            className={cn(
              "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all",
              page === id
                ? "bg-primary/12 text-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                : "text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            <Icon className={cn("h-4 w-4", page === id ? "text-primary" : "")} />
            {label}
          </button>
        ))}
      </nav>

      {/* 底部：模型信息 + 状态 */}
      <div className="space-y-3 px-4 pb-5">
        <div className="rounded-2xl border border-border/60 bg-card/70 p-3.5">
          <div className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            当前模型
          </div>
          <div className="truncate text-sm font-semibold">{config?.model ?? "加载中…"}</div>
          <div className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span
              className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                apiOk === null ? "bg-muted-foreground/50" : apiOk ? "bg-emerald-400" : "bg-red-400",
              )}
            />
            {apiOk === null ? "连接中…" : apiOk ? "API 已连接" : "API 不可用"}
          </div>
        </div>

        <div className="flex items-center justify-between rounded-xl border border-border/50 bg-card/40 px-3 py-2">
          <span className="text-xs text-muted-foreground">界面主题</span>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            aria-label="切换主题"
          >
            {resolvedTheme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </aside>
  )
}

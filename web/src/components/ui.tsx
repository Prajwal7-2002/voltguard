import clsx from "clsx";
import { AlertTriangle, Info, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { FAULT_LABELS, faultColor } from "@/lib/api";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {children && <p className="mt-1 max-w-3xl text-sm text-muted">{children}</p>}
    </header>
  );
}

export function Card({
  title,
  action,
  children,
  className,
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("rounded-lg border border-border bg-surface p-4", className)}>
      {(title || action) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-medium">{title}</h2>}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "good" | "warn" | "bad";
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-xs text-muted">{label}</div>
      <div
        className={clsx(
          "tabular mt-1 text-2xl font-semibold tracking-tight",
          tone === "good" && "text-good",
          tone === "warn" && "text-warn",
          tone === "bad" && "text-bad",
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function FaultBadge({ code, label }: { code: number | null; label?: string | null }) {
  if (code === null) return <span className="text-xs text-muted">n/a</span>;
  return (
    <span className="inline-flex items-center gap-1.5 text-sm whitespace-nowrap">
      <span className="size-2 shrink-0 rounded-full" style={{ background: faultColor(code) }} />
      {label ?? FAULT_LABELS[code]}
    </span>
  );
}

const VERDICT_STYLE: Record<string, string> = {
  hit: "bg-good-soft text-good",
  "false alarm": "bg-warn-soft text-warn",
  missed: "bg-bad-soft text-bad",
  ok: "bg-surface-2 text-muted",
  unknown: "bg-surface-2 text-muted",
};

export function VerdictBadge({ verdict }: { verdict: string }) {
  return (
    <span className={clsx("rounded px-1.5 py-0.5 text-xs font-medium whitespace-nowrap", VERDICT_STYLE[verdict])}>
      {verdict}
    </span>
  );
}

export function Note({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn";
  title?: string;
  children: ReactNode;
}) {
  const Icon = tone === "warn" ? AlertTriangle : Info;
  return (
    <div
      className={clsx(
        "flex gap-3 rounded-lg border p-3 text-sm",
        tone === "warn" ? "border-warn/30 bg-warn-soft" : "border-accent/20 bg-accent-soft",
      )}
    >
      <Icon className={clsx("mt-0.5 size-4 shrink-0", tone === "warn" ? "text-warn" : "text-accent")} aria-hidden />
      <div>
        {title && <div className="font-medium">{title}</div>}
        <div className="text-muted">{children}</div>
      </div>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted">
      <Loader2 className="size-4 animate-spin" aria-hidden /> {label}
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <Note tone="warn" title="Couldn't load data">
      {message}
    </Note>
  );
}

export function Button({
  children,
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" }) {
  return (
    <button
      {...props}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary"
          ? "bg-accent text-white hover:opacity-90"
          : "border border-border bg-surface hover:bg-surface-2",
        className,
      )}
    >
      {children}
    </button>
  );
}

export const pct = (x: number | undefined | null, digits = 1) =>
  x === undefined || x === null ? "—" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | undefined | null, digits = 2) =>
  x === undefined || x === null ? "—" : x.toFixed(digits);

"use client";

import { type ReactElement, cloneElement, useEffect, useRef, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FAULT_LABELS, faultColor } from "@/lib/api";

/** Measures its box and renders the chart at that exact size (robust inside grids/flex). */
function Sized({ className, children }: { className: string; children: ReactElement<{ width?: number; height?: number }> }) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className={`w-full min-w-0 ${className}`}>
      {size && cloneElement(children, size)}
    </div>
  );
}

/** Tick labels with just enough decimals for the visible range. */
const smartTick = (v: number) =>
  Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en") : Number(v.toFixed(1)).toString();

const axis = { stroke: "var(--muted)", fontSize: 11, tickLine: false };
const grid = { stroke: "var(--border)", strokeDasharray: "3 3" };
const tooltipStyle = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    fontSize: 12,
    color: "var(--text)",
  },
  labelStyle: { color: "var(--muted)" },
};

/** Horizontal bars for 0..1 values (class probabilities). */
export function ProbabilityBars({ items }: { items: { code: number; value: number }[] }) {
  return (
    <ul className="space-y-2">
      {items.map(({ code, value }) => (
        <li key={code} className="text-sm">
          <div className="mb-1 flex justify-between gap-2">
            <span className="truncate">{FAULT_LABELS[code]}</span>
            <span className="tabular text-muted">{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded bg-surface-2">
            <div
              className="h-2 rounded"
              style={{ width: `${Math.max(value * 100, 0.5)}%`, background: faultColor(code) }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Diverging bars for SHAP contributions: toward the predicted class (its colour) vs. away (grey). */
export function ShapBars({ items, predicted }: { items: { feature: string; shap: number }[]; predicted: number }) {
  const max = Math.max(...items.map((d) => Math.abs(d.shap)), 1e-9);
  return (
    <ul className="space-y-1.5">
      {items.map(({ feature, shap }) => {
        const w = (Math.abs(shap) / max) * 50;
        return (
          <li key={feature} className="grid grid-cols-[7.5rem_1fr_3.5rem] items-center gap-2 text-sm">
            <span className="truncate font-mono text-xs">{feature}</span>
            <div className="relative h-3 rounded bg-surface-2">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <div
                className="absolute inset-y-0 rounded"
                style={{
                  width: `${w}%`,
                  left: shap >= 0 ? "50%" : `${50 - w}%`,
                  background: shap < 0 ? "var(--muted)" : predicted === 0 ? "var(--good)" : faultColor(predicted),
                  opacity: shap >= 0 ? 1 : 0.45,
                }}
              />
            </div>
            <span className="tabular text-right text-xs text-muted">
              {shap >= 0 ? "+" : ""}
              {shap.toFixed(2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export type TimelinePoint = {
  tick: number;
  score: number;
  truth: number | null;
  derated: number;
  signal?: number;
};

/** Fault score over time, with proxy-truth fault periods and derate periods shaded. */
export function FaultTimeline({ data }: { data: TimelinePoint[] }) {
  return (
    <Sized className="h-64">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid {...grid} vertical={false} />
          <XAxis dataKey="tick" {...axis} />
          <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}`} />
          <Tooltip
            {...tooltipStyle}
            formatter={(v, name) =>
              name === "Fault score" ? [`${(Number(v) * 100).toFixed(0)}%`, name] : [Number(v) ? "yes" : "no", name]
            }
            labelFormatter={(t) => `Tick ${t}`}
          />
          <Area
            type="stepAfter"
            dataKey="truth"
            name="Actual thermal stress"
            fill="var(--bad)"
            fillOpacity={0.12}
            stroke="none"
            isAnimationActive={false}
          />
          <Area
            type="stepAfter"
            dataKey="derated"
            name="Derate active"
            fill="var(--accent)"
            fillOpacity={0.1}
            stroke="none"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="score"
            name="Fault score"
            stroke="var(--c1)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
    </Sized>
  );
}

export function SignalChart({ data, label }: { data: TimelinePoint[]; label: string }) {
  return (
    <Sized className="h-40">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid {...grid} vertical={false} />
          <XAxis dataKey="tick" {...axis} />
          <YAxis
            {...axis}
            domain={([lo, hi]: readonly number[]) => {
              const pad = Math.max(1, (hi - lo) * 0.1);
              return [Math.floor(lo - pad), Math.ceil(hi + pad)];
            }}
            tickFormatter={smartTick}
          />
          <Tooltip {...tooltipStyle} formatter={(v) => [Number(v).toFixed(1), label]} labelFormatter={(t) => `Tick ${t}`} />
          <Line type="monotone" dataKey="signal" stroke="var(--accent)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </ComposedChart>
    </Sized>
  );
}

/** Class counts on a log axis, so rare fault classes stay visible next to Nominal. */
export function ClassDistributionChart({ counts }: { counts: Record<string, number> }) {
  const data = Object.entries(counts).map(([code, rows]) => ({
    code: Number(code),
    name: FAULT_LABELS[Number(code)],
    rows,
  }));
  return (
    <Sized className="h-64">
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid {...grid} horizontal={false} />
          <XAxis
            type="number"
            scale="log"
            domain={[100, "auto"]}
            allowDataOverflow
            {...axis}
            tickFormatter={(v) => Intl.NumberFormat("en", { notation: "compact" }).format(v)}
          />
          <YAxis type="category" dataKey="name" width={150} {...axis} />
          <Tooltip
            {...tooltipStyle}
            cursor={{ fill: "var(--surface-2)" }}
            formatter={(v) => [Number(v).toLocaleString("en"), "rows"]}
          />
          <Bar dataKey="rows" radius={[0, 3, 3, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.code} fill={faultColor(d.code)} />
            ))}
          </Bar>
        </BarChart>
    </Sized>
  );
}

export function ProfileFaultChart({ profiles }: { profiles: { profile_id: number; fault_rate: number }[] }) {
  const data = [...profiles].sort((a, b) => b.fault_rate - a.fault_rate);
  return (
    <Sized className="h-56">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid {...grid} vertical={false} />
          <XAxis dataKey="profile_id" {...axis} interval={0} tick={false} label={{ value: "drives, sorted", fill: "var(--muted)", fontSize: 11 }} />
          <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip
            {...tooltipStyle}
            cursor={{ fill: "var(--surface-2)" }}
            formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "rows under thermal stress"]}
            labelFormatter={(p) => `Drive ${p}`}
          />
          <Bar dataKey="fault_rate" fill="var(--c1)" radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
    </Sized>
  );
}

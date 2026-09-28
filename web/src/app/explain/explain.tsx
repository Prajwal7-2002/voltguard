"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ProbabilityBars, ShapBars } from "@/components/charts";
import { Button, Card, ErrorState, FaultBadge, Loading, Note, PageHeader, Stat, pct } from "@/components/ui";
import { type FeatureRange, type Scenario, FEATURE_UNITS, explainScenario, getDataSummary } from "@/lib/api";

const INPUTS: { key: keyof Scenario; label: string }[] = [
  { key: "ambient", label: "Ambient temperature" },
  { key: "coolant", label: "Coolant temperature" },
  { key: "motor_speed", label: "Motor speed" },
  { key: "torque", label: "Torque" },
  { key: "i_d", label: "d-axis current" },
  { key: "i_q", label: "q-axis current" },
  { key: "u_d", label: "d-axis voltage" },
  { key: "u_q", label: "q-axis voltage" },
];

type Ranges = Record<string, FeatureRange>;

const PRESETS: { name: string; build: (r: Ranges) => Scenario }[] = [
  { name: "Typical", build: (r) => pick(r, () => "median") },
  {
    name: "Hard acceleration, hot coolant",
    build: (r) => ({ ...pick(r, () => "median"), coolant: r.coolant.max, torque: r.torque.max, i_q: r.i_q.max, motor_speed: r.motor_speed.max * 0.8 }),
  },
  {
    name: "Standstill, hot coolant",
    build: (r) => ({ ...pick(r, () => "median"), coolant: r.coolant.max, motor_speed: Math.max(r.motor_speed.min, 0), torque: 0, i_q: 0 }),
  },
];

function pick(r: Ranges, which: () => keyof FeatureRange): Scenario {
  return Object.fromEntries(INPUTS.map(({ key }) => [key, r[key][which()]])) as Scenario;
}

function useDebounced<T>(value: T, ms: number) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export function Explain() {
  const summary = useQuery({ queryKey: ["data-summary"], queryFn: getDataSummary });
  if (summary.isPending) return <Loading />;
  if (summary.error) return <ErrorState error={summary.error} />;
  return <Workbench ranges={summary.data.feature_ranges} realRanges={summary.data.dataset_available} />;
}

function Workbench({ ranges, realRanges }: { ranges: Ranges; realRanges: boolean }) {
  const [scenario, setScenario] = useState<Scenario>(() => PRESETS[0].build(ranges));
  const debounced = useDebounced(scenario, 250);
  const result = useQuery({
    queryKey: ["explain", debounced],
    queryFn: () => explainScenario(debounced),
    placeholderData: keepPreviousData,
  });

  const r = result.data;
  const drivers = r
    ? Object.entries(r.explanations)
        .map(([feature, shap]) => ({ feature, shap }))
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .slice(0, 10)
    : [];

  return (
    <>
      <PageHeader title="Explainability">
        Build a what-if operating point and see what the model predicts and which inputs drive
        it. Each scenario is scored on its own: nothing is remembered between runs.
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
        <Card
          title="Scenario"
          action={<span className="text-xs text-muted">{realRanges ? "ranges: 2nd–98th percentile of real drives" : "typical ranges"}</span>}
        >
          <div className="mb-4 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <Button key={p.name} variant="secondary" className="text-xs" onClick={() => setScenario(p.build(ranges))}>
                {p.name}
              </Button>
            ))}
          </div>
          <div className="space-y-3">
            {INPUTS.map(({ key, label }) => {
              const { min, max } = ranges[key];
              const id = `in-${key}`;
              return (
                <div key={key}>
                  <div className="mb-1 flex justify-between text-sm">
                    <label htmlFor={id}>{label}</label>
                    <span className="tabular text-muted">
                      {scenario[key].toFixed(1)} {FEATURE_UNITS[key]}
                    </span>
                  </div>
                  <input
                    id={id}
                    type="range"
                    className="w-full accent-[var(--accent)]"
                    min={min}
                    max={max}
                    step={(max - min) / 200 || 1}
                    value={Math.min(Math.max(scenario[key], min), max)}
                    onChange={(e) => setScenario((s) => ({ ...s, [key]: Number(e.target.value) }))}
                  />
                </div>
              );
            })}
          </div>
        </Card>

        <div className="space-y-4">
          {result.error ? (
            <ErrorState error={result.error} />
          ) : !r ? (
            <Loading label="Scoring…" />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Prediction" value={<FaultBadge code={r.predicted_fault} />} />
                <Stat label="Fault score" value={pct(1 - (r.probabilities["0"] ?? 0), 0)} hint="1 − P(Nominal)" />
              </div>
              <Card title="Class probabilities">
                <ProbabilityBars
                  items={Object.entries(r.probabilities)
                    .map(([code, value]) => ({ code: Number(code), value }))
                    .sort((a, b) => b.value - a.value)}
                />
              </Card>
              <Card title="Top drivers (SHAP)">
                <ShapBars items={drivers} predicted={r.predicted_fault} />
                <p className="mt-3 text-xs text-muted">
                  Coloured bars push toward <strong>{r.fault_label}</strong>; grey bars push away
                  from it. Derived
                  inputs (<code>thermal_delta</code>, <code>power_draw</code>, <code>battery_temp</code>)
                  are computed from the sliders. Rate-of-change inputs are zero because a single
                  scenario has no history.
                </p>
              </Card>
              {r.action && (
                <Card title="Recommended action">
                  <p className="text-sm">{r.root_cause}</p>
                  <p className="mt-1 text-sm text-muted">{r.action}</p>
                </Card>
              )}
            </>
          )}
          <Note>
            Treat these as model behaviour, not physics. The model only sees these eight inputs,
            and real drives rarely combine extreme values the way sliders can.
          </Note>
        </div>
      </div>
    </>
  );
}

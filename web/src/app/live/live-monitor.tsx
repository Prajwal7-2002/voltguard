"use client";

import { useMemo, useState } from "react";
import { FaultTimeline, ProbabilityBars, ShapBars, SignalChart } from "@/components/charts";
import { SimControls } from "@/components/sim-controls";
import { Card, ErrorState, FaultBadge, Loading, Note, PageHeader, Stat, VerdictBadge, pct } from "@/components/ui";
import { FAULT_LABELS, FEATURE_UNITS, tallyAlerts, toEpisodes, verdict } from "@/lib/api";
import { useSimulator } from "@/lib/use-simulator";

const SIGNALS: { key: string; label: string; note?: string }[] = [
  { key: "coolant", label: "Coolant temperature" },
  { key: "motor_speed", label: "Motor speed" },
  { key: "torque", label: "Torque" },
  { key: "i_q", label: "q-axis current" },
  { key: "stator_winding", label: "Stator winding temperature", note: "Hidden from the model; used to define the labels." },
  { key: "battery_temp", label: "Battery temperature", note: "Simulated from electrical power, not measured." },
  { key: "soc", label: "State of charge", note: "Simulated, not measured." },
];

export function LiveMonitor() {
  const sim = useSimulator(1);
  const [signal, setSignal] = useState(SIGNALS[0]);

  const frames = sim.frames;
  const latest = frames.at(-1);
  const tally = useMemo(() => tallyAlerts(frames), [frames]);
  const timeline = useMemo(
    () =>
      frames.slice(-300).map((f) => ({
        tick: f.tick,
        score: f.fault_score,
        truth: f.true_fault === null ? null : f.true_fault !== 0 ? 1 : 0,
        derated: f.derated ? 1 : 0,
        signal: f.reading[signal.key],
      })),
    [frames, signal.key],
  );
  const episodes = useMemo(() => toEpisodes(frames).slice(0, 25), [frames]);

  const alerts = tally ? tally.hits + tally.falseAlarms : 0;
  const stressTicks = tally ? tally.hits + tally.missed : 0;

  return (
    <>
      <PageHeader title="Live monitor">
        Replays a recorded drive tick by tick through the real model. Each alert is checked
        against what the hidden temperatures actually did, so you can see hits, false alarms and
        misses as they happen.
      </PageHeader>

      <Card className="mb-4">
        <SimControls sim={sim} />
      </Card>

      {sim.error && <div className="mb-4"><ErrorState error={sim.error} /></div>}
      {sim.session && !sim.session.dataset_available && (
        <div className="mb-4">
          <Note tone="warn" title="No replay dataset on the API host">
            The simulator is replaying a fixed placeholder reading and there is no ground truth.
            Add <code>data/comprehensive_fault_training_data.csv</code> to replay real drives.
          </Note>
        </div>
      )}

      {!sim.session && !sim.error ? (
        <Loading label="Starting simulator (the first start loads the dataset)…" />
      ) : !latest ? (
        <p className="py-10 text-center text-sm text-muted">Press Play or step forward to start the replay.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat
              label={`Model says · tick ${latest.tick}`}
              value={<FaultBadge code={latest.predicted_fault} />}
              hint={`fault score ${pct(latest.fault_score, 0)}${latest.derated ? " · derate active" : ""}`}
            />
            <Stat
              label="Actually happening"
              value={<FaultBadge code={latest.true_fault} label={latest.true_label} />}
              hint={<VerdictBadge verdict={verdict(latest)} />}
            />
            <Stat
              label="Alert precision (this run)"
              value={alerts ? pct(tally!.hits / alerts, 0) : "—"}
              hint={tally ? `${tally.hits} hits · ${tally.falseAlarms} false alarms` : "needs ground truth"}
              tone={alerts ? (tally!.hits / alerts >= 0.5 ? "good" : "warn") : undefined}
            />
            <Stat
              label="Stress caught (this run)"
              value={stressTicks ? pct(tally!.hits / stressTicks, 0) : "—"}
              hint={tally ? `${tally.missed} stressed ticks missed` : "needs ground truth"}
            />
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-[1.6fr_1fr]">
            <div className="min-w-0 space-y-4">
              <Card
                title="Fault score over time"
                action={
                  <div className="flex flex-wrap gap-3 text-xs text-muted">
                    <Legend color="var(--c1)" line label="fault score" />
                    <Legend color="var(--bad)" label="actual thermal stress" />
                    <Legend color="var(--accent)" label="derate active" />
                  </div>
                }
              >
                <FaultTimeline data={timeline} />
                <p className="mt-2 text-xs text-muted">
                  Fault score = 1 − P(Nominal). Training up-weights rare classes, so the score runs
                  high. Read it as a ranking, not a calibrated probability.
                </p>
              </Card>
              <Card
                title={`${signal.label}${FEATURE_UNITS[signal.key] ? ` (${FEATURE_UNITS[signal.key]})` : ""}`}
                action={
                  <select
                    aria-label="Signal"
                    className="rounded-md border border-border bg-surface px-2 py-1 text-sm"
                    value={signal.key}
                    onChange={(e) => setSignal(SIGNALS.find((s) => s.key === e.target.value)!)}
                  >
                    {SIGNALS.map((s) => (
                      <option key={s.key} value={s.key}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                }
              >
                <SignalChart data={timeline} label={signal.label} />
                {signal.note && <p className="mt-2 text-xs text-muted">{signal.note}</p>}
              </Card>
            </div>

            <div className="min-w-0 space-y-4">
              <Card title="Class probabilities">
                <ProbabilityBars
                  items={FAULT_LABELS.map((label, code) => ({ code, value: latest.probabilities[label] ?? 0 }))
                    .sort((a, b) => b.value - a.value)}
                />
              </Card>
              <Card title="Why: top drivers (SHAP)">
                <ShapBars items={latest.top_drivers} predicted={latest.predicted_fault} />
                <p className="mt-3 text-xs text-muted">
                  Coloured bars push toward <strong>{latest.predicted_label}</strong>; grey bars push away from it.
                </p>
              </Card>
              {latest.action && (
                <Card title="Recommended action">
                  <p className="text-sm">{latest.root_cause}</p>
                  <p className="mt-1 text-sm text-muted">{latest.action}</p>
                  {latest.derate_applied && (
                    <p className="mt-2 text-xs text-accent">Derate applied to the replay for the next ticks.</p>
                  )}
                </Card>
              )}
            </div>
          </div>

          <Card title="Alert and miss episodes" className="mt-4">
            {episodes.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr className="border-b border-border">
                      <th className="py-2 pr-4 font-medium">Ticks</th>
                      <th className="py-2 pr-4 font-medium">Verdict</th>
                      <th className="py-2 pr-4 font-medium">Model said</th>
                      <th className="py-2 pr-4 font-medium">Actual</th>
                      <th className="py-2 pr-4 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {episodes.map((e) => (
                      <tr key={e.start} className="border-b border-border/60 last:border-0">
                        <td className="tabular py-2 pr-4 whitespace-nowrap">
                          {e.start === e.end ? e.start : `${e.start}–${e.end}`}
                          <span className="ml-1 text-xs text-muted">({e.end - e.start + 1})</span>
                        </td>
                        <td className="py-2 pr-4"><VerdictBadge verdict={e.verdict} /></td>
                        <td className="py-2 pr-4"><FaultBadge code={e.predicted} /></td>
                        <td className="py-2 pr-4"><FaultBadge code={e.truth} /></td>
                        <td className="py-2 pr-4 text-muted">
                          {e.action ?? "—"}
                          {e.derated && <span className="ml-2 rounded bg-accent-soft px-1 py-0.5 text-xs text-accent">derated</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted">No alerts or missed stress so far.</p>
            )}
          </Card>
        </>
      )}
    </>
  );
}

function Legend({ color, label, line }: { color: string; label: string; line?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={line ? "h-0.5 w-3" : "size-2.5 rounded-sm opacity-40"} style={{ background: color }} />
      {label}
    </span>
  );
}

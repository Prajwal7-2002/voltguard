"use client";

import { useMemo } from "react";
import { SimControls } from "@/components/sim-controls";
import { Card, ErrorState, FaultBadge, Loading, Note, PageHeader, Stat, VerdictBadge, pct } from "@/components/ui";
import { type Frame, FAULT_LABELS, faultColor, tallyAlerts, verdict } from "@/lib/api";
import { useSimulator } from "@/lib/use-simulator";

const FLEET_SIZE = 8;
const HEATMAP_TICKS = 30;

export function Fleet() {
  const sim = useSimulator(FLEET_SIZE);

  const byVehicle = useMemo(() => {
    const map = new Map<string, Frame[]>();
    for (const f of sim.frames) {
      if (!map.has(f.vehicle_id)) map.set(f.vehicle_id, []);
      map.get(f.vehicle_id)!.push(f);
    }
    return map;
  }, [sim.frames]);

  const latestTick = sim.frames.at(-1)?.tick ?? 0;
  const tally = useMemo(() => tallyAlerts(sim.frames), [sim.frames]);
  const rows = useMemo(
    () =>
      (sim.session?.vehicles ?? []).map((id) => {
        const fs = byVehicle.get(id) ?? [];
        const t = tallyAlerts(fs);
        return { id, latest: fs.at(-1), alerts: fs.filter((f) => f.predicted_fault !== 0).length, tally: t };
      }),
    [byVehicle, sim.session],
  );
  const alerting = rows.filter((r) => r.latest && r.latest.predicted_fault !== 0).length;
  const stressed = rows.filter((r) => r.latest?.true_fault).length;
  const alerts = tally ? tally.hits + tally.falseAlarms : 0;

  return (
    <>
      <PageHeader title="Fleet">
        {FLEET_SIZE} vehicles, each replaying a different recorded drive. Use it to see which units
        the model flags and how many of those flags turn out to be real.
      </PageHeader>

      <Card className="mb-4">
        <SimControls sim={sim} stepLabel="rounds" />
      </Card>

      {sim.error && <div className="mb-4"><ErrorState error={sim.error} /></div>}
      {sim.session && !sim.session.dataset_available && (
        <div className="mb-4">
          <Note tone="warn" title="No replay dataset on the API host">
            All vehicles replay the same placeholder reading and there is no ground truth.
          </Note>
        </div>
      )}

      {!sim.session && !sim.error ? (
        <Loading label="Starting fleet…" />
      ) : !latestTick ? (
        <p className="py-10 text-center text-sm text-muted">Press Play or step forward to start the fleet.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Round" value={latestTick} />
            <Stat label="Vehicles alerting now" value={`${alerting} / ${FLEET_SIZE}`} tone={alerting ? "warn" : "good"} />
            <Stat label="Actually stressed now" value={tally ? `${stressed} / ${FLEET_SIZE}` : "—"} />
            <Stat
              label="Fleet alert precision"
              value={alerts ? pct(tally!.hits / alerts, 0) : "—"}
              hint={tally ? `${tally.hits} hits · ${tally.falseAlarms} false alarms · ${tally.missed} missed` : "needs ground truth"}
            />
          </div>

          <Card title={`Last ${HEATMAP_TICKS} rounds`} className="mt-4">
            <div className="overflow-x-auto">
              <div className="min-w-[36rem] space-y-1">
                {rows.map(({ id }) => {
                  const fs = (byVehicle.get(id) ?? []).slice(-HEATMAP_TICKS);
                  return (
                    <div key={id} className="grid grid-cols-[3.5rem_1fr] items-center gap-2">
                      <span className="font-mono text-xs text-muted">{id}</span>
                      <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${HEATMAP_TICKS}, minmax(0, 1fr))` }}>
                        {Array.from({ length: HEATMAP_TICKS - fs.length }, (_, i) => (
                          <div key={`pad-${i}`} className="h-6 rounded-sm bg-surface-2" />
                        ))}
                        {fs.map((f) => (
                          <div
                            key={f.tick}
                            title={`Tick ${f.tick}: model ${f.predicted_label}${f.true_label ? `, actual ${f.true_label}` : ""}`}
                            className="relative h-6 rounded-sm bg-surface-2"
                          >
                            {f.predicted_fault !== 0 && (
                              <div className="absolute inset-0 rounded-sm" style={{ background: faultColor(f.predicted_fault), opacity: 0.35 + 0.65 * f.fault_score }} />
                            )}
                            {!!f.true_fault && <div className="absolute inset-x-0 bottom-0 h-1 rounded-b-sm bg-text" />}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
              {FAULT_LABELS.slice(1).map((label, i) => (
                <span key={label} className="flex items-center gap-1.5">
                  <span className="size-2.5 rounded-sm" style={{ background: faultColor(i + 1) }} /> alert: {label}
                </span>
              ))}
              <span className="flex items-center gap-1.5">
                <span className="h-1 w-3 rounded-sm bg-text" /> actual thermal stress
              </span>
            </div>
          </Card>

          <Card title="Vehicles" className="mt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted">
                  <tr className="border-b border-border">
                    <th className="py-2 pr-4 font-medium">Vehicle</th>
                    <th className="py-2 pr-4 font-medium">Model says</th>
                    <th className="py-2 pr-4 font-medium">Actual</th>
                    <th className="py-2 pr-4 font-medium">Now</th>
                    <th className="py-2 pr-4 text-right font-medium">Fault score</th>
                    <th className="py-2 pr-4 text-right font-medium">Alerts</th>
                    <th className="py-2 pr-4 text-right font-medium">False alarms</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows]
                    .sort((a, b) => (b.latest?.fault_score ?? 0) - (a.latest?.fault_score ?? 0))
                    .map(({ id, latest, alerts, tally: t }) => (
                      <tr key={id} className="border-b border-border/60 last:border-0">
                        <td className="py-2 pr-4 font-mono text-xs">
                          {id}
                          {latest?.derated && <span className="ml-2 rounded bg-accent-soft px-1 py-0.5 font-sans text-accent">derated</span>}
                        </td>
                        <td className="py-2 pr-4">{latest && <FaultBadge code={latest.predicted_fault} />}</td>
                        <td className="py-2 pr-4">{latest && <FaultBadge code={latest.true_fault} />}</td>
                        <td className="py-2 pr-4">{latest && <VerdictBadge verdict={verdict(latest)} />}</td>
                        <td className="tabular py-2 pr-4 text-right">{pct(latest?.fault_score, 0)}</td>
                        <td className="tabular py-2 pr-4 text-right">{alerts}</td>
                        <td className="tabular py-2 pr-4 text-right">{t ? t.falseAlarms : "—"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}

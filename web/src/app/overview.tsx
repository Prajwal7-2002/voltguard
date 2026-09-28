"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ClassDistributionChart, ProfileFaultChart } from "@/components/charts";
import { Card, ErrorState, Loading, Note, PageHeader, Stat, num, pct } from "@/components/ui";
import { FAULT_LABELS, getDataSummary, getLatestModel } from "@/lib/api";

export function Overview() {
  const model = useQuery({ queryKey: ["model"], queryFn: getLatestModel });
  const data = useQuery({ queryKey: ["data-summary"], queryFn: getDataSummary });

  return (
    <>
      <PageHeader title="Overview">
        VoltGuard watches an EV traction motor&apos;s electrical and control signals (voltages,
        currents, speed, torque, coolant and ambient temperature) and warns when the motor is
        heading into thermal stress, before the temperatures it can&apos;t see directly get there.
      </PageHeader>

      {model.isPending ? (
        <Loading />
      ) : model.error ? (
        <ErrorState error={model.error} />
      ) : (
        <Headline info={model.data} />
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card title="How often each condition occurs">
          {data.data ? (
            <>
              <ClassDistributionChart counts={data.data.class_distribution} />
              <p className="mt-2 text-xs text-muted">
                Log scale: about 91% of rows are Nominal. Below: how many of the{" "}
                {data.data.profiles_per_class["0"] ?? "recorded"} drives contain each class.
              </p>
              <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted sm:grid-cols-3">
                {Object.entries(data.data.profiles_per_class).map(([code, drives]) => (
                  <li key={code}>
                    {FAULT_LABELS[Number(code)]}: <span className="tabular text-text">{drives}</span> drives
                  </li>
                ))}
              </ul>
            </>
          ) : data.error ? (
            <ErrorState error={data.error} />
          ) : (
            <Loading />
          )}
        </Card>

        <Card title="Thermal stress by drive">
          {data.data?.dataset_available ? (
            <>
              <ProfileFaultChart profiles={data.data.profiles} />
              <p className="mt-2 text-xs text-muted">
                Share of each drive&apos;s rows labelled as thermal stress. Stress is concentrated in
                a minority of drives, which is why results vary between test splits.
              </p>
            </>
          ) : data.data ? (
            <Note title="Replay dataset not loaded">
              Place the PMSM dataset at <code>data/comprehensive_fault_training_data.csv</code> on
              the API host to see per-drive statistics and run the simulator with ground truth.
            </Note>
          ) : (
            <Loading />
          )}
        </Card>
      </div>

      <div className="mt-6">
        <Note title="What the labels mean">
          The dataset has no recorded failures. A &ldquo;fault&rdquo; here means a hidden motor
          temperature (stator winding, tooth, yoke or magnet) is in its top few percent for the
          dataset, so each class is a <em>thermal-stress warning</em>, not a diagnosed component
          failure. See <Link href="/model" className="text-accent underline">Model</Link> for
          how this is evaluated.
        </Note>
      </div>
    </>
  );
}

function Headline({ info }: { info: Awaited<ReturnType<typeof getLatestModel>> }) {
  const m = info.metrics;
  const base = m.baseline_always_nominal;
  const heldOut = m.benchmark_split === "profile_group_holdout";
  return (
    <>
      {!heldOut && (
        <div className="mb-4">
          <Note tone="warn" title="Optimistic metrics">
            Model {info.version} was not evaluated on held-out drives, so the scores below are
            inflated. Retrain with <code>python scripts/train.py</code>.
          </Note>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Macro F1 · unseen drives"
          value={num(m.test_f1_macro)}
          hint={base ? `always-Nominal baseline: ${num(base.test_f1_macro)}` : undefined}
          tone={base && m.test_f1_macro > base.test_f1_macro ? "good" : undefined}
        />
        <Stat
          label="Macro F1 · cross-validated"
          value={num(m.cv_f1_macro_mean)}
          hint={m.cv_f1_macro_std !== undefined ? `± ${num(m.cv_f1_macro_std)} across drive folds` : undefined}
        />
        <Stat
          label="Stress recall (macro)"
          value={pct(m.test_recall_macro, 0)}
          hint={`precision ${pct(m.test_precision_macro, 0)}: many false alarms`}
          tone="warn"
        />
        <Stat
          label="Accuracy"
          value={pct(m.test_accuracy)}
          hint={base ? `baseline ${pct(base.test_accuracy)}: accuracy isn't the goal` : undefined}
        />
      </div>
      <p className="mt-3 text-xs text-muted">
        Model <span className="font-mono">{info.version}</span>
        {heldOut && m.test_groups ? ` · tested on ${m.test_groups} drives it never saw during training` : ""}
        {m.test_rows ? ` · ${m.test_rows.toLocaleString("en")} test rows` : ""}
      </p>
    </>
  );
}

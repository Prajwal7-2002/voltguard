"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, ErrorState, FaultBadge, Loading, Note, PageHeader, num, pct } from "@/components/ui";
import { FAULT_LABELS, type ModelMetrics, getLatestModel } from "@/lib/api";

export function ModelReport() {
  const model = useQuery({ queryKey: ["model"], queryFn: getLatestModel });

  return (
    <>
      <PageHeader title="Model">
        How the served model was evaluated and where it falls short. All numbers are from drives
        held out of training, unless flagged.
      </PageHeader>
      {model.isPending ? (
        <Loading />
      ) : model.error ? (
        <ErrorState error={model.error} />
      ) : (
        <Report version={model.data.version} m={model.data.metrics} />
      )}
    </>
  );
}

function Report({ version, m }: { version: string; m: ModelMetrics }) {
  const base = m.baseline_always_nominal;
  const heldOut = m.benchmark_split === "profile_group_holdout";
  const notEvaluable = new Set(Object.values(m.not_evaluable_classes ?? {}));

  return (
    <div className="space-y-4">
      {!heldOut && (
        <Note tone="warn" title="Leaky evaluation">
          Model {version} was scored on a random row split. At 2 Hz, neighbouring rows are
          near-duplicates, so these numbers are inflated.
        </Note>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Model vs. always predicting “Nominal”">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr className="border-b border-border">
                <th className="py-2 font-medium">Metric</th>
                <th className="py-2 text-right font-medium">Model {version}</th>
                <th className="py-2 text-right font-medium">Baseline</th>
              </tr>
            </thead>
            <tbody className="tabular">
              <Row label="Macro F1" a={num(m.test_f1_macro)} b={num(base?.test_f1_macro)} better={!!base && m.test_f1_macro > base.test_f1_macro} />
              <Row label="Macro F1 (CV over drives)" a={m.cv_f1_macro_mean !== undefined ? `${num(m.cv_f1_macro_mean)} ± ${num(m.cv_f1_macro_std)}` : "—"} b="—" />
              <Row label="Macro precision" a={pct(m.test_precision_macro, 0)} b="—" />
              <Row label="Macro recall" a={pct(m.test_recall_macro, 0)} b="—" />
              <Row label="Accuracy" a={pct(m.test_accuracy)} b={pct(base?.test_accuracy)} better={!!base && m.test_accuracy > base.test_accuracy} />
            </tbody>
          </table>
          <p className="mt-3 text-xs text-muted">
            The model is tuned for recall on rare stress classes, so it trades accuracy for
            catching more stress events. Macro scores exclude classes that can&apos;t be evaluated.
          </p>
        </Card>

        <Card title="Evaluation setup">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            <Item k="Test set" v={heldOut ? `${m.test_groups} held-out drives · ${m.test_rows?.toLocaleString("en")} rows` : "random rows (leaky)"} />
            <Item k="Training set" v={m.train_groups ? `${m.train_groups} drives · ${m.train_rows?.toLocaleString("en")} rows` : "—"} />
            <Item k="Cross-validation" v={m.cv_split?.replaceAll("_", " ") ?? "—"} />
            <Item k="Class weighting" v={m.class_weighting ?? "none"} />
            <Item k="Features" v={m.features_used ? `${m.features_used.length} inputs` : "—"} />
          </dl>
          {m.leakage_notes && (
            <ul className="mt-4 list-disc space-y-1 pl-5 text-xs text-muted">
              {m.leakage_notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      {m.per_class_report && (
        <Card title="Per class">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr className="border-b border-border">
                  <th className="py-2 pr-4 font-medium">Class</th>
                  <th className="py-2 pr-4 text-right font-medium">Precision</th>
                  <th className="py-2 pr-4 text-right font-medium">Recall</th>
                  <th className="py-2 pr-4 text-right font-medium">F1</th>
                  <th className="py-2 pr-4 text-right font-medium">Test rows</th>
                  <th className="py-2 pr-4 text-right font-medium">Drives with class</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {FAULT_LABELS.map((label, code) => {
                  const c = m.per_class_report?.[label];
                  if (!c) return null;
                  const skip = notEvaluable.has(label);
                  return (
                    <tr key={label} className="border-b border-border/60 last:border-0">
                      <td className="py-2 pr-4"><FaultBadge code={code} /></td>
                      {skip ? (
                        <td colSpan={4} className="py-2 pr-4 text-right text-xs text-muted">
                          not evaluable: occurs in only one drive
                        </td>
                      ) : (
                        <>
                          <td className="py-2 pr-4 text-right">{num(c.precision)}</td>
                          <td className="py-2 pr-4 text-right">{num(c.recall)}</td>
                          <td className="py-2 pr-4 text-right">{num(c["f1-score"])}</td>
                          <td className="py-2 pr-4 text-right">{c.support.toLocaleString("en")}</td>
                        </>
                      )}
                      <td className="py-2 pr-4 text-right">{m.profiles_per_class?.[String(code)] ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {m.confusion_matrix && m.confusion_matrix_labels && (
        <Card title="Where predictions go (rows: actual, columns: predicted)">
          <ConfusionMatrix matrix={m.confusion_matrix} labels={m.confusion_matrix_labels} />
          <p className="mt-3 text-xs text-muted">
            Each row is shaded by the share of that actual class; the numbers are row counts. Read
            across a row to see what the model says when that condition is really present.
          </p>
        </Card>
      )}
    </div>
  );
}

function Row({ label, a, b, better }: { label: string; a: string; b: string; better?: boolean }) {
  return (
    <tr className="border-b border-border/60 last:border-0">
      <td className="py-2 font-sans">{label}</td>
      <td className={`py-2 text-right ${better ? "text-good" : ""}`}>{a}</td>
      <td className="py-2 text-right text-muted">{b}</td>
    </tr>
  );
}

function Item({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-muted">{k}</dt>
      <dd>{v}</dd>
    </>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const short = (l: string) => l.split(" ")[0];
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[32rem] border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} className="px-1 pb-1 text-center font-medium text-muted" title={l}>
                {short(l)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={labels[i]}>
                <th className="pr-2 text-left font-medium whitespace-nowrap text-muted">{labels[i]}</th>
                {row.map((v, j) => {
                  const share = total ? v / total : 0;
                  return (
                    <td
                      key={j}
                      className="rounded px-2 py-2 text-center"
                      style={{
                        background: `color-mix(in srgb, ${i === j ? "var(--good)" : "var(--bad)"} ${Math.round(share * 70)}%, var(--surface-2))`,
                      }}
                      title={`${labels[i]} → ${labels[j]}: ${v.toLocaleString("en")} (${(share * 100).toFixed(1)}%)`}
                    >
                      {total ? v.toLocaleString("en") : "·"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

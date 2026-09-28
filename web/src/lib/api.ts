// Typed client for the VoltGuard FastAPI service, reached through the
// same-origin proxy at /api (see app/api/[...path]/route.ts).

export const FAULT_LABELS = [
  "Nominal",
  "Stator Overheat",
  "Battery Thermal Stress",
  "Coolant System Failure",
  "Inverter Over-Current",
] as const;

export const faultColor = (code: number) => `var(--c${code})`;

export const FEATURE_UNITS: Record<string, string> = {
  ambient: "°C",
  coolant: "°C",
  u_d: "V",
  u_q: "V",
  motor_speed: "rpm",
  torque: "Nm",
  i_d: "A",
  i_q: "A",
  stator_winding: "°C",
  pm: "°C",
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

// ── Model metrics ──────────────────────────────────────────────────────────

export type ClassReport = { precision: number; recall: number; "f1-score": number; support: number };

export type ModelMetrics = {
  test_accuracy: number;
  test_f1_macro: number;
  test_precision_macro?: number;
  test_recall_macro?: number;
  cv_f1_macro_mean?: number;
  cv_f1_macro_std?: number;
  baseline_always_nominal?: { test_accuracy: number; test_f1_macro: number };
  benchmark_split?: string;
  cv_split?: string;
  class_weighting?: string;
  train_groups?: number;
  test_groups?: number;
  train_rows?: number;
  test_rows?: number;
  dataset_rows?: number;
  per_class_report?: Record<string, ClassReport>;
  confusion_matrix?: number[][];
  confusion_matrix_labels?: string[];
  not_evaluable_classes?: Record<string, string>;
  profiles_per_class?: Record<string, number>;
  class_distribution?: Record<string, number>;
  features_used?: string[];
  leakage_notes?: string[];
  per_class_precision_recall?: Record<string, { average_precision: number }>;
};

export type ModelInfo = { version: string; metrics: ModelMetrics };

export const getLatestModel = () => request<ModelInfo>("/models/latest");

// ── Data summary ───────────────────────────────────────────────────────────

export type FeatureRange = { min: number; median: number; max: number };

export type DataSummary = {
  dataset_available: boolean;
  fault_labels: Record<string, string>;
  class_distribution: Record<string, number>;
  profiles_per_class: Record<string, number>;
  profiles: { profile_id: number; rows: number; fault_rate: number }[];
  feature_ranges: Record<string, FeatureRange>;
};

export const getDataSummary = () => request<DataSummary>("/data/summary");

// ── Simulator ──────────────────────────────────────────────────────────────

export type SessionInfo = {
  session_id: string;
  vehicles: string[];
  auto_derate: boolean;
  dataset_available: boolean;
};

export type Frame = {
  tick: number;
  vehicle_id: string;
  reading: Record<string, number>;
  derated: boolean;
  predicted_fault: number;
  predicted_label: string;
  fault_score: number;
  probabilities: Record<string, number>;
  top_drivers: { feature: string; shap: number }[];
  true_fault: number | null;
  true_label: string | null;
  root_cause: string | null;
  action: string | null;
  derate_applied: boolean;
};

export const createSession = (vehicles: number, autoDerate: boolean) =>
  request<SessionInfo>("/simulator/sessions", {
    method: "POST",
    body: JSON.stringify({ vehicles, auto_derate: autoDerate }),
  });

export const tickSession = (sessionId: string, steps: number) =>
  request<{ session_id: string; tick: number; frames: Frame[] }>(
    `/simulator/sessions/${sessionId}/tick`,
    { method: "POST", body: JSON.stringify({ steps }) },
  );

export const deleteSession = (sessionId: string) =>
  request<void>(`/simulator/sessions/${sessionId}`, { method: "DELETE" });

// ── Explain ────────────────────────────────────────────────────────────────

export type Scenario = Record<
  "ambient" | "coolant" | "u_d" | "u_q" | "motor_speed" | "torque" | "i_d" | "i_q",
  number
>;

export type ExplainResult = {
  predicted_fault: number;
  fault_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  explanations: Record<string, number>;
  root_cause: string | null;
  action: string | null;
};

export const explainScenario = (scenario: Scenario) =>
  request<ExplainResult>("/predict/explain", {
    method: "POST",
    body: JSON.stringify(scenario),
  });

// ── Alert scoring (prediction vs proxy truth) ──────────────────────────────

export type AlertTally = { hits: number; falseAlarms: number; missed: number; quiet: number };

export function tallyAlerts(frames: Frame[]): AlertTally | null {
  const scored = frames.filter((f) => f.true_fault !== null);
  if (!scored.length) return null;
  const t = { hits: 0, falseAlarms: 0, missed: 0, quiet: 0 };
  for (const f of scored) {
    const alert = f.predicted_fault !== 0;
    const truth = f.true_fault !== 0;
    if (alert && truth) t.hits++;
    else if (alert) t.falseAlarms++;
    else if (truth) t.missed++;
    else t.quiet++;
  }
  return t;
}

export function verdict(f: Frame): "hit" | "false alarm" | "missed" | "ok" | "unknown" {
  if (f.true_fault === null) return "unknown";
  const alert = f.predicted_fault !== 0;
  const truth = f.true_fault !== 0;
  if (alert && truth) return "hit";
  if (alert) return "false alarm";
  if (truth) return "missed";
  return "ok";
}

export type Episode = {
  start: number;
  end: number;
  verdict: ReturnType<typeof verdict>;
  predicted: number;
  truth: number | null;
  action: string | null;
  derated: boolean;
};

/** Collapse consecutive ticks with the same verdict and classes into episodes (newest first). */
export function toEpisodes(frames: Frame[]): Episode[] {
  const out: Episode[] = [];
  for (const f of frames) {
    const v = verdict(f);
    if (v === "ok" || (v === "unknown" && f.predicted_fault === 0)) continue;
    const last = out.at(-1);
    if (last && last.end === f.tick - 1 && last.verdict === v && last.predicted === f.predicted_fault && last.truth === f.true_fault) {
      last.end = f.tick;
      last.derated ||= f.derated;
    } else {
      out.push({ start: f.tick, end: f.tick, verdict: v, predicted: f.predicted_fault, truth: f.true_fault, action: f.action, derated: f.derated });
    }
  }
  return out.reverse();
}

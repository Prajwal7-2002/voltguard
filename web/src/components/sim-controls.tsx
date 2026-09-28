"use client";

import { Pause, Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui";
import type { useSimulator } from "@/lib/use-simulator";

export function SimControls({ sim, stepLabel = "ticks" }: { sim: ReturnType<typeof useSimulator>; stepLabel?: string }) {
  const ready = !!sim.session;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={() => sim.setPlaying(!sim.playing)} disabled={!ready}>
        {sim.playing ? <Pause className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
        {sim.playing ? "Pause" : "Play"}
      </Button>
      {[1, 10, 50].map((n) => (
        <Button key={n} variant="secondary" onClick={() => sim.run(n)} disabled={!ready || sim.busy || sim.playing}>
          +{n} {n === 1 ? stepLabel.replace(/s$/, "") : stepLabel}
        </Button>
      ))}
      <Button variant="secondary" onClick={sim.reset} disabled={sim.busy}>
        <RotateCcw className="size-4" aria-hidden /> Reset
      </Button>
      <label className="ml-auto flex cursor-pointer items-center gap-2 text-sm text-muted select-none">
        <input
          type="checkbox"
          className="size-4 accent-[var(--accent)]"
          checked={sim.autoDerate}
          onChange={sim.toggleAutoDerate}
        />
        Auto-derate on alerts
      </label>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, type Frame, type SessionInfo, createSession, deleteSession, tickSession } from "./api";

const MAX_FRAMES_PER_VEHICLE = 600;

/** Owns one simulator session: creation, ticking, autoplay and cleanup. */
export function useSimulator(vehicles: number) {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [autoDerate, setAutoDerate] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<SessionInfo | null>(null);
  const busyRef = useRef(false);

  const start = useCallback(
    async (derate: boolean) => {
      setError(null);
      setPlaying(false);
      if (sessionRef.current) deleteSession(sessionRef.current.session_id).catch(() => {});
      sessionRef.current = null;
      setSession(null);
      setFrames([]);
      try {
        const s = await createSession(vehicles, derate);
        sessionRef.current = s;
        setSession(s);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [vehicles],
  );

  useEffect(() => {
    // Creating the session is an external side effect; state updates happen after the await.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    start(true);
    return () => {
      if (sessionRef.current) deleteSession(sessionRef.current.session_id).catch(() => {});
    };
  }, [start]);

  const run = useCallback(
    async (steps: number) => {
      const s = sessionRef.current;
      if (!s || busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      try {
        const res = await tickSession(s.session_id, steps);
        setFrames((prev) => [...prev, ...res.frames].slice(-MAX_FRAMES_PER_VEHICLE * vehicles));
      } catch (e) {
        setPlaying(false);
        if (e instanceof ApiError && e.status === 404) {
          setError("The simulator session expired. Press Reset to start a new one.");
        } else {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [vehicles],
  );

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => run(2), 700);
    return () => clearInterval(id);
  }, [playing, run]);

  const toggleAutoDerate = useCallback(() => {
    const next = !autoDerate;
    setAutoDerate(next);
    start(next);
  }, [autoDerate, start]);

  return {
    session,
    frames,
    busy,
    error,
    playing,
    setPlaying,
    run,
    reset: () => start(autoDerate),
    autoDerate,
    toggleAutoDerate,
  };
}

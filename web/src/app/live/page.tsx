import type { Metadata } from "next";
import { LiveMonitor } from "./live-monitor";

export const metadata: Metadata = { title: "Live monitor" };

export default function Page() {
  return <LiveMonitor />;
}

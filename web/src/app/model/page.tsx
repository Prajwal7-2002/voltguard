import type { Metadata } from "next";
import { ModelReport } from "./model-report";

export const metadata: Metadata = { title: "Model" };

export default function Page() {
  return <ModelReport />;
}

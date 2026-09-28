import type { Metadata } from "next";
import { Explain } from "./explain";

export const metadata: Metadata = { title: "Explainability" };

export default function Page() {
  return <Explain />;
}

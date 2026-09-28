"use client";

import clsx from "clsx";
import { Activity, BarChart3, Gauge, LayoutGrid, SlidersHorizontal, Zap } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/live", label: "Live monitor", icon: Activity },
  { href: "/fleet", label: "Fleet", icon: LayoutGrid },
  { href: "/explain", label: "Explainability", icon: SlidersHorizontal },
  { href: "/model", label: "Model", icon: BarChart3 },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <aside className="border-b border-border bg-surface md:sticky md:top-0 md:h-screen md:w-56 md:shrink-0 md:border-r md:border-b-0">
      <div className="flex items-center gap-2 px-4 py-3 md:py-5">
        <Zap className="size-5 text-accent" aria-hidden />
        <span className="font-semibold tracking-tight">VoltGuard</span>
      </div>
      <nav className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:overflow-visible md:pb-0">
        {LINKS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-accent-soft font-medium text-accent"
                  : "text-muted hover:bg-surface-2 hover:text-text",
              )}
            >
              <Icon className="size-4" aria-hidden />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

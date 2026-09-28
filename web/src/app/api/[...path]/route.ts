import type { NextRequest } from "next/server";

// Same-origin proxy to the FastAPI service. The target is read at request time,
// so one built image works in any environment (no CORS, no build-time URL).
const API_URL = () => (process.env.VOLTGUARD_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function proxy(req: NextRequest, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  const target = `${API_URL()}/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  try {
    const upstream = await fetch(target, {
      method: req.method,
      headers: { "Content-Type": req.headers.get("content-type") ?? "application/json" },
      body: hasBody ? await req.text() : undefined,
      cache: "no-store",
    });
    return new Response(upstream.status === 204 ? null : upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { detail: `VoltGuard API unreachable at ${API_URL()}. Is it running?` },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST, proxy as DELETE };

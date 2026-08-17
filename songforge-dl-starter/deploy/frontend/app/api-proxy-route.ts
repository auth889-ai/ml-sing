// Next.js route handler: proxies /api/songforge/* to the GPU backend.
// Place at app/api/songforge/[...path]/route.ts in a real Next.js tree.

const BACKEND = process.env.SONGFORGE_API_URL ?? "http://localhost:8000";

async function forward(req: Request, path: string[]) {
  const url = `${BACKEND}/api/${path.join("/")}`;
  const init: RequestInit = {
    method: req.method,
    headers: {
      "Content-Type": "application/json",
      // Preserve the caller's IP so backend rate limiting keys on the user,
      // not on this proxy.
      "x-forwarded-for": req.headers.get("x-forwarded-for") ?? "",
    },
  };
  if (req.method === "POST") init.body = await req.text();

  const res = await fetch(url, init);
  const type = res.headers.get("content-type") ?? "";
  if (type.startsWith("audio/")) {
    return new Response(res.body, {
      status: res.status,
      headers: {
        "content-type": type,
        "content-disposition": res.headers.get("content-disposition") ?? "",
      },
    });
  }
  return new Response(await res.text(), { status: res.status, headers: { "content-type": "application/json" } });
}

export async function GET(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
export async function POST(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}

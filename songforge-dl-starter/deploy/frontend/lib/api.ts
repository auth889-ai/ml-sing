// Thin client for the SongForge API. The browser never talks to the GPU host
// directly: requests go through the Next.js route so the backend origin stays
// private and can be rotated without shipping a new frontend.

export type ControlSupport = "native" | "prompt" | "ignored";

export interface GenerateBody {
  prompt: string;
  lyrics?: string;
  duration_seconds?: number;
  bpm?: number;
  key?: string;
  time_signature?: string;
  vocal_language?: string;
  vocal_gender?: string;
  vocal_style?: string;
  instruments?: string[];
  genre?: string[];
  mood?: string[];
  structure?: string[];
  seed?: number;
}

export interface JobState {
  job_id: string;
  status: "queued" | "running" | "done" | "failed" | "rejected" | "timeout";
  queue_position?: number;
  queue_seconds: number;
  generate_seconds: number | null;
  error: string | null;
  // Controls this deployment could NOT honour. Always show these: a knob that
  // silently does nothing is worse than a missing knob.
  control_warnings: string[];
  controls_applied?: Record<string, string>;
  audio_ready: boolean;
  audio_url?: string;
  result?: Record<string, unknown>;
}

const BASE = "/api/songforge";

/** Whether this deployment has a reachable GPU backend behind it. */
export type BackendStatus =
  | { reachable: true; modelLoaded: boolean; detail?: string }
  | { reachable: false; detail: string };

/**
 * A frontend can be deployed without a GPU host behind it — that is the normal
 * state for a preview or a demo link. Probing once on load lets the UI say so
 * plainly, instead of letting the first click return a bare "HTTP 500" that
 * reads as a broken product rather than an unconfigured one.
 */
export async function backendStatus(): Promise<BackendStatus> {
  try {
    const res = await fetch(`${BASE}/capabilities`, { cache: "no-store" });
    if (!res.ok) {
      return {
        reachable: false,
        detail: `The generation backend answered ${res.status}.`,
      };
    }
    const body = await res.json();
    return {
      reachable: true,
      modelLoaded: body?.model_loaded !== false,
      detail: typeof body?.model_error === "string" ? body.model_error : undefined,
    };
  } catch {
    return {
      reachable: false,
      detail: "No GPU backend is connected to this deployment.",
    };
  }
}

export async function submit(body: GenerateBody): Promise<JobState> {
  const res = await fetch(`${BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({} as Record<string, unknown>));
    // 5xx from this route almost always means the proxy could not reach the GPU
    // host at all. Say that, rather than surfacing the status code by itself.
    const fallback =
      res.status >= 500
        ? "The generation backend is not reachable from this deployment."
        : `Request failed (HTTP ${res.status}).`;
    throw new Error((body as { detail?: string }).detail ?? fallback);
  }
  return res.json();
}

export async function poll(jobId: string): Promise<JobState> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** Poll until the job settles. Generation is seconds, so a 1s interval is fine. */
export async function waitFor(jobId: string, onUpdate: (s: JobState) => void): Promise<JobState> {
  for (;;) {
    const state = await poll(jobId);
    onUpdate(state);
    if (state.status !== "queued" && state.status !== "running") return state;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

export async function capabilities() {
  const res = await fetch(`${BASE}/capabilities`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

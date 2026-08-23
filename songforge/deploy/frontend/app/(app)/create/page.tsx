"use client";

// Create page: compose → submit → poll → play → download → save.
//
// The composer exists because output variety is bounded by conditioning
// variety: a three-word prompt lets the model return the same thing every
// time. Control labels come from /api/capabilities so no knob is presented as
// stronger than it is — a knob that silently does nothing is worse than a
// missing knob.

import { useEffect, useRef, useState } from "react";
import {
  capabilities,
  submit,
  waitFor,
  backendStatus,
  type BackendStatus,
  type GenerateBody,
  type JobState,
} from "../../../lib/api";
import { saveSong } from "../../../lib/library";
import { compose, EMPTY, type Composition } from "../../../lib/promptkit";
import Composer from "./Composer";

interface Limits {
  min_duration_seconds: number;
  max_duration_seconds: number;
  max_prompt_chars: number;
  max_lyrics_chars: number;
}

const STAGES = ["Planning", "Rendering", "Ranking", "Mastering"];

export default function Page() {
  const [comp, setComp] = useState<Composition>(EMPTY);
  const [lyrics, setLyrics] = useState("");
  const [duration, setDuration] = useState("60");
  const [bpm, setBpm] = useState("");
  const [key, setKey] = useState("");
  const [seed, setSeed] = useState("");
  const [lockSeed, setLockSeed] = useState(false);
  const [advanced, setAdvanced] = useState(false);

  const [limits, setLimits] = useState<Limits | null>(null);
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    capabilities()
      .then((c) => setLimits(c.limits as Limits))
      .catch(() => setLimits(null));
    backendStatus().then(setBackend);
  }, []);

  const offline = backend !== null && !backend.reachable;
  const modelDown = backend?.reachable === true && backend.modelLoaded === false;
  const prompt = compose(comp);

  // Which pipeline stage to highlight. Queued work has not started planning yet.
  const stageIndex =
    job?.status === "running" ? 1 : job?.status === "done" ? STAGES.length : job ? 0 : -1;

  async function run() {
    setError(null);
    setJob(null);

    const usedSeed =
      lockSeed && seed.trim() !== ""
        ? Number(seed)
        : Math.floor(Math.random() * 2_147_483_647);
    if (!lockSeed) setSeed(String(usedSeed));

    const body: GenerateBody = {
      prompt,
      seed: usedSeed,
      duration_seconds: Number(duration) || undefined,
    };
    if (lyrics.trim()) body.lyrics = lyrics.trim();
    if (bpm) body.bpm = Number(bpm);
    if (key.trim()) body.key = key.trim();

    setBusy(true);
    try {
      const submitted = await submit(body);
      jobIdRef.current = submitted.job_id;
      setJob(submitted);
      const settled = await waitFor(submitted.job_id, (s) => {
        if (jobIdRef.current === submitted.job_id) setJob(s);
      });
      if (jobIdRef.current !== submitted.job_id) return;
      setJob(settled);
      if (settled.status === "done") {
        saveSong({
          id: settled.job_id,
          jobId: settled.job_id,
          prompt,
          createdAt: Date.now(),
          durationSeconds: body.duration_seconds,
          seed: usedSeed,
        });
      } else {
        setError(settled.error ?? `Job ${settled.status}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const audioUrl =
    job?.status === "done" && job.audio_url ? `/api/songforge/jobs/${job.job_id}/audio` : null;

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Create</span>
        <h1>
          Compose a prompt, get a <span className="accent-text">finished track</span>
        </h1>
        <p className="lede">
          The more the model is told, the less two songs resemble each other. Type freely,
          then sharpen it with the options below.
        </p>
      </header>

      {offline && (
        <div className="note note-warn" style={{ marginBottom: 16 }}>
          <strong>No GPU backend is connected to this deployment.</strong>
          <p style={{ marginTop: 6 }}>
            The interface is fully functional, but generation needs a machine with a GPU
            behind it. Set <code>SONGFORGE_API_URL</code> to point at one, or run the backend
            locally — the procedure is in <code>deploy/RUNBOOK.md</code>.
          </p>
        </div>
      )}

      {modelDown && (
        <div className="note note-warn" style={{ marginBottom: 16 }}>
          <strong>Backend reachable, but its model is not loaded.</strong>
          {backend?.detail && (
            <p style={{ marginTop: 6, fontFamily: "ui-monospace, monospace", fontSize: "0.78rem" }}>
              {backend.detail}
            </p>
          )}
        </div>
      )}

      <section className="card">
        <Composer value={comp} onChange={setComp} />
      </section>

      <section className="card">
        <label htmlFor="lyrics">Lyrics — optional, leave empty for instrumental</label>
        <textarea
          id="lyrics"
          rows={3}
          maxLength={limits?.max_lyrics_chars}
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder={"[verse]\nYour lines here…"}
        />

        <div className="row">
          <div>
            <label htmlFor="duration">Duration (s)</label>
            <input
              id="duration"
              type="number"
              min={limits?.min_duration_seconds ?? 10}
              max={limits?.max_duration_seconds ?? 120}
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
            />
            {limits && (
              <div className="hint">
                {limits.min_duration_seconds}–{limits.max_duration_seconds}s here
              </div>
            )}
          </div>
          <div>
            <label htmlFor="seed">Seed</label>
            <input
              id="seed"
              type="number"
              value={seed}
              placeholder="random each time"
              onChange={(e) => setSeed(e.target.value)}
            />
            <label htmlFor="lockseed" style={{ display: "flex", gap: 7, alignItems: "center", marginTop: 7 }}>
              <input
                id="lockseed"
                type="checkbox"
                checked={lockSeed}
                onChange={(e) => setLockSeed(e.target.checked)}
                style={{ width: "auto" }}
              />
              Lock this seed
            </label>
            <div className="hint">
              {lockSeed ? "Locked — same prompt returns the same song." : "New seed each render."}
            </div>
          </div>
        </div>

        <button
          type="button"
          className="btn-ghost"
          style={{ marginTop: 14, padding: "7px 13px", fontSize: "0.82rem" }}
          onClick={() => setAdvanced((v) => !v)}
        >
          {advanced ? "Hide" : "Show"} advanced controls
        </button>

        {advanced && (
          <div className="row" style={{ marginTop: 12 }}>
            <div>
              <label htmlFor="bpm">BPM</label>
              <input id="bpm" type="number" min={30} max={300} value={bpm} placeholder="auto"
                onChange={(e) => setBpm(e.target.value)} />
            </div>
            <div>
              <label htmlFor="key">Key</label>
              <input id="key" value={key} placeholder="e.g. A minor"
                onChange={(e) => setKey(e.target.value)} />
            </div>
          </div>
        )}

        <div style={{ marginTop: 18, display: "flex", gap: 9, flexWrap: "wrap" }}>
          <button onClick={run} disabled={busy || !prompt.trim() || offline}>
            {busy ? "Generating…" : offline ? "Backend not connected" : "Generate song"}
          </button>
          {job?.status === "done" && !busy && (
            <button className="btn-ghost" onClick={run}>
              Generate a variation
            </button>
          )}
        </div>

        {job && (
          <div className="pipeline">
            {STAGES.map((s, i) => (
              <div
                key={s}
                className={`pstage${i < stageIndex ? " pstage-done" : i === stageIndex ? " pstage-on" : ""}`}
              >
                {s}
              </div>
            ))}
          </div>
        )}

        {job && (job.status === "queued" || job.status === "running") && (
          <div className="note note-info">
            {job.status === "queued"
              ? `Queued${job.queue_position != null ? ` — position ${job.queue_position}` : ""}…`
              : "Rendering on the GPU…"}
          </div>
        )}

        {job && job.control_warnings.length > 0 && (
          <div className="note note-warn">
            {job.control_warnings.map((w) => (
              <div key={w}>⚠ {w}</div>
            ))}
          </div>
        )}

        {error && <div className="note note-error">{error}</div>}

        {audioUrl && (
          <div>
            <div className="note note-ok">Done — saved to your library.</div>
            <audio controls src={audioUrl} />
            <div className="dl-row">
              <a className="dl" href={audioUrl} download={`songforge_${job!.job_id}.wav`}>
                ↓ Download WAV
              </a>
            </div>
          </div>
        )}
      </section>

      <footer>
        Foundation: ACE-Step 1.5 XL-turbo (MIT, pretrained — not ours). Adapter, planner,
        ranking, originality screen, finishing and this service: SongForge.
      </footer>
    </>
  );
}

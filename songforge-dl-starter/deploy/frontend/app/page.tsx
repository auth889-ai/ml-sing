"use client";

// The whole product surface: form → submit → poll → play → download.
// Control labels come from /api/capabilities so no knob is presented as
// stronger than it is — "native" controls condition the model, "prompt"
// controls are only text the model may or may not follow.

import { useEffect, useRef, useState } from "react";
import { capabilities, submit, waitFor, type GenerateBody, type JobState } from "../lib/api";

interface Limits {
  min_duration_seconds: number;
  max_duration_seconds: number;
  max_prompt_chars: number;
  max_lyrics_chars: number;
}

export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [duration, setDuration] = useState("60");
  const [bpm, setBpm] = useState("");
  const [key, setKey] = useState("");
  const [timeSignature, setTimeSignature] = useState("");
  const [language, setLanguage] = useState("en");
  const [seed, setSeed] = useState("0");

  const [limits, setLimits] = useState<Limits | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    capabilities()
      .then((c) => setLimits(c.limits as Limits))
      .catch(() => setLimits(null)); // backend warming up; submit still validates server-side
  }, []);

  async function onGenerate() {
    setError(null);
    setJob(null);
    const body: GenerateBody = {
      prompt: prompt.trim(),
      seed: Number(seed) || 0,
      duration_seconds: Number(duration) || undefined,
      vocal_language: language || undefined,
    };
    if (lyrics.trim()) body.lyrics = lyrics.trim();
    if (bpm) body.bpm = Number(bpm);
    if (key.trim()) body.key = key.trim();
    if (timeSignature.trim()) body.time_signature = timeSignature.trim();

    setBusy(true);
    try {
      const submitted = await submit(body);
      jobIdRef.current = submitted.job_id;
      setJob(submitted);
      const settled = await waitFor(submitted.job_id, (s) => {
        // Ignore stale polls if the user resubmitted meanwhile.
        if (jobIdRef.current === submitted.job_id) setJob(s);
      });
      if (jobIdRef.current === submitted.job_id) {
        setJob(settled);
        if (settled.status !== "done") setError(settled.error ?? `Job ${settled.status}`);
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
      <h1>SongForge</h1>
      <p className="sub">
        Describe a song. The backend generates it with ACE-Step 1.5 XL-turbo plus the
        SongForge adapter, then you can listen and download the WAV.
      </p>

      <label htmlFor="prompt">Prompt — instruments, style, mood (required)</label>
      <textarea
        id="prompt"
        rows={3}
        maxLength={limits?.max_prompt_chars}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="warm acoustic ballad, fingerpicked guitar, soft female vocal, gentle strings"
      />

      <label htmlFor="lyrics">Lyrics (optional — leave empty for instrumental or model-written)</label>
      <textarea
        id="lyrics"
        rows={5}
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
              {limits.min_duration_seconds}–{limits.max_duration_seconds}s on this deployment
            </div>
          )}
        </div>
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
        <div>
          <label htmlFor="ts">Time signature</label>
          <input id="ts" value={timeSignature} placeholder="e.g. 4/4"
            onChange={(e) => setTimeSignature(e.target.value)} />
        </div>
        <div>
          <label htmlFor="lang">Vocal language</label>
          <input id="lang" value={language} onChange={(e) => setLanguage(e.target.value)} />
        </div>
        <div>
          <label htmlFor="seed">Seed</label>
          <input id="seed" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
          <div className="hint">Same inputs + same seed → same song</div>
        </div>
      </div>

      <button onClick={onGenerate} disabled={busy || !prompt.trim()}>
        {busy ? "Generating…" : "Generate"}
      </button>

      {job && (job.status === "queued" || job.status === "running") && (
        <p className="status">
          {job.status === "queued"
            ? `Queued${job.queue_position != null ? ` — position ${job.queue_position}` : ""}…`
            : "Rendering on the GPU…"}
        </p>
      )}

      {job && job.control_warnings.length > 0 && (
        <div className="warn">
          {job.control_warnings.map((w) => (
            <div key={w}>⚠ {w}</div>
          ))}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {audioUrl && (
        <div className="player">
          <audio controls src={audioUrl} />
          <a className="download" href={audioUrl} download={`songforge_${job!.job_id}.wav`}>
            Download WAV
          </a>
        </div>
      )}

      <footer>
        Base model: ACE-Step 1.5 XL-turbo (MIT, pretrained — not ours). Adapter, control
        layer, pipeline and this service: SongForge. Generated audio is deleted from the
        server a few hours after rendering.
      </footer>
    </>
  );
}

// Saved library, persisted in the visitor's own browser.
//
// Deliberately localStorage rather than a server table: the backend is a
// single GPU with no user accounts, and inventing auth to store a list of
// prompts would add a login wall to a demo whose whole point is that a
// stranger can try it in one click. Each visitor keeps their own shelf.

export interface SavedSong {
  id: string;
  prompt: string;
  createdAt: number;
  durationSeconds?: number;
  seed?: number;
  /** Backend job id — the audio route stays valid while that job is retained. */
  jobId: string;
}

const KEY = "songforge.library.v1";
const LIMIT = 60;

/** Every accessor is guarded: storage throws in private windows and previews. */
function read(): SavedSong[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as SavedSong[]) : [];
  } catch {
    return [];
  }
}

function write(songs: SavedSong[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(songs.slice(0, LIMIT)));
    window.dispatchEvent(new Event("songforge:library"));
  } catch {
    /* storage unavailable — the app still works, it just cannot remember */
  }
}

export function listSongs(): SavedSong[] {
  return read().sort((a, b) => b.createdAt - a.createdAt);
}

export function saveSong(song: SavedSong): void {
  const existing = read().filter((s) => s.id !== song.id);
  write([song, ...existing]);
}

export function removeSong(id: string): void {
  write(read().filter((s) => s.id !== id));
}

export function clearLibrary(): void {
  write([]);
}

export function audioUrlFor(song: SavedSong): string {
  return `/api/songforge/jobs/${song.jobId}/audio`;
}

export function formatWhen(ts: number): string {
  const mins = Math.round((Date.now() - ts) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(ts).toLocaleDateString();
}

// Vocabulary for the prompt composer.
//
// Output variety is bounded by conditioning variety: a three-word prompt gives
// the model almost nothing to separate one render from the next, and the result
// is the "every song sounds the same" complaint. Composing genre, mood,
// instrumentation, vocal and structure into the caption gives each render a
// materially different conditioning vector before a single model change.
//
// These lists mirror the capability families the adapter was actually trained
// on, so the chips promise things the corpus can support rather than every
// genre a user might imagine.

export interface Choice {
  id: string;
  label: string;
  /** Text contributed to the composed prompt. */
  phrase: string;
}

export const GENRES: Choice[] = [
  { id: "cinematic", label: "Cinematic", phrase: "cinematic orchestral" },
  { id: "classical", label: "Classical", phrase: "classical chamber" },
  { id: "folk", label: "Folk", phrase: "acoustic folk" },
  { id: "rock", label: "Rock", phrase: "rock" },
  { id: "jazz", label: "Jazz", phrase: "jazz" },
  { id: "electronic", label: "Electronic", phrase: "electronic" },
  { id: "ambient", label: "Ambient", phrase: "ambient" },
  { id: "pop", label: "Pop", phrase: "pop" },
  { id: "hiphop", label: "Hip-hop", phrase: "hip-hop" },
  { id: "lofi", label: "Lo-fi", phrase: "lo-fi" },
];

export const MOODS: Choice[] = [
  { id: "uplifting", label: "Uplifting", phrase: "uplifting and bright" },
  { id: "melancholy", label: "Melancholy", phrase: "melancholic and wistful" },
  { id: "epic", label: "Epic", phrase: "epic and sweeping" },
  { id: "calm", label: "Calm", phrase: "calm and spacious" },
  { id: "tense", label: "Tense", phrase: "tense and driving" },
  { id: "warm", label: "Warm", phrase: "warm and intimate" },
  { id: "dark", label: "Dark", phrase: "dark and brooding" },
  { id: "playful", label: "Playful", phrase: "playful and light" },
];

export const INSTRUMENTS: Choice[] = [
  { id: "piano", label: "Grand piano", phrase: "grand piano" },
  { id: "violin", label: "Violin", phrase: "violin" },
  { id: "cello", label: "Cello", phrase: "cello" },
  { id: "strings", label: "String ensemble", phrase: "string ensemble" },
  { id: "aguitar", label: "Acoustic guitar", phrase: "fingerpicked acoustic guitar" },
  { id: "eguitar", label: "Electric guitar", phrase: "electric guitar" },
  { id: "bass", label: "Bass", phrase: "bass guitar" },
  { id: "drums", label: "Drums", phrase: "live drum kit" },
  { id: "synth", label: "Synth", phrase: "analog synth pads" },
  { id: "brass", label: "Brass", phrase: "brass section" },
  { id: "woodwind", label: "Woodwinds", phrase: "woodwinds" },
  { id: "keys", label: "Keys", phrase: "electric piano" },
];

export const VOCALS: Choice[] = [
  { id: "none", label: "Instrumental", phrase: "" },
  { id: "female", label: "Female vocal", phrase: "soft female lead vocal" },
  { id: "male", label: "Male vocal", phrase: "male lead vocal" },
  { id: "choir", label: "Choir", phrase: "layered choir" },
];

export const STRUCTURES: Choice[] = [
  { id: "none", label: "Let the model decide", phrase: "" },
  { id: "build", label: "Slow build", phrase: "sparse intro building to a full-band climax" },
  { id: "versechorus", label: "Verse / chorus", phrase: "clear verse and chorus sections" },
  { id: "loop", label: "Steady groove", phrase: "steady looping groove" },
  { id: "arc", label: "Rise and fall", phrase: "rising then resolving arc" },
];

export interface Composition {
  genres: string[];
  moods: string[];
  instruments: string[];
  vocal: string;
  structure: string;
  freeText: string;
}

export const EMPTY: Composition = {
  genres: [],
  moods: [],
  instruments: [],
  vocal: "none",
  structure: "none",
  freeText: "",
};

function phrases(all: Choice[], ids: string[]): string[] {
  return ids
    .map((id) => all.find((c) => c.id === id)?.phrase)
    .filter((p): p is string => Boolean(p));
}

/**
 * Fold the selections and the free-text box into one caption.
 *
 * Free text leads: whatever the user actually typed is the strongest signal of
 * intent, and the chips refine it rather than overrule it.
 */
export function compose(c: Composition): string {
  const parts: string[] = [];
  const free = c.freeText.trim();
  if (free) parts.push(free);

  const genre = phrases(GENRES, c.genres);
  if (genre.length) parts.push(genre.join(" and "));

  const mood = phrases(MOODS, c.moods);
  if (mood.length) parts.push(mood.join(", "));

  const instruments = phrases(INSTRUMENTS, c.instruments);
  if (instruments.length) parts.push(instruments.join(", "));

  const vocal = VOCALS.find((v) => v.id === c.vocal)?.phrase;
  if (vocal) parts.push(vocal);
  else if (c.vocal === "none" && !free) parts.push("instrumental");

  const structure = STRUCTURES.find((s) => s.id === c.structure)?.phrase;
  if (structure) parts.push(structure);

  return parts.join(", ");
}

/** How much conditioning the model will actually receive, 0–100. */
export function richness(c: Composition): number {
  const signals =
    (c.freeText.trim() ? 2 : 0) +
    Math.min(c.genres.length, 2) +
    Math.min(c.moods.length, 2) +
    Math.min(c.instruments.length, 3) +
    (c.vocal !== "none" ? 1 : 0) +
    (c.structure !== "none" ? 1 : 0);
  return Math.min(100, Math.round((signals / 9) * 100));
}

"use client";

// Prompt composer: chips for genre, mood, instrumentation, vocal and structure,
// folded into one caption alongside whatever the user typed.
//
// This exists for output quality, not decoration. Conditioning variety bounds
// output variety — a three-word prompt leaves the model free to return the same
// thing every time, which is the complaint that kills a generator's credibility
// on second use. The meter shows how much signal the model will actually get,
// so a thin prompt is visible before it produces a thin song.

import {
  GENRES, MOODS, INSTRUMENTS, VOCALS, STRUCTURES,
  compose, richness, type Composition, type Choice,
} from "../../../lib/promptkit";

function ChipRow({
  choices, selected, onToggle, multi = true,
}: {
  choices: Choice[];
  selected: string[];
  onToggle: (id: string) => void;
  multi?: boolean;
}) {
  return (
    <div className="chip-row">
      {choices.map((c) => {
        const on = selected.includes(c.id);
        return (
          <button
            key={c.id}
            type="button"
            className={`chip${on ? " chip-on" : ""}`}
            aria-pressed={multi ? on : undefined}
            onClick={() => onToggle(c.id)}
          >
            {c.label}
          </button>
        );
      })}
    </div>
  );
}

export default function Composer({
  value, onChange,
}: {
  value: Composition;
  onChange: (next: Composition) => void;
}) {
  const set = (patch: Partial<Composition>) => onChange({ ...value, ...patch });

  const toggleMulti = (key: "genres" | "moods" | "instruments", id: string) => {
    const list = value[key];
    set({ [key]: list.includes(id) ? list.filter((x) => x !== id) : [...list, id] } as Partial<Composition>);
  };

  const composed = compose(value);
  const score = richness(value);

  return (
    <>
      <label htmlFor="freetext">Describe your song</label>
      <textarea
        id="freetext"
        rows={2}
        value={value.freeText}
        onChange={(e) => set({ freeText: e.target.value })}
        placeholder="a hopeful theme for a film ending"
      />

      <label>Genre</label>
      <ChipRow choices={GENRES} selected={value.genres} onToggle={(id) => toggleMulti("genres", id)} />

      <label>Mood</label>
      <ChipRow choices={MOODS} selected={value.moods} onToggle={(id) => toggleMulti("moods", id)} />

      <label>Instruments</label>
      <ChipRow
        choices={INSTRUMENTS}
        selected={value.instruments}
        onToggle={(id) => toggleMulti("instruments", id)}
      />

      <div className="row">
        <div>
          <label>Vocals</label>
          <ChipRow
            choices={VOCALS}
            selected={[value.vocal]}
            multi={false}
            onToggle={(id) => set({ vocal: id })}
          />
        </div>
        <div>
          <label>Structure</label>
          <ChipRow
            choices={STRUCTURES}
            selected={[value.structure]}
            multi={false}
            onToggle={(id) => set({ structure: id })}
          />
        </div>
      </div>

      <div className="composed">
        <div className="composed-head">
          <span className="composed-label">Prompt sent to the model</span>
          <span className="pill">{score}% detail</span>
        </div>
        <p className="composed-text">
          {composed || <span style={{ color: "var(--muted)" }}>Pick a few options above…</span>}
        </p>
        <div className="meter" aria-hidden="true">
          <span style={{ width: `${score}%` }} />
        </div>
        <p className="hint">
          {score < 35
            ? "Thin prompts leave the model free to repeat itself. Add instruments or a mood."
            : score < 70
            ? "Good — more instrumentation or a structure will sharpen it further."
            : "Rich prompt: the model has plenty to work with."}
        </p>
      </div>
    </>
  );
}

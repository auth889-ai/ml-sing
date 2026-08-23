// How it works — the pipeline, stage by stage.
//
// A generator that returns one take from one prompt is a demo. The stages
// below are what make it a product, so they are stated plainly rather than
// hidden behind a single "Generate" button.

const STAGES = [
  {
    icon: "✎",
    tone: "ico-coral",
    title: "Free-form planner",
    body:
      "Your prompt is parsed into genre, mood, instruments, BPM, key, structure and energy. Anything you set explicitly always overrides what the planner inferred.",
  },
  {
    icon: "◈",
    tone: "ico-amber",
    title: "Foundation + our adapter",
    body:
      "ACE-Step 1.5 XL-turbo stays frozen. A SongForge-trained LoKr adapter — 1,835,008 parameters, 0.04% of the 4,991,023,206-parameter base — shifts instrumentation and realism.",
  },
  {
    icon: "◎",
    tone: "ico-green",
    title: "Best-of-N ranking",
    body:
      "Several takes are rendered and scored, and the best one is returned. Broken takes — silence, clipping, collapse — are rejected before you ever hear them.",
  },
  {
    icon: "⚖",
    tone: "ico-coral",
    title: "Originality screen",
    body:
      "The winning take is checked against a corpus fingerprint database using a PCM hash and chroma-sequence similarity, so a track that leans too close to its training data gets flagged.",
  },
  {
    icon: "◐",
    tone: "ico-amber",
    title: "Conservative finishing",
    body:
      "Loudness, peak control and fades — enough to make it listenable, not enough to squash it — then exported as WAV and MP3.",
  },
];

export default function HowItWorksPage() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">How it works</span>
        <h1>
          Five stages between your prompt and a <span className="accent-text">finished master</span>
        </h1>
        <p className="lede">
          Most of the perceived quality comes from what happens around the model, not just
          the model itself.
        </p>
      </header>

      <div className="grid">
        {STAGES.map((stage) => (
          <section className="card" key={stage.title}>
            <div className={`feature-ico ${stage.tone}`} aria-hidden="true">
              {stage.icon}
            </div>
            <h2 className="card-title" style={{ marginBottom: 6 }}>{stage.title}</h2>
            <p style={{ color: "var(--text-2)", fontSize: "0.88rem" }}>{stage.body}</p>
          </section>
        ))}
      </div>

      <section className="card" style={{ marginTop: 20 }}>
        <h2 className="card-title">What is ours, and what is not</h2>
        <dl className="kv">
          <dt>Pretrained foundation</dt>
          <dd>ACE-Step 1.5 XL-turbo (MIT) — not ours, used frozen, never retrained</dd>
          <dt>Trained by us</dt>
          <dd>LoKr adapter, 1,835,008 trainable parameters (0.04%)</dd>
          <dt>Built by us</dt>
          <dd>Corpus, ten-gate admission chain, planner, ranking, originality, finishing, API, this app</dd>
        </dl>
        <p className="hint" style={{ marginTop: 10 }}>
          Adapting a strong open foundation is the correct engineering choice at this scale.
          Claiming to have trained a five-billion-parameter music model from scratch would not be.
        </p>
      </section>
    </>
  );
}

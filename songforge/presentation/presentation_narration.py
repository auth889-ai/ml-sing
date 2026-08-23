# Scene list for the SongForge presentation video.
#
# Every factual claim below is checkable against the repository. Nothing here
# says or implies that a song was generated for this video -- no song has been,
# because the foundation model needs a GPU we do not currently have, and a
# demo video that fakes its own result is worth less than no video at all.

SCENES = [
    dict(
        key="01_title", kind="title",
        title="SongForge",
        subtitle="Text to song, with every second of training audio licence-cleared",
        voice=(
            "SongForge. Text to song, with every second of training audio "
            "licence cleared."
        ),
    ),
    dict(
        key="02_problem", kind="bullets",
        title="Two ways music tools fail",
        bullets=[
            "Closed APIs — training data unseen, licensing unauditable",
            "Research checkpoints — a wave file in a notebook, and nothing more",
        ],
        kicker="A musician cannot use them. A lawyer cannot clear them.",
        voice=(
            "Generative music tools tend to fail in one of two ways. Either "
            "they are closed A P Is. You cannot see the training data, cannot "
            "audit the licensing, and cannot run them yourself. Or they are "
            "research checkpoints that write a wave file in a notebook and "
            "stop there. Both leave the same gap. A musician cannot use them, "
            "and a lawyer cannot clear them."
        ),
    ),
    dict(
        key="03_home", kind="shot", shot="00_home.png",
        caption="Deployed and live",
        voice=(
            "SongForge closes that gap, and it is a deployed web application, "
            "not a notebook. You describe a song in plain language. The system "
            "plans its structure, renders several candidate takes, ranks them, "
            "screens them for similarity against the training corpus, and "
            "masters the result."
        ),
    ),
    dict(
        key="04_create", kind="shot", shot="03_create.png",
        caption="The composer",
        voice=(
            "There is a prompt composer for genre, mood, instruments and "
            "vocals, with a live detail meter showing how much conditioning "
            "the model will actually receive."
        ),
    ),
    dict(
        key="05_library", kind="shot", shot="04_library.png",
        caption="Your library stays in your browser",
        voice=(
            "Saved songs live in your browser and never reach a server. There "
            "is no account wall, because there is no server side user data to "
            "protect. An account would gate the interface without protecting "
            "anything."
        ),
    ),
    dict(
        key="06_ownership", kind="table",
        title="What is ours, and what is not",
        rows=[
            ("Audio foundation — ACE-Step 1.5", "NOT OURS", "4,991,023,206 params, MIT, frozen"),
            ("LoKr adapter", "OURS", "1,835,008 params trained"),
            ("Neural audio codec", "OURS", "5,068,481 params, from scratch"),
            ("Structure planner", "OURS", "transformer encoder"),
            ("Corpus + licence gates", "OURS", "106,574 tracks censused"),
            ("Product — API, ranking, web app", "OURS", "~15,600 lines"),
        ],
        voice=(
            "Here is what is ours and what is not, stated plainly. The audio "
            "foundation is ACE-Step one point five. Five billion parameters, "
            "M I T licensed, used frozen. We did not build it, and we do not "
            "claim to. What we built is the adapter trained on top of it, a "
            "neural audio codec written from scratch, the structure planner, "
            "the corpus and its licence gates, and the entire product around "
            "them."
        ),
    ),
    dict(
        key="07_census", kind="stat",
        big="8,839",
        big_label="deployable tracks, from 106,574 censused",
        detail="97,735 discarded — NonCommercial, ShareAlike, NoDerivatives, unresolved",
        voice=(
            "The Free Music Archive ships one hundred and six thousand, five "
            "hundred and seventy four tracks, licensed individually. We "
            "censused every one of them. Eight thousand, eight hundred and "
            "thirty nine were C C zero or C C B Y, and deployable. We "
            "discarded ninety seven thousand tracks to keep the clean ones."
        ),
    ),
    dict(
        key="08_gates", kind="bullets",
        title="Ten gates, in order",
        bullets=[
            "licence → provenance → duplicate → integrity → silence",
            "clipping → split leakage → metadata → caption → deployability",
        ],
        kicker="A record failing any gate is excluded and counted in the report.",
        voice=(
            "Every record passes ten gates in order. Licence, provenance, "
            "cross corpus duplicates, integrity, silence, clipping, split "
            "leakage, metadata, caption quality, and deployability. Anything "
            "that fails is excluded, and counted in the corpus report."
        ),
    ),
    dict(
        key="09_codec", kind="table",
        title="Our codec — 64 held-out segments",
        rows=[
            ("Signal-to-noise", "−2.81 dB  →  +3.64 dB", "+6.45 dB"),
            ("Reconstruction loss", "0.281  →  0.146", "−47.9%"),
            ("Waveform L1", "0.119  →  0.061", "−49.2%"),
            ("Multi-resolution STFT", "0.162  →  0.086", "−47.0%"),
            ("Compression vs PCM16", "100×  at 3,840 bps", "runs on CPU"),
        ],
        voice=(
            "Our own codec is five million parameters. A convolutional "
            "encoder, residual vector quantisation, a convolutional decoder, "
            "trained from random initialisation. On sixty four held out "
            "segments, signal to noise went from minus two point eight "
            "decibels to plus three point six. Reconstruction loss fell forty "
            "eight percent. It compresses audio one hundred times against P C "
            "M sixteen, and unlike the foundation model, it runs on a C P U."
        ),
    ),
    dict(
        key="10_limits", kind="bullets",
        title="Stated honestly",
        bullets=[
            "Live generation needs a GPU with compute capability ≥ 8.0",
            "V1 is instrumental only — Slakh has no vocals",
            "A 0.04% adapter steers a foundation; it does not replace one",
            "Originality screening reduces risk. It does not guarantee.",
        ],
        kicker="The app reports a missing GPU rather than failing silently.",
        voice=(
            "And the honest limits. Live song generation needs a G P U with "
            "compute capability eight or higher. The public deployment does "
            "not have one attached yet, and the application says so rather "
            "than failing silently. Version one is instrumental only, because "
            "Slakh contains no vocals. A zero point zero four percent adapter "
            "steers a foundation model. It does not replace one. And "
            "originality screening reduces risk. It does not guarantee "
            "anything."
        ),
    ),
    dict(
        key="11_close", kind="title",
        title="SongForge",
        subtitle="github.com/auth889-ai/ml-sing   ·   frontend-ashen-mu-51.vercel.app",
        voice=(
            "SongForge. The code, the licence census, and every number in this "
            "video are in the repository. Thank you for watching."
        ),
    ),
]

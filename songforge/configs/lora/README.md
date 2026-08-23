# LoRA experiment configs

One YAML per experiment. Never one giant LoRA over everything: each config names
the weakness it targets, the dataset that addresses it, and the ablation that
will decide whether it is kept.

Naming: `<order>_<corpus>_<goal>_<variant>.yaml`, e.g. `lora01_slakh_instrument_r16.yaml`.

Required keys: `goal`, `targets_weakness`, `dataset`, `licence_class`,
`base_checkpoint`, `adapter` (lora|lokr), `rank`, `lr`, `epochs`, `ablation`.

`licence_class` must be `permissive` for anything destined for the deployable
adapter line. Research-only adapters carry `research-only` and are never merged
into it.

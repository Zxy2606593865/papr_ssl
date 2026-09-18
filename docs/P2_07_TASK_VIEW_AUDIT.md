# P2-07A Task View / Label Inventory Audit

## Purpose

The raw MDSC manifest contains 3,858 distinct transcript strings. P2-07A does
not assume these are 3,858 final KWS classes.

Instead, it constructs descriptive candidate inventories using only observed
manifest metadata.

## Candidate MDSC inventories

- `all_transcripts`
- `seen_in_all_speakers`
- `control_and_dysarthria_overlap`
- `train_and_eval_overlap`
- `enrollment_and_eval_overlap`
- `train_enrollment_eval_overlap`
- `eval_only_relative_to_enrollment`
- `enrollment_only_relative_to_eval`

These names describe set relationships only.

For example:

```text
enrollment_and_eval_overlap
```

means that the exact transcript appears at least once in both roles. It does
NOT yet mean that the transcript is officially a target keyword.

## GSC development view

For PAPR-SSL representation/backbone screening, P2-07A records a simple
project development view:

```text
gsc_all_35_closed_set
```

using all 35 observed Speech Commands labels while preserving the official
train/dev/test split.

This is a project-side evaluation view, not a claim about the only official
Speech Commands benchmark protocol.

## Next decision: P2-07B

After inspecting the real candidate inventory, P2-07B freezes the semantic task
policy:

- which MDSC phrases are eligible personalized targets;
- what data can act as non-target / unknown speech;
- how enrollment and eval are separated;
- whether command phrases and wake words are evaluated together or separately;
- which view is used for backbone/layer selection versus final domain
  qualification.

No SCAF training should begin before P2-07B is frozen.

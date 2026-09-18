# P2-07D — MDSC Phrase Inventory Audit

Purpose: determine how many MDSC phrases have enough speaker/domain support for
phrase-level embedding experiments beyond the 10-wake-word auxiliary benchmark.

This stage is descriptive only. It does not rewrite the manifest, merge semantic
synonyms, declare 3858 classes, or freeze a 30/50/100-class task.

Candidate support is measured from train split only:
- Control/train speaker coverage
- Dysarthria/train speaker coverage

Dev/test counts are reported only descriptively until the phrase set is frozen.

Outputs:
- mdsc_phrase_inventory.csv
- mdsc_phrase_inventory_summary.json

The threshold sweep is used to see whether a natural 30/50/100/200-class
operating point exists.

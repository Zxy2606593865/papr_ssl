# P6 Unknown Exposure Audit v5

v4 returned a syntactically successful but scientifically invalid result:

```text
manifest = gsc_v2.jsonl
84,843 train rows
domain = standard
```

That is Google Speech Commands, not MDSC.

It also reported only one open-set DEV phrase because the open-set task index
uses a generic task label for the reject class. The real unknown phrase must be
recovered by mapping each open-set DEV `utt_id` back to the full MDSC manifest.

v5 fixes both problems:

1. Auto-discovery requires at least 5,000 rows explicitly identified as MDSC.
2. Open-set DEV real phrase identities are recovered from the full MDSC
   manifest using the 1,178 open-set DEV utterance IDs.
3. Candidate exposure is restricted to MDSC train only.
4. Core30 and open-set DEV phrase identities are excluded.
5. generic_test remains sealed.

Run:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v5.py
python scripts/review_p6_unknown_exposure_audit_v5.py
```

If auto-discovery refuses to choose a manifest, list likely MDSC manifests:

```powershell
Get-ChildItem artifacts -Recurse -Filter *.jsonl |
  Where-Object {$_.FullName -match 'mdsc'} |
  Select-Object FullName
```

Then pass the correct full MDSC manifest explicitly:

```powershell
python scripts/audit_p6_unknown_exposure_candidates_v5.py `
  --mdsc-manifest "artifacts\...\mdsc....jsonl"
```

# H0-02 — Unseen-Speaker Personalized Episode Audit

H0-01 found:

```text
train speakers = 34
dev speakers = 4
train/dev overlap = 0
```

That means the official split is speaker-disjoint.

For ordinary "support must be in train and query must be in dev" logic this
looks blocked. But for **few-shot personalization of a new user**, speaker
disjointness is actually desirable.

The correct protocol is:

```text
HEAD-FIT:
    TRAIN speaker
    ├─ support from that speaker
    └─ different recordings as query

UNSEEN-SPEAKER DEVELOPMENT:
    DEV speaker (never used to fit shared Head parameters)
    ├─ reserve N recordings as personal support
    └─ use remaining recordings as query

FINAL TEST:
    unseen TEST speaker
    ├─ N labeled support recordings are allowed
    └─ remaining query recordings are scored
```

This is the standard logical shape of few-shot personalization: the test user
may supply a small labeled support set without becoming a training speaker.

H0-02 therefore audits whether each speaker/phrase has at least:

```text
N support recordings + 1 independent query recording
```

for N in:

```text
1 / 2 / 5 / 10 / 15
```

Run:

```powershell
python scripts/audit_mdsc_unseen_speaker_episodes.py
```

No audio, embeddings, or TEST examples are loaded.

Because H0-01 found no explicit session/day metadata, any successful result here
is **user-level / unseen-speaker personalization**, not cross-session
personalization.

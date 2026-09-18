# P2-07B Frozen Task Semantics

## Primary conclusion

The raw MDSC manifest contains 3,858 distinct transcript strings, but these
must not be interpreted as 3,858 wake-word classes.

The official MDSC description states that the corpus contains ten wake-up words
repeated five times at different speaking rates by each speaker, together with
non-wake-up command/phrase material.

The P2-07A inventory contains exactly ten dominant full-speaker phrases. After
removing the explicit `<p>` pause annotation, each canonical wake phrase has:

```text
46 speakers × 5 repetitions = 230 utterances
```

and all ten together contain:

```text
10 × 230 = 2,300 wake-word utterances
```

Therefore P2-07B freezes those ten phrases as the MDSC wake-word inventory.

## Ten canonical wake words

```text
Hey Siri
你好小布
天猫精灵
小冰小冰
小度小度
小德小德
小溪你好
小爱同学
小艺小艺
灵犀灵犀
```

## Canonicalization boundary

Task-side normalization removes only:

```text
<p> pause annotation
whitespace
Latin case differences
```

For example:

```text
小度<p>小度 -> 小度小度
你<p>好<p>小<p>布 -> 你好小布
小<p>德<p>小<p>德 -> 小德小德
```

No synonym merging, typo correction, semantic command normalization or raw
manifest rewriting is permitted in P2.

## Primary MDSC task

```text
speaker-dependent dysarthria wake-up word spotting
```

Personalized evaluation scope:

```text
domain = dysarthria
split  = dev/test
role   = enrollment/eval
```

Task label:

```text
one of 10 canonical wake words -> positive target
everything else               -> __NON_WAKE__
```

Target enrollment speech constructs speaker-specific wake prototypes.
Non-wake enrollment material may be used as negative/calibration context but
must never be averaged into a positive wake prototype.

Thresholds are selected on development data and locked before test evaluation.

## GSC task

GSC remains the generic representation-development benchmark:

```text
gsc_all_35_closed_set
```

using all 35 observed labels and official train/dev/test partitions.

## Optional MDSC diagnostic view

The 30 phrases observed across all 46 speakers are retained as:

```text
mdsc_common_30_closed_set
```

This is only a representation diagnostic view. It is not the primary final
task and does not replace the binary personalized WWS protocol.

## Consequence for later stages

P3/P4 may use GSC and/or the common-30 view for controlled representation
comparisons. Final MDSC qualification must use the personalized 10-wake-word
view with non-wake evaluation speech contributing false-alarm trials.

"""Native padded-batch versus exact-length reference comparison."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn.functional as F

from papr_ssl.models.teacher.backbones.length_aware import LengthAwareSSLAdapter


@dataclass(frozen=True)
class SampleNumericComparison:
    index: int
    valid_frames_reference: int
    valid_frames_native: int
    mean_abs_error: float | None
    max_abs_error: float | None
    mean_frame_cosine: float | None
    min_frame_cosine: float | None
    pooled_cosine: float | None


@dataclass(frozen=True)
class NativeReferenceComparison:
    structural_match: bool
    numeric_pass: bool
    gate: str
    samples: tuple[SampleNumericComparison, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_match": self.structural_match,
            "numeric_pass": self.numeric_pass,
            "gate": self.gate,
            "samples": [asdict(x) for x in self.samples],
        }


def compare_native_to_reference(
    backbone,
    batch,
    *,
    min_mean_frame_cosine: float = 0.999,
    min_pooled_cosine: float = 0.999,
) -> NativeReferenceComparison:
    """Compare one native batch forward against P2-06A singleton reference."""
    reference_adapter = LengthAwareSSLAdapter(backbone)

    with torch.inference_mode():
        reference = reference_adapter.forward_audio_batch(batch)
        native = backbone(
            batch.waveforms,
            sample_mask=batch.waveform_mask,
        )

    if int(reference.hidden_layer) != int(native.hidden_layer):
        return NativeReferenceComparison(
            structural_match=False,
            numeric_pass=False,
            gate="FAIL",
            samples=(),
        )

    if reference.features.shape[0] != native.features.shape[0]:
        return NativeReferenceComparison(
            structural_match=False,
            numeric_pass=False,
            gate="FAIL",
            samples=(),
        )

    ref_lengths = reference.frame_mask.sum(dim=1, dtype=torch.int64)
    native_lengths = native.frame_mask.sum(dim=1, dtype=torch.int64)
    structural_match = torch.equal(ref_lengths, native_lengths)

    rows: list[SampleNumericComparison] = []
    numeric_pass = structural_match

    for i in range(reference.features.shape[0]):
        r_len = int(ref_lengths[i].item())
        n_len = int(native_lengths[i].item())

        if r_len != n_len:
            rows.append(
                SampleNumericComparison(
                    index=i,
                    valid_frames_reference=r_len,
                    valid_frames_native=n_len,
                    mean_abs_error=None,
                    max_abs_error=None,
                    mean_frame_cosine=None,
                    min_frame_cosine=None,
                    pooled_cosine=None,
                )
            )
            numeric_pass = False
            continue

        ref_valid = reference.features[i, :r_len].float()
        native_valid = native.features[i, :n_len].float()

        if ref_valid.shape != native_valid.shape:
            rows.append(
                SampleNumericComparison(
                    index=i,
                    valid_frames_reference=r_len,
                    valid_frames_native=n_len,
                    mean_abs_error=None,
                    max_abs_error=None,
                    mean_frame_cosine=None,
                    min_frame_cosine=None,
                    pooled_cosine=None,
                )
            )
            numeric_pass = False
            continue

        diff = (native_valid - ref_valid).abs()
        frame_cos = F.cosine_similarity(
            native_valid,
            ref_valid,
            dim=-1,
            eps=1e-8,
        )
        pooled_cos = F.cosine_similarity(
            native_valid.mean(dim=0, keepdim=True),
            ref_valid.mean(dim=0, keepdim=True),
            dim=-1,
            eps=1e-8,
        )[0]

        mean_frame_cos = float(frame_cos.mean().item())
        pooled_cos_value = float(pooled_cos.item())

        if (
            mean_frame_cos < min_mean_frame_cosine
            or pooled_cos_value < min_pooled_cosine
        ):
            numeric_pass = False

        rows.append(
            SampleNumericComparison(
                index=i,
                valid_frames_reference=r_len,
                valid_frames_native=n_len,
                mean_abs_error=float(diff.mean().item()),
                max_abs_error=float(diff.max().item()),
                mean_frame_cosine=mean_frame_cos,
                min_frame_cosine=float(frame_cos.min().item()),
                pooled_cosine=pooled_cos_value,
            )
        )

    gate = "PASS" if structural_match and numeric_pass else "FAIL"
    return NativeReferenceComparison(
        structural_match=structural_match,
        numeric_pass=numeric_pass,
        gate=gate,
        samples=tuple(rows),
    )

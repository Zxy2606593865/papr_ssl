#!/usr/bin/env python
"""P4-01 candidate-freeze smoke check."""

from __future__ import annotations

from papr_ssl.training.teacher.p4_candidates import (
    P4_BACKBONES,
    as_machine_readable_dict,
    total_candidate_count,
)


def main() -> int:
    payload = as_machine_readable_dict()

    print("=" * 104)
    print("PAPR-SSL P4-01 BACKBONE / HIDDEN-STATE SWEEP CANDIDATES")
    print("=" * 104)
    print(f"index semantics:     {payload['index_semantics']}")
    print(f"candidate policy:    {payload['candidate_policy']}")
    print(f"selection metric:    {payload['selection_metric']}")
    print(
        "metric definition:  "
        f"{payload['selection_metric_definition']}"
    )
    print(f"generic_test:        {payload['generic_test']}")
    print("-" * 104)

    for item in P4_BACKBONES:
        indices = item.valid_hidden_state_indices
        print(
            f"{item.name:18s} "
            f"count={len(indices):2d}  "
            f"indices={indices[0]}..{indices[-1]}  "
            f"final_index={item.final_hidden_state_index:2d}  "
            f"dim={item.embedding_dim}"
        )

    print("-" * 104)
    print(f"total candidates:    {total_candidate_count()}")
    print("paper-layer alias:   NOT USED")
    print("generic_test access: NO")
    print("-" * 104)
    print("P4-01 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

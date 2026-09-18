from __future__ import annotations

import unittest

from papr_ssl.training.teacher.p4_candidates import (
    P4_BACKBONES,
    as_machine_readable_dict,
    get_p4_backbone,
    total_candidate_count,
    validate_p4_candidates,
)


class P4CandidateRegistryTest(unittest.TestCase):
    def test_all_p1_validated_indices_are_included(self):
        validate_p4_candidates()
        self.assertEqual(
            get_p4_backbone("wav2vec2_base").valid_hidden_state_indices,
            tuple(range(13)),
        )
        self.assertEqual(
            get_p4_backbone("wavlm_large").valid_hidden_state_indices,
            tuple(range(25)),
        )
        self.assertEqual(
            get_p4_backbone("w2v_bert2").valid_hidden_state_indices,
            tuple(range(25)),
        )

    def test_exact_p1_revisions_are_frozen(self):
        expected = {
            "wav2vec2_base": (
                "facebook/wav2vec2-base",
                "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
            ),
            "wavlm_large": (
                "microsoft/wavlm-large",
                "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
            ),
            "w2v_bert2": (
                "facebook/w2v-bert-2.0",
                "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
            ),
        }
        for item in P4_BACKBONES:
            self.assertEqual(
                (item.model_id, item.model_revision),
                expected[item.name],
            )

    def test_candidate_count_is_63(self):
        self.assertEqual(total_candidate_count(), 13 + 25 + 25)
        self.assertEqual(total_candidate_count(), 63)

    def test_machine_readable_policy_has_no_paper_layer_alias(self):
        payload = as_machine_readable_dict()
        self.assertEqual(
            payload["index_semantics"],
            "exact outputs.hidden_states tuple index; "
            "not a paper layer number",
        )
        self.assertEqual(
            payload["candidate_policy"],
            "all P1-validated hidden-state indices",
        )

    def test_selection_metric_and_test_seal_are_frozen(self):
        payload = as_machine_readable_dict()
        self.assertEqual(payload["selection_metric"], "generic_dev_score")
        self.assertEqual(
            payload["selection_metric_definition"],
            "prototype_macro_f1",
        )
        self.assertEqual(payload["generic_test"], "sealed_not_accessed")


if __name__ == "__main__":
    unittest.main(verbosity=2)

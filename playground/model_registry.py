"""Explicit experiment metadata shared by accuracy and inference tooling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    config: str
    label: str
    backbone: str
    scorer: str
    exact_top_k: str
    inference_adapter: str
    supervision_mode: str = "all_positions"


MODEL_SPECS = {
    "sasrec_lcar": ModelSpec("sasrec-lcar.gin", "LCAR-SASRec", "SASRec",
                             "LCAR", "LearnedBruteForceTopK", "lcar", "full_prefix"),
    "hstu_lcar": ModelSpec("hstu-lcar.gin", "LCAR-HSTU", "HSTU",
                           "LCAR", "LearnedBruteForceTopK", "lcar", "full_prefix"),
    "sasrec": ModelSpec("sasrec-sampled-softmax-n128-final.gin", "SASRec", "SASRec",
                         "DotProduct", "MIPSBruteForceTopK", "dot_product"),
    "sasrec_mol": ModelSpec("sasrec-mol-sampled-softmax-n128-8x4x64-rails-final.gin", "SASRec + MoL", "SASRec",
                             "MoL", "MoLBruteForceTopK", "mol"),
    "hstu": ModelSpec("hstu-sampled-softmax-n128-rails-final.gin", "HSTU", "HSTU",
                       "DotProduct", "MIPSBruteForceTopK", "dot_product"),
    "hstu_mol": ModelSpec("hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin", "HSTU + MoL", "HSTU",
                           "MoL", "MoLBruteForceTopK", "mol"),

}
BASELINE_MODELS = tuple(name for name, spec in MODEL_SPECS.items() if spec.scorer != "LCAR")
DEFAULT_MODELS=BASELINE_MODELS+('sasrec_lcar','hstu_lcar')
MODELS = {name: spec.config for name, spec in MODEL_SPECS.items()}
LABELS = {name: spec.label for name, spec in MODEL_SPECS.items()}

ABLATIONS = {
    "full": "",
    "no_global": "create_lcar_interaction_module.use_global_token = False\n",
    "no_source_embeddings": "create_lcar_interaction_module.use_source_embeddings = False\n",
    "raw_only": "create_lcar_interaction_module.use_summary_tokens = False\n",
    "recent_only": ("create_lcar_interaction_module.use_selected_tokens = False\n"
                    "create_lcar_interaction_module.use_summary_tokens = False\n"),
    "no_recent": "create_lcar_interaction_module.recent_events = 0\n",
    "one_expert": "create_lcar_interaction_module.num_item_experts = 1\n",
    "full_history": "create_lcar_interaction_module.events_per_slot = 200\n",
}

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
                             "LCAR", "LearnedBruteForceTopK", "lcar", "sampled_prefix"),
    "hstu_lcar": ModelSpec("hstu-lcar.gin", "LCAR-HSTU", "HSTU",
                           "LCAR", "LearnedBruteForceTopK", "lcar", "sampled_prefix"),
    "sasrec": ModelSpec("sasrec-sampled-softmax-n128-final.gin", "SASRec", "SASRec",
                         "DotProduct", "MIPSBruteForceTopK", "dot_product"),
    "sasrec_mol": ModelSpec("sasrec-mol-sampled-softmax-n128-8x4x64-rails-final.gin", "SASRec + MoL", "SASRec",
                             "MoL", "MoLBruteForceTopK", "mol"),
    "hstu": ModelSpec("hstu-sampled-softmax-n128-rails-final.gin", "HSTU", "HSTU",
                       "DotProduct", "MIPSBruteForceTopK", "dot_product"),
    "hstu_mol": ModelSpec("hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin", "HSTU + MoL", "HSTU",
                           "MoL", "MoLBruteForceTopK", "mol"),
    "sasrec_expert_history": ModelSpec("sasrec-expert-history.gin", "SASRec + ExpertHistory", "SASRec",
                                        "ExpertHistory", "LearnedBruteForceTopK", "expert_history", "sampled_prefix"),
    "hstu_expert_history": ModelSpec("hstu-expert-history.gin", "HSTU + ExpertHistory", "HSTU",
                                      "ExpertHistory", "LearnedBruteForceTopK", "expert_history", "sampled_prefix"),
}
BASELINE_MODELS = tuple(name for name, spec in MODEL_SPECS.items() if spec.scorer not in ("ExpertHistory","LCAR"))
DEFAULT_MODELS=BASELINE_MODELS+('sasrec_lcar','hstu_lcar')
MODELS = {name: spec.config for name, spec in MODEL_SPECS.items()}
LABELS = {name: spec.label for name, spec in MODEL_SPECS.items()}

ABLATIONS = {
    "full": "",
    "raw_only": "create_expert_history_interaction_module.use_full_history_summaries = False\n",
    "uniform_summary": 'create_expert_history_interaction_module.activation_weighting = "uniform"\n',
    "ordinary_context": "create_expert_history_interaction_module.latent_conditioned_history = False\n",
    "recent_only": ("create_expert_history_interaction_module.use_full_history_summaries = False\n"
                    "create_expert_history_interaction_module.use_selected_attention = False\n"),
    "no_recent": "create_expert_history_interaction_module.recent_events = 0\n",
    "shared_memory": "create_expert_history_interaction_module.share_expert_memory = True\n",
    "one_expert": "create_expert_history_interaction_module.num_item_experts = 1\n",
    "full_history": "create_expert_history_interaction_module.events_per_slot = 200\n",
}

LCAR_ABLATIONS = {
    "full": "",
    "raw_only": "create_lcar_interaction_module.use_summary_tokens = False\n",
    "recent_only": ("create_lcar_interaction_module.use_selected_tokens = False\n"
                    "create_lcar_interaction_module.use_summary_tokens = False\n"),
    "no_recent": "create_lcar_interaction_module.recent_events = 0\n",
    "one_expert": "create_lcar_interaction_module.num_item_experts = 1\n",
    "full_history": "create_lcar_interaction_module.events_per_slot = 200\n",
}

REFERENCE_ONTOLOGIES = {

    # AD (Alzheimer's disease)
    "Amyloid": [
        "app_variant_or_mutation",
        "secretase_enzyme",
        "cleavage_site",
        "amyloidogenic_processing_marker",
        "abeta_species",
        "abeta_ratio_or_profile",
        "gamma_secretase_processivity",
        "app_trafficking_location",
        "post_translational_modification",
        "abeta_pathology_measure",
    ],

    "Tau": [
        "tau_species",
        "tau_seed_or_strain",
        "tau_release_route",
        "tau_uptake_mechanism",
        "tau_receptor_or_binding_factor",
        "synaptic_localization",
        "spread_direction_or_route",
        "brain_region_or_epicenter",
        "network_connectivity_measure",
        "clinical_or_pathological_outcome",
    ],

    "Neuroinflammation": [
        "microglial_activation_state",
        "m1_m2_polarization",
        "microglial_state_marker",
        "inflammatory_mediator",
        "immune_signaling_pathway",
        "microglial_receptor_or_target",
        "phagocytosis_or_clearance_function",
        "amyloid_or_tau_interaction",
        "disease_stage_or_progression",
        "microglial_transcriptional_state",
    ],

    # Chemistry
    "COF": [
        "cof_name",
        "building_blocks",
        "linkage_type",
        "dimensionality",
        "topology_or_network",
        "pore_structure",
        "synthesis_method",
        "crystallinity_characterization",
        "chemical_or_thermal_stability",
        "application_or_performance",
    ],

    "MOF": [
        "mof_name",
        "metal_node_or_sbu",
        "organic_ligand_or_linker",
        "framework_topology",
        "pore_structure",
        "surface_area_or_porosity",
        "synthesis_method",
        "microfine_design_strategy",
        "active_site_or_functional_group",
        "separation_or_catalytic_performance",
    ],

    "ZIF": [
        "zif_name",
        "metal_center",
        "imidazolate_linker",
        "zeolite_topology",
        "pore_aperture_or_cage_size",
        "surface_area_or_porosity",
        "synthesis_method",
        "mixed_linker_strategy",
        "chemical_or_thermal_stability",
        "gas_separation_performance",
    ],

    # Computer Science
    "Prompt": [
        "prompting_method",
        "prompting_setting",
        "task_instruction",
        "demonstration_examples",
        "exemplar_format",
        "reasoning_chain",
        "trigger_phrase",
        "answer_extraction_step",
        "target_task_type",
        "performance_metric",
    ],

    "LLM": [
        "agent_type",
        "agent_profile_or_role",
        "memory_mechanism",
        "planning_strategy",
        "reasoning_trace",
        "action_space_or_action_type",
        "observation_or_feedback",
        "reasoning_action_trajectory",
        "tool_or_api_used",
        "multi_agent_communication",
    ],

    "RLHF": [
        "rlhf_method",
        "base_model_or_policy",
        "feedback_type",
        "comparison_unit",
        "reward_model",
        "preference_modeling_objective",
        "policy_optimization_algorithm",
        "kl_regularization",
        "sft_or_demonstration_stage",
        "evaluation_metric",
    ],
}

def list_subdomains():
    return sorted(REFERENCE_ONTOLOGIES.keys())

def get_reference(subdomain: str):
    if subdomain not in REFERENCE_ONTOLOGIES:
        raise KeyError(
            f"Unknown subdomain {subdomain!r}. Available: {list_subdomains()}"
        )
    return list(REFERENCE_ONTOLOGIES[subdomain])

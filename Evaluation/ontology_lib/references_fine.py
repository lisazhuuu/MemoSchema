"""Fine-grained chemistry references reviewed by a domain expert.

Built from domain knowledge only: no induced schema, candidate pool or run output
was consulted, so it is independent of the pipeline being evaluated. Each fine
concept names its parent among the existing 10 coarse concepts
(references.REFERENCE_ONTOLOGIES), so coarse scores can be recovered by roll-up.
parent=None marks a concept with no coarse parent (added; not in the 10).
Used by Evaluation/evaluate_fine.py (hierarchical F1).
"""

def _c(name, parent, definition):
    return {"name": name, "parent": parent, "definition": definition}

FINE_REFERENCES = {
    "MOF": [
        _c("mof_name_or_code", "mof_name", "Name or code of the MOF studied (e.g. UiO-66, MOF-74, HKUST-1)."),
        _c("chemical_formula_or_composition", "mof_name", "Reported chemical formula or composition of the framework."),
        _c("metal_identity_and_oxidation_state", "metal_node_or_sbu", "Metal element(s) of the nodes and their oxidation state."),
        _c("sbu_or_cluster_type", "metal_node_or_sbu", "Secondary building unit / metal cluster (e.g. Zr6-oxo, Cu paddlewheel, rod SBU)."),
        _c("linker_identity", "organic_ligand_or_linker", "Organic linker used (e.g. BDC, BTC, DOBDC)."),
        _c("linker_functionalization", "organic_ligand_or_linker", "Functional groups on the linker (e.g. -NH2, -SO3H, -F)."),
        _c("net_topology", "framework_topology", "Network topology symbol (e.g. fcu, pcu, tbo)."),
        _c("dimensionality_or_interpenetration", "framework_topology", "Framework dimensionality (2D/3D) and interpenetration."),
        _c("pore_size_or_aperture", "pore_structure", "Pore diameter, window or aperture size."),
        _c("pore_volume", "pore_structure", "Total or micropore volume (cm3/g)."),
        _c("bet_surface_area", "surface_area_or_porosity", "BET (or Langmuir) specific surface area."),
        _c("porosity_measurement_conditions", "surface_area_or_porosity", "Probe gas and temperature of the sorption measurement (e.g. N2 77 K)."),
        _c("synthesis_route", "synthesis_method", "Synthesis route (solvothermal, microwave, mechanochemical, electrochemical, room temperature)."),
        _c("synthesis_conditions", "synthesis_method", "Temperature, time, solvent and reagent ratios of the synthesis."),
        _c("modulator_use", "synthesis_method", "Modulator used to control growth or defects (e.g. acetic acid, HCl)."),
        _c("activation_procedure", "synthesis_method", "Solvent exchange, heating or supercritical drying used to activate the pores."),
        _c("post_synthetic_modification", "microfine_design_strategy", "Post-synthetic modification, metalation or ligand exchange."),
        _c("defect_engineering", "microfine_design_strategy", "Deliberate missing-linker / missing-cluster defects."),
        _c("mixed_metal_or_mixed_linker", "microfine_design_strategy", "Multivariate frameworks with mixed metals or linkers."),
        _c("composite_or_hybrid_form", "microfine_design_strategy", "MOF composites, films, membranes or hybrids with other materials."),
        _c("open_metal_sites", "active_site_or_functional_group", "Coordinatively unsaturated metal sites acting as active/binding sites."),
        _c("adsorption_capacity", "separation_or_catalytic_performance", "Gas or adsorbate uptake with its conditions (pressure, temperature)."),
        _c("adsorption_selectivity", "separation_or_catalytic_performance", "Selectivity for a target species (e.g. CO2/N2, IAST)."),
        _c("isosteric_heat_of_adsorption", "separation_or_catalytic_performance", "Qst / binding enthalpy of the adsorbate."),
        _c("catalytic_activity", "separation_or_catalytic_performance", "Conversion, yield, TOF or selectivity of a catalysed reaction."),
        _c("reusability_or_cycling", "separation_or_catalytic_performance", "Performance retention over repeated cycles."),
        _c("chemical_thermal_water_stability", None, "Thermal decomposition temperature and stability in water, acid or base."),
        _c("structural_characterization_method", None, "Methods confirming structure and phase (PXRD, single-crystal XRD, TGA, SEM)."),
    ],
    "COF": [
        _c("cof_name_or_code", "cof_name", "Name or code of the COF (e.g. COF-5, TpPa-1)."),
        _c("node_monomer", "building_blocks", "Node/knot monomer (e.g. TAPB, TFP)."),
        _c("linker_monomer", "building_blocks", "Linker/edge monomer (e.g. terephthalaldehyde, PDA)."),
        _c("building_block_symmetry", "building_blocks", "Symmetry/connectivity of the monomers (C2, C3, C4, C6)."),
        _c("linkage_chemistry", "linkage_type", "Linkage formed (imine, boronate ester, beta-ketoenamine, triazine, sp2 C=C, hydrazone)."),
        _c("linkage_conversion_or_locking", "linkage_type", "Post-synthetic conversion or locking of the linkage."),
        _c("dimensionality", "dimensionality", "2D layered or 3D framework."),
        _c("layer_stacking_mode", "dimensionality", "Interlayer stacking of 2D COFs (AA eclipsed, AB staggered, slipped)."),
        _c("net_topology", "topology_or_network", "Network topology (e.g. hcb, sql, kgm, dia, pts)."),
        _c("pore_size", "pore_structure", "Pore diameter or pore size distribution."),
        _c("pore_wall_functionalization", "pore_structure", "Functional groups or metal sites decorating the pore walls."),
        _c("bet_surface_area", "pore_structure", "BET specific surface area."),
        _c("synthesis_route", "synthesis_method", "Route (solvothermal, ionothermal, mechanochemical, interfacial, microwave)."),
        _c("reaction_conditions", "synthesis_method", "Solvent mixture, catalyst (e.g. acetic acid), temperature and time."),
        _c("material_form_or_morphology", "synthesis_method", "Powder, film, membrane, nanosheet or particle morphology."),
        _c("pxrd_crystallinity", "crystallinity_characterization", "PXRD pattern / crystallinity evidence."),
        _c("structure_modeling_or_refinement", "crystallinity_characterization", "Simulated stacking models, Pawley/Rietveld refinement."),
        _c("linkage_spectroscopic_confirmation", "crystallinity_characterization", "FTIR / solid-state NMR evidence of linkage formation."),
        _c("thermal_stability", "chemical_or_thermal_stability", "TGA decomposition temperature."),
        _c("chemical_stability_conditions", "chemical_or_thermal_stability", "Stability in water, acid, base or solvents and duration."),
        _c("application_area", "application_or_performance", "Application (photocatalysis, adsorption, energy storage, sensing, separation)."),
        _c("performance_metric_value", "application_or_performance", "Key performance value (e.g. H2 evolution rate, uptake, capacity)."),
        _c("band_gap_or_optoelectronic_property", "application_or_performance", "Optical band gap, absorption edge or charge-transport property."),
        _c("reusability_or_cycling", "application_or_performance", "Performance retention over repeated cycles."),
    ],
    "ZIF": [
        _c("zif_name_or_code", "zif_name", "Name or code of the ZIF (e.g. ZIF-8, ZIF-67, ZIF-L)."),
        _c("metal_ion", "metal_center", "Tetrahedral metal ion (e.g. Zn2+, Co2+)."),
        _c("mixed_metal_composition", "metal_center", "Bimetallic / doped metal composition and ratio."),
        _c("imidazolate_linker_identity", "imidazolate_linker", "Imidazolate linker (e.g. 2-methylimidazole, benzimidazole)."),
        _c("linker_substituent", "imidazolate_linker", "Substituent/functional group on the imidazolate."),
        _c("topology_symbol", "zeolite_topology", "Zeolite-like topology (e.g. sod, rho, gme, lta)."),
        _c("aperture_size", "pore_aperture_or_cage_size", "Pore window / aperture size."),
        _c("cage_size", "pore_aperture_or_cage_size", "Cage or cavity diameter."),
        _c("framework_flexibility_or_gate_opening", "pore_aperture_or_cage_size", "Linker swing, gate opening or flexibility."),
        _c("bet_surface_area", "surface_area_or_porosity", "BET specific surface area."),
        _c("pore_volume", "surface_area_or_porosity", "Total or micropore volume."),
        _c("synthesis_route", "synthesis_method", "Route (solvothermal, aqueous room temperature, mechanochemical, seeded/secondary growth)."),
        _c("synthesis_conditions", "synthesis_method", "Metal:linker ratio, solvent, temperature and time."),
        _c("particle_size_or_morphology", "synthesis_method", "Crystal size and morphology (e.g. rhombic dodecahedra, nanoparticles)."),
        _c("membrane_or_film_fabrication", "synthesis_method", "Membrane/film support, fabrication method and thickness."),
        _c("mixed_linker_ratio", "mixed_linker_strategy", "Ratio of linkers in mixed-linker ZIFs."),
        _c("linker_exchange", "mixed_linker_strategy", "Post-synthetic or solvent-assisted linker exchange."),
        _c("thermal_stability", "chemical_or_thermal_stability", "Thermal decomposition temperature."),
        _c("chemical_or_hydrothermal_stability", "chemical_or_thermal_stability", "Stability in water, acid, base or steam."),
        _c("target_gas_pair", "gas_separation_performance", "Separated gas pair or target species (e.g. C3H6/C3H8, H2/CO2)."),
        _c("permeance_or_permeability", "gas_separation_performance", "Gas permeance or permeability."),
        _c("separation_selectivity", "gas_separation_performance", "Separation factor or ideal/IAST selectivity."),
        _c("adsorption_uptake", "gas_separation_performance", "Gas uptake with its conditions."),
        _c("zif_derived_material", None, "Carbon or oxide materials derived from ZIFs and their pyrolysis conditions."),
    ],
}

def get_fine_reference(subdomain: str):
    return [dict(x) for x in FINE_REFERENCES[subdomain]]

"""Layer 4: Workflows - Public entry points.

Workflows provide the public entry points users call. Workflows can import
I/O libraries and plotting libraries. Put file loading and saving here.
Put plotting here.
"""

from geosmith.utils.optional_imports import optional_import
from geosmith.workflows.drillhole import (
    compute_3d_coordinates,
    find_column,
    merge_collar_assay,
    process_drillhole_data,
)

# Optional DLIS support (requires dlisio)
DLIS_AVAILABLE, _dlis = optional_import(
    "geosmith.workflows.dlis", ["DlisParser", "read_dlis_file"]
)
DlisParser = _dlis["DlisParser"]  # type: ignore
read_dlis_file = _dlis["read_dlis_file"]  # type: ignore

# Optional RESQML support (requires resqpy)
RESQML_AVAILABLE, _resqml = optional_import(
    "geosmith.workflows.resqml",
    ["read_resqml_grid", "read_resqml_properties", "ResqmlParser"],
)
read_resqml_grid = _resqml["read_resqml_grid"]  # type: ignore
read_resqml_properties = _resqml["read_resqml_properties"]  # type: ignore
ResqmlParser = _resqml["ResqmlParser"]  # type: ignore

# Optional WITSML support (requires xml.etree.ElementTree, typically available)
WITSML_AVAILABLE, _witsml = optional_import(
    "geosmith.workflows.witsml",
    ["read_witsml_log", "read_witsml_trajectory", "read_witsml_well", "WitsmlParser"],
)
read_witsml_log = _witsml["read_witsml_log"]  # type: ignore
read_witsml_trajectory = _witsml["read_witsml_trajectory"]  # type: ignore
read_witsml_well = _witsml["read_witsml_well"]  # type: ignore
WitsmlParser = _witsml["WitsmlParser"]  # type: ignore

# Optional PPDM support
PPDM_AVAILABLE, _ppdm = optional_import(
    "geosmith.workflows.ppdm", ["PpdmDataModel", "PpdmParser", "read_ppdm_csv"]
)
PpdmDataModel = _ppdm["PpdmDataModel"]  # type: ignore
PpdmParser = _ppdm["PpdmParser"]  # type: ignore
read_ppdm_csv = _ppdm["read_ppdm_csv"]  # type: ignore
from geosmith.workflows.grdecl import (
    export_block_model,
    read_grdecl,
    write_grdecl,
)
from geosmith.workflows.io import (
    load_csv_from_string,
    read_raster,
    read_vector,
    write_raster,
    write_vector,
)
from geosmith.workflows.las import read_las
from geosmith.workflows.segy import (
    read_segy_summary,
    read_segy_traces,
    SegySummary,
    TraceHeader,
)

# Optional NLP workflows (requires spacy or transformers)
NLP_AVAILABLE, _nlp = optional_import(
    "geosmith.workflows.nlp",
    [
        "extract_entities_from_file",
        "load_entity_list",
        "load_text_documents",
        "train_custom_ner_model",
    ],
)
extract_entities_from_file = _nlp["extract_entities_from_file"]  # type: ignore
load_entity_list = _nlp["load_entity_list"]  # type: ignore
load_text_documents = _nlp["load_text_documents"]  # type: ignore
train_custom_ner_model = _nlp["train_custom_ner_model"]  # type: ignore
from geosmith.workflows.geostatistics import (
    GeostatisticalModel,
    GeostatisticalResult,
)
from geosmith.workflows.gslib import (
    export_block_model_gslib,
    read_gslib,
    write_gslib,
)
from geosmith.workflows.workflows import (
    make_features,
    process_raster,
    reproject_to,
    zonal_stats,
)

# Optional plotting (requires matplotlib)
PLOTTING_AVAILABLE, _plotting = optional_import(
    "geosmith.workflows.plotting",
    [
        "add_facies_track",
        "add_log_track",
        "buckles_plot",
        "create_3d_well_log_viewer",
        "create_combined_map",
        "create_cross_section_viewer",
        "create_facies_log_plot",
        "create_field_map",
        "create_interactive_kriging_map",
        "create_interactive_well_map",
        "create_multi_well_3d_viewer",
        "create_multi_well_strip_chart",
        "create_strip_chart",
        "create_well_trajectory_map",
        "mineral_composition_plot",
        "neutron_density_crossplot",
        "pickett_plot",
        "plot_mud_weight_profile",
        "plot_pressure_profile",
        "qfl_plot",
        "sand_silt_clay_plot",
        "ternary_plot",
    ],
)
add_facies_track = _plotting["add_facies_track"]  # type: ignore
add_log_track = _plotting["add_log_track"]  # type: ignore
buckles_plot = _plotting["buckles_plot"]  # type: ignore
create_3d_well_log_viewer = _plotting["create_3d_well_log_viewer"]  # type: ignore
create_combined_map = _plotting["create_combined_map"]  # type: ignore
create_cross_section_viewer = _plotting["create_cross_section_viewer"]  # type: ignore
create_facies_log_plot = _plotting["create_facies_log_plot"]  # type: ignore
create_field_map = _plotting["create_field_map"]  # type: ignore
create_interactive_kriging_map = _plotting["create_interactive_kriging_map"]  # type: ignore
create_interactive_well_map = _plotting["create_interactive_well_map"]  # type: ignore
create_multi_well_3d_viewer = _plotting["create_multi_well_3d_viewer"]  # type: ignore
create_multi_well_strip_chart = _plotting["create_multi_well_strip_chart"]  # type: ignore
create_strip_chart = _plotting["create_strip_chart"]  # type: ignore
create_well_trajectory_map = _plotting["create_well_trajectory_map"]  # type: ignore
mineral_composition_plot = _plotting["mineral_composition_plot"]  # type: ignore
neutron_density_crossplot = _plotting["neutron_density_crossplot"]  # type: ignore
pickett_plot = _plotting["pickett_plot"]  # type: ignore
plot_mud_weight_profile = _plotting["plot_mud_weight_profile"]  # type: ignore
plot_pressure_profile = _plotting["plot_pressure_profile"]  # type: ignore
qfl_plot = _plotting["qfl_plot"]  # type: ignore
sand_silt_clay_plot = _plotting["sand_silt_clay_plot"]  # type: ignore
ternary_plot = _plotting["ternary_plot"]  # type: ignore

__all__ = [
    "buckles_plot",
    "compute_3d_coordinates",
    "create_3d_well_log_viewer",
    "create_combined_map",
    "create_cross_section_viewer",
    "create_facies_log_plot",
    "create_field_map",
    "create_interactive_kriging_map",
    "create_interactive_well_map",
    "create_multi_well_3d_viewer",
    "create_multi_well_strip_chart",
    "create_strip_chart",
    "create_well_trajectory_map",
    "DlisParser",
    "export_block_model",
    "find_column",
    "load_csv_from_string",
    "make_features",
    "merge_collar_assay",
    "mineral_composition_plot",
    "neutron_density_crossplot",
    "pickett_plot",
    "plot_mud_weight_profile",
    "plot_pressure_profile",
    "PpdmDataModel",
    "PpdmParser",
    "process_drillhole_data",
    "process_raster",
    "qfl_plot",
    "read_dlis_file",
    "read_grdecl",
    "read_las",
    "read_ppdm_csv",
    "read_raster",
    "read_resqml_grid",
    "read_resqml_properties",
    "read_segy_summary",
    "read_segy_traces",
    "read_vector",
    "read_witsml_log",
    "read_witsml_trajectory",
    "read_witsml_well",
    "reproject_to",
    "ResqmlParser",
    "sand_silt_clay_plot",
    "SegySummary",
    "ternary_plot",
    "TraceHeader",
    "WitsmlParser",
    "write_grdecl",
    "write_raster",
    "write_vector",
    "zonal_stats",
]

# Conditionally add I/O exports if available
if DLIS_AVAILABLE:
    __all__.extend(["DlisParser", "read_dlis_file"])
if RESQML_AVAILABLE:
    __all__.extend(["ResqmlParser", "read_resqml_grid", "read_resqml_properties"])
if WITSML_AVAILABLE:
    __all__.extend(
        [
            "WitsmlParser",
            "read_witsml_well",
            "read_witsml_log",
            "read_witsml_trajectory",
        ]
    )
if PPDM_AVAILABLE:
    __all__.extend(["PpdmDataModel", "PpdmParser", "read_ppdm_csv"])

# Conditionally add NLP exports if available
if NLP_AVAILABLE:
    __all__.extend(
        [
            "extract_entities_from_file",
            "load_entity_list",
            "load_text_documents",
            "train_custom_ner_model",
        ]
    )

# Conditionally add plotting exports if available
if PLOTTING_AVAILABLE:
    # Already in __all__, but we can add them conditionally if needed
    pass

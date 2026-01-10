"""Layer 4: Workflows - Public entry points.

Workflows provide the public entry points users call. Workflows can import
I/O libraries and plotting libraries. Put file loading and saving here.
Put plotting here.
"""

from geosmith.workflows.drillhole import (
    compute_3d_coordinates,
    find_column,
    merge_collar_assay,
    process_drillhole_data,
)
# Optional DLIS support (requires dlisio)
try:
    from geosmith.workflows.dlis import DlisParser, read_dlis_file

    DLIS_AVAILABLE = True
except ImportError:
    DLIS_AVAILABLE = False
    DlisParser = None  # type: ignore
    read_dlis_file = None  # type: ignore
# Optional RESQML support (requires resqpy)
try:
    from geosmith.workflows.resqml import (
        read_resqml_grid,
        read_resqml_properties,
        ResqmlParser,
    )

    RESQML_AVAILABLE = True
except ImportError:
    RESQML_AVAILABLE = False
    ResqmlParser = None  # type: ignore
    read_resqml_grid = None  # type: ignore
    read_resqml_properties = None  # type: ignore
# Optional WITSML support (requires xml.etree.ElementTree, typically available)
try:
    from geosmith.workflows.witsml import (
        read_witsml_log,
        read_witsml_trajectory,
        read_witsml_well,
        WitsmlParser,
    )

    WITSML_AVAILABLE = True
except ImportError:
    WITSML_AVAILABLE = False
    WitsmlParser = None  # type: ignore
    read_witsml_well = None  # type: ignore
    read_witsml_log = None  # type: ignore
    read_witsml_trajectory = None  # type: ignore
# Optional PPDM support
try:
    from geosmith.workflows.ppdm import PpdmDataModel, PpdmParser, read_ppdm_csv

    PPDM_AVAILABLE = True
except ImportError:
    PPDM_AVAILABLE = False
    PpdmDataModel = None  # type: ignore
    PpdmParser = None  # type: ignore
    read_ppdm_csv = None  # type: ignore
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
try:
    from geosmith.workflows.nlp import (
        extract_entities_from_file,
        load_entity_list,
        load_text_documents,
        train_custom_ner_model,
    )

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    extract_entities_from_file = None  # type: ignore
    load_entity_list = None  # type: ignore
    load_text_documents = None  # type: ignore
    train_custom_ner_model = None  # type: ignore
from geosmith.workflows.workflows import (
    make_features,
    process_raster,
    reproject_to,
    zonal_stats,
)

# Optional plotting (requires matplotlib)
try:
    from geosmith.workflows.plotting import (
        add_facies_track,
        add_log_track,
        buckles_plot,
        create_3d_well_log_viewer,
        create_combined_map,
        create_cross_section_viewer,
        create_facies_log_plot,
        create_field_map,
        create_interactive_kriging_map,
        create_interactive_well_map,
        create_multi_well_3d_viewer,
        create_multi_well_strip_chart,
        create_strip_chart,
        create_well_trajectory_map,
        mineral_composition_plot,
        neutron_density_crossplot,
        pickett_plot,
        plot_mud_weight_profile,
        plot_pressure_profile,
        qfl_plot,
        sand_silt_clay_plot,
        ternary_plot,
    )

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    add_facies_track = None  # type: ignore
    add_log_track = None  # type: ignore
    buckles_plot = None  # type: ignore
    create_3d_well_log_viewer = None  # type: ignore
    create_combined_map = None  # type: ignore
    create_cross_section_viewer = None  # type: ignore
    create_facies_log_plot = None  # type: ignore
    create_field_map = None  # type: ignore
    create_interactive_kriging_map = None  # type: ignore
    create_interactive_well_map = None  # type: ignore
    create_multi_well_3d_viewer = None  # type: ignore
    create_multi_well_strip_chart = None  # type: ignore
    create_strip_chart = None  # type: ignore
    create_well_trajectory_map = None  # type: ignore
    mineral_composition_plot = None  # type: ignore
    neutron_density_crossplot = None  # type: ignore
    pickett_plot = None  # type: ignore
    plot_mud_weight_profile = None  # type: ignore
    plot_pressure_profile = None  # type: ignore
    qfl_plot = None  # type: ignore
    sand_silt_clay_plot = None  # type: ignore
    ternary_plot = None  # type: ignore

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
        ["WitsmlParser", "read_witsml_well", "read_witsml_log", "read_witsml_trajectory"]
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


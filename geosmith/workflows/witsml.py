"""WITSML v2.0 XML Parser for GeoSmith.

Migrated from geosuite.io.witsml_parser.
Layer 4: Workflows - I/O operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import xml.etree.ElementTree as ET

    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False
    ET = None  # type: ignore

logger = logging.getLogger(__name__)

# WITSML v2.0 namespaces
WITSML_NS = {
    "witsml": "http://www.energistics.org/energyml/data/witsmlv2",
    "eml": "http://www.energistics.org/energyml/data/commonv2",
}


class WitsmlParser:
    """WITSML XML parser for well data integration."""

    def __init__(self):
        """Initialize WITSML parser."""
        if not XML_AVAILABLE:
            raise ImportError(
                "xml.etree.ElementTree is required for WITSML support. "
                "This is typically available in Python standard library."
            )
        self.namespaces = WITSML_NS

    def _get_element_text(self, element: "ET.Element", xpath: str) -> Optional[str]:
        """Safely extract text from XML element using xpath."""
        found = element.find(xpath, self.namespaces)
        return found.text if found is not None else None

    def _get_element_attr(
        self, element: "ET.Element", xpath: str, attr: str
    ) -> Optional[str]:
        """Safely extract attribute from XML element."""
        found = element.find(xpath, self.namespaces)
        return found.get(attr) if found is not None else None

    def _clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean and convert numeric values."""
        if value is None:
            return None
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None

    def parse_well_header(self, xml_file: Union[str, Path]) -> Dict[str, Any]:
        """Parse Well header information from WITSML XML.

        Args:
            xml_file: Path to WITSML Well XML file.

        Returns:
            Dictionary with well header information:
                - 'uid': Well UID
                - 'name': Well name
                - 'field': Field name
                - 'country': Country
                - 'operator': Operator name
                - 'status_well': Well status
                - 'purpose_well': Well purpose
                - 'spud_date': Spud date
                - 'surface_location': Surface location data
                - 'bottom_hole_location': Bottom hole location data

        Example:
            >>> from geosmith.workflows.witsml import WitsmlParser
            >>>
            >>> parser = WitsmlParser()
            >>> well_header = parser.parse_well_header('well.xml')
            >>> print(f"Well: {well_header['name']}, Field: {well_header['field']}")
        """
        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"WITSML file not found: {xml_file}")

        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            well_data = {
                "uid": root.get("uuid"),
                "name": self._get_element_text(root, ".//eml:Citation/eml:Title"),
                "field": self._get_element_text(root, ".//witsml:Field"),
                "country": self._get_element_text(root, ".//witsml:Country"),
                "state": self._get_element_text(root, ".//witsml:State"),
                "county": self._get_element_text(root, ".//witsml:County"),
                "operator": self._get_element_text(root, ".//witsml:Operator"),
                "status_well": self._get_element_text(root, ".//witsml:StatusWell"),
                "purpose_well": self._get_element_text(root, ".//witsml:PurposeWell"),
                "spud_date": self._get_element_text(root, ".//witsml:DTimSpud"),
                "license_number": self._get_element_text(root, ".//witsml:NumLicense"),
                "api_number": self._get_element_text(root, ".//witsml:NumGovt"),
                "surface_location": self._extract_location_data(root, "Surface"),
                "bottom_hole_location": self._extract_location_data(root, "BottomHole"),
            }

            return well_data

        except Exception as e:
            raise Exception(f"Error parsing WITSML Well file {xml_file}: {str(e)}")

    def parse_wellbore_data(self, xml_file: Union[str, Path]) -> Dict[str, Any]:
        """Parse Wellbore information from WITSML XML.

        Args:
            xml_file: Path to WITSML Wellbore XML file.

        Returns:
            Dictionary with wellbore information.

        Example:
            >>> from geosmith.workflows.witsml import WitsmlParser
            >>>
            >>> parser = WitsmlParser()
            >>> wellbore = parser.parse_wellbore_data('wellbore.xml')
            >>> print(f"Wellbore: {wellbore['name']}, MD: {wellbore['md_current']} m")
        """
        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"WITSML file not found: {xml_file}")

        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            wellbore_data = {
                "uid": root.get("uuid"),
                "name": self._get_element_text(root, ".//eml:Citation/eml:Title"),
                "well_uid": self._get_element_text(root, ".//witsml:Well"),
                "number": self._get_element_text(root, ".//witsml:Number"),
                "suffix_api": self._get_element_text(root, ".//witsml:SuffixAPI"),
                "num_govt": self._get_element_text(root, ".//witsml:NumGovt"),
                "status_wellbore": self._get_element_text(
                    root, ".//witsml:StatusWellbore"
                ),
                "purpose_wellbore": self._get_element_text(
                    root, ".//witsml:PurposeWellbore"
                ),
                "type_wellbore": self._get_element_text(root, ".//witsml:TypeWellbore"),
                "shape": self._get_element_text(root, ".//witsml:Shape"),
                "dtime_kick_off": self._get_element_text(
                    root, ".//witsml:DTimeKickOff"
                ),
                "md_planned": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:MdPlanned")
                ),
                "tvd_planned": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:TvdPlanned")
                ),
                "md_subseabed": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:MdSubSeaBed")
                ),
                "tvd_subseabed": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:TvdSubSeaBed")
                ),
                "md_current": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:MdCurrent")
                ),
                "tvd_current": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:TvdCurrent")
                ),
            }

            return wellbore_data

        except Exception as e:
            raise Exception(f"Error parsing WITSML Wellbore file {xml_file}: {str(e)}")

    def parse_log_data(self, xml_file: Union[str, Path]) -> pd.DataFrame:
        """Parse Log data from WITSML XML and return pandas DataFrame.

        Args:
            xml_file: Path to WITSML Log XML file.

        Returns:
            DataFrame with log data, with channels as columns.
            Metadata stored in DataFrame.attrs['log_info'] and .attrs['channels'].

        Example:
            >>> from geosmith.workflows.witsml import WitsmlParser
            >>>
            >>> parser = WitsmlParser()
            >>> log_df = parser.parse_log_data('log.xml')
            >>> print(f"Log channels: {list(log_df.columns)}")
            >>> print(
            ...     f"Log depth range: {log_df.index.min():.1f} - "
            ...     f"{log_df.index.max():.1f} m"
            ... )
        """
        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"WITSML file not found: {xml_file}")

        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            # Extract log header information
            log_info = {
                "uid": root.get("uuid"),
                "name": self._get_element_text(root, ".//eml:Citation/eml:Title"),
                "wellbore_uid": self._get_element_text(root, ".//witsml:Wellbore"),
                "service_company": self._get_element_text(
                    root, ".//witsml:ServiceCompany"
                ),
                "run_number": self._get_element_text(root, ".//witsml:RunNumber"),
                "creation_date": self._get_element_text(
                    root, ".//eml:Citation/eml:Creation"
                ),
                "start_md": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:StartIndex")
                ),
                "end_md": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:EndIndex")
                ),
                "direction": self._get_element_text(root, ".//witsml:Direction"),
            }

            # Extract channel information
            channels = []
            channel_elements = root.findall(".//witsml:Channel", self.namespaces)

            for channel in channel_elements:
                channel_info = {
                    "mnemonic": self._get_element_text(channel, ".//witsml:Mnemonic"),
                    "unit": self._get_element_attr(channel, ".//witsml:Index", "uom")
                    or self._get_element_attr(channel, ".//witsml:Value", "uom"),
                    "channel_class": self._get_element_text(
                        channel, ".//witsml:ChannelClass"
                    ),
                    "description": self._get_element_text(
                        channel, ".//eml:Citation/eml:Description"
                    ),
                    "data_type": self._get_element_text(channel, ".//witsml:DataType"),
                    "null_value": self._get_element_text(
                        channel, ".//witsml:NullValue"
                    ),
                }
                channels.append(channel_info)

            # Extract log data
            log_data_elements = root.findall(
                ".//witsml:LogData/witsml:Data", self.namespaces
            )

            if not log_data_elements:
                return pd.DataFrame()

            # Parse the actual log data
            data_rows = []
            for data_element in log_data_elements:
                data_text = data_element.text
                if data_text:
                    # Split comma-separated values
                    values = [v.strip() for v in data_text.split(",")]
                    data_rows.append(values)

            # Create column names from channel mnemonics
            column_names = [ch["mnemonic"] for ch in channels]

            # Create DataFrame
            if data_rows and column_names:
                df = pd.DataFrame(data_rows, columns=column_names[: len(data_rows[0])])

                # Convert numeric columns
                for col in df.columns:
                    if col.upper() in ["DEPT", "DEPTH", "MD", "TVD"]:  # Depth columns
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        # Try to convert other columns to numeric
                        df[col] = pd.to_numeric(df[col], errors="ignore")

                # Add metadata as attributes
                df.attrs["log_info"] = log_info
                df.attrs["channels"] = channels

                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            raise Exception(f"Error parsing WITSML Log file {xml_file}: {str(e)}")

    def parse_trajectory_data(self, xml_file: Union[str, Path]) -> pd.DataFrame:
        """Parse Trajectory data from WITSML XML.

        Args:
            xml_file: Path to WITSML Trajectory XML file.

        Returns:
            DataFrame with trajectory stations (MD, TVD, incl, azi, etc.).
            Metadata stored in DataFrame.attrs['trajectory_info'].

        Example:
            >>> from geosmith.workflows.witsml import WitsmlParser
            >>>
            >>> parser = WitsmlParser()
            >>> traj_df = parser.parse_trajectory_data('trajectory.xml')
            >>> print(f"Trajectory stations: {len(traj_df)}")
            >>> print(
            ...     f"MD range: {traj_df['md'].min():.1f} - "
            ...     f"{traj_df['md'].max():.1f} m"
            ... )
        """
        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"WITSML file not found: {xml_file}")

        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            # Extract trajectory header
            trajectory_info = {
                "uid": root.get("uuid"),
                "name": self._get_element_text(root, ".//eml:Citation/eml:Title"),
                "wellbore_uid": self._get_element_text(root, ".//witsml:Wellbore"),
                "service_company": self._get_element_text(
                    root, ".//witsml:ServiceCompany"
                ),
                "magnetic_declination": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:MagDeclination")
                ),
                "grid_correction": self._clean_numeric_value(
                    self._get_element_text(root, ".//witsml:GridCorrection")
                ),
            }

            # Extract trajectory stations
            stations = []
            station_elements = root.findall(
                ".//witsml:TrajectoryStation", self.namespaces
            )

            for station in station_elements:
                station_data = {
                    "md": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Md")
                    ),
                    "tvd": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Tvd")
                    ),
                    "incl": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Incl")
                    ),
                    "azi": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Azi")
                    ),
                    "north": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Dispns")
                    ),
                    "east": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Dispew")
                    ),
                    "vs": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Vs")
                    ),
                    "dls": self._clean_numeric_value(
                        self._get_element_text(station, ".//witsml:Dls")
                    ),
                    "type_survey": self._get_element_text(
                        station, ".//witsml:TypeSurveyTool"
                    ),
                }
                stations.append(station_data)

            # Create DataFrame
            df = pd.DataFrame(stations)
            df.attrs["trajectory_info"] = trajectory_info

            return df

        except Exception as e:
            raise Exception(
                f"Error parsing WITSML Trajectory file {xml_file}: {str(e)}"
            )

    def _extract_location_data(
        self, root: "ET.Element", location_type: str
    ) -> Dict[str, Any]:
        """Extract location data (surface or bottom hole)."""
        location_path = f".//witsml:WellLocation/witsml:{location_type}Location"
        location_element = root.find(location_path, self.namespaces)

        if location_element is None:
            return {}

        return {
            "latitude": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:Latitude")
            ),
            "longitude": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:Longitude")
            ),
            "projected_x": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:ProjectedX")
            ),
            "projected_y": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:ProjectedY")
            ),
            "local_x": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:LocalX")
            ),
            "local_y": self._clean_numeric_value(
                self._get_element_text(location_element, ".//witsml:LocalY")
            ),
            "original_text": self._get_element_text(
                location_element, ".//witsml:Original"
            ),
        }


def read_witsml_well(xml_file: Union[str, Path]) -> Dict[str, Any]:
    """Read WITSML Well header from XML file.

    Convenience function for parsing well headers.

    Args:
        xml_file: Path to WITSML Well XML file.

    Returns:
        Dictionary with well header information.

    Example:
        >>> from geosmith.workflows.witsml import read_witsml_well
        >>>
        >>> well = read_witsml_well('well.xml')
        >>> print(f"Well: {well['name']}, Operator: {well['operator']}")
    """
    parser = WitsmlParser()
    return parser.parse_well_header(xml_file)


def read_witsml_log(xml_file: Union[str, Path]) -> pd.DataFrame:
    """Read WITSML Log data from XML file.

    Convenience function for parsing log data.

    Args:
        xml_file: Path to WITSML Log XML file.

    Returns:
        DataFrame with log data.

    Example:
        >>> from geosmith.workflows.witsml import read_witsml_log
        >>>
        >>> log_df = read_witsml_log('log.xml')
        >>> print(f"Channels: {list(log_df.columns)}")
    """
    parser = WitsmlParser()
    return parser.parse_log_data(xml_file)


def read_witsml_trajectory(xml_file: Union[str, Path]) -> pd.DataFrame:
    """Read WITSML Trajectory data from XML file.

    Convenience function for parsing trajectory data.

    Args:
        xml_file: Path to WITSML Trajectory XML file.

    Returns:
        DataFrame with trajectory stations.

    Example:
        >>> from geosmith.workflows.witsml import read_witsml_trajectory
        >>>
        >>> traj_df = read_witsml_trajectory('trajectory.xml')
        >>> print(f"Trajectory stations: {len(traj_df)}")
    """
    parser = WitsmlParser()
    return parser.parse_trajectory_data(xml_file)

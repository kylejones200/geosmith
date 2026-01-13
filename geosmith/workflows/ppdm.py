"""PPDM (Professional Petroleum Data Management) Integration for GeoSmith.

Migrated from geosuite.io.ppdm_parser.
Layer 4: Workflows - I/O operations.
"""

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PpdmDataModel:
    """PPDM data model definitions and utilities."""

    # Core PPDM table structures
    CORE_TABLES = {
        "well": {
            "table_name": "ppdm_well",
            "primary_key": "uwi",
            "required_fields": [
                "uwi",
                "well_name",
                "operator",
                "surface_longitude",
                "surface_latitude",
            ],
            "field_mappings": {
                "UWI": "uwi",
                "ALT_WELL_NAME": "well_name",
                "OPERATOR": "operator",
                "SURFACE_LATITUDE": "surface_latitude",
                "SURFACE_LONGITUDE": "surface_longitude",
                "ASSIGNED_FIELD": "field_name",
                "CURRENT_STATUS": "well_status",
                "CURRENT_CLASS": "well_class",
                "COUNTRY": "country",
                "PROVINCE_STATE": "province_state",
                "PPDM_GUID": "ppdm_guid",
            },
        },
        "business_associate": {
            "table_name": "ppdm_business_associate",
            "primary_key": "business_associate_id",
            "required_fields": ["business_associate_id", "ba_name"],
            "field_mappings": {
                "BUSINESS_ASSOCIATE": "business_associate_id",
                "BA_NAME": "ba_name",
                "BA_TYPE": "ba_type",
                "BA_CATEGORY": "ba_category",
                "CURRENT_STATUS": "current_status",
                "PPDM_GUID": "ppdm_guid",
            },
        },
        "production": {
            "table_name": "ppdm_production",
            "primary_key": ["uwi", "production_date", "product_type"],
            "required_fields": ["uwi", "production_date", "product_type"],
            "field_mappings": {
                "UWI": "uwi",
                "PRODUCTION_DATE": "production_date",
                "PRODUCT_TYPE": "product_type",
                "DAILY_PROD_VOL": "daily_volume",
                "MONTHLY_PROD_VOL": "monthly_volume",
                "CUMULATIVE_PROD_VOL": "cumulative_volume",
                "PROD_METHOD": "production_method",
                "PPDM_GUID": "ppdm_guid",
            },
        },
    }

    @classmethod
    def get_table_definition(cls, table_type: str) -> Dict[str, Any]:
        """Get PPDM table definition.

        Args:
            table_type: Table type ('well', 'business_associate', 'production').

        Returns:
            Dictionary with table definition or empty dict if not found.
        """
        return cls.CORE_TABLES.get(table_type, {})

    @classmethod
    def validate_uwi(cls, uwi: str) -> bool:
        """Validate UWI format (basic validation).

        Args:
            uwi: Unique Well Identifier.

        Returns:
            True if UWI format is valid, False otherwise.
        """
        if not uwi or len(uwi) < 10:
            return False
        return uwi.replace("-", "").replace(".", "").isdigit()

    @classmethod
    def standardize_uwi(cls, uwi: str) -> str:
        """Standardize UWI format.

        Args:
            uwi: Unique Well Identifier.

        Returns:
            Standardized UWI (14-digit format).
        """
        if not uwi:
            return ""
        # Remove common separators and pad if needed
        clean_uwi = uwi.replace("-", "").replace(".", "").replace(" ", "")
        return clean_uwi.zfill(14)  # Standard 14-digit UWI


class PpdmParser:
    """Parser for PPDM data files and database structures."""

    def __init__(self, ppdm_directory: Optional[Union[str, Path]] = None):
        """Initialize PPDM parser.

        Args:
            ppdm_directory: Optional directory path for PPDM data files.
        """
        self.ppdm_directory = Path(ppdm_directory) if ppdm_directory else None
        self.data_model = PpdmDataModel()

    def parse_sql_schema(self, sql_file: Union[str, Path]) -> Dict[str, Any]:
        """Parse PPDM SQL schema file to extract table definitions.

        Args:
            sql_file: Path to PPDM SQL schema file.

        Returns:
            Dictionary mapping table names to table definitions.

        Example:
            >>> from geosmith.workflows.ppdm import PpdmParser
            >>>
            >>> parser = PpdmParser()
            >>> schema = parser.parse_sql_schema('ppdm_schema.sql')
            >>> print(f"Found {len(schema)} tables")
        """
        sql_file = Path(sql_file)
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL schema file not found: {sql_file}")

        try:
            with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract CREATE TABLE statements
            table_pattern = r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);"
            tables = {}

            matches = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)

            for table_name, definition in matches:
                # Parse column definitions
                columns = []

                for line in definition.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("--"):
                        col_match = re.match(r"(\w+)\s+([^,\n]+)", line)
                        if col_match:
                            col_name, col_type = col_match.groups()
                            col_type = col_type.strip().rstrip(",")
                            columns.append(
                                {
                                    "name": col_name,
                                    "type": col_type,
                                    "nullable": "NOT NULL" not in col_type,
                                }
                            )

                tables[table_name.lower()] = {"name": table_name, "columns": columns}

            return tables

        except Exception as e:
            raise Exception(f"Error parsing SQL schema {sql_file}: {str(e)}")

    def load_ppdm_csv(
        self, csv_file: Union[str, Path], data_type: str = "well"
    ) -> pd.DataFrame:
        """Load PPDM CSV data with proper formatting.

        Args:
            csv_file: Path to PPDM CSV file.
            data_type: Data type ('well', 'production', 'business_associate'),
                default 'well'.

        Returns:
            DataFrame with PPDM data, properly formatted and mapped.

        Example:
            >>> from geosmith.workflows.ppdm import PpdmParser
            >>>
            >>> parser = PpdmParser()
            >>> well_df = parser.load_ppdm_csv('wells.csv', data_type='well')
            >>> print(f"Loaded {len(well_df)} wells")
        """
        csv_file = Path(csv_file)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        try:
            # Load CSV with error handling for large files
            if data_type == "production":
                # For large production files, use chunking
                chunk_size = 10000
                chunks = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(csv_file, low_memory=False)

            # Get table definition
            table_def = self.data_model.get_table_definition(data_type)
            if not table_def:
                return df  # Return as-is if no definition

            # Apply field mappings
            field_mappings = table_def.get("field_mappings", {})
            for old_col, new_col in field_mappings.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Data type conversions
            if data_type == "well":
                df = self._process_well_data(df)
            elif data_type == "production":
                df = self._process_production_data(df)
            elif data_type == "business_associate":
                df = self._process_business_associate_data(df)

            return df

        except Exception as e:
            raise Exception(f"Error loading PPDM CSV {csv_file}: {str(e)}")

    def _process_well_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process well data with PPDM standards."""
        processed_df = df.copy()

        # Standardize UWI
        if "uwi" in processed_df.columns:
            processed_df["uwi"] = processed_df["uwi"].apply(
                lambda x: self.data_model.standardize_uwi(str(x)) if pd.notna(x) else ""
            )

        # Convert coordinates to numeric
        coord_columns = ["surface_latitude", "surface_longitude"]
        for col in coord_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # Clean well names
        if "well_name" in processed_df.columns:
            processed_df["well_name"] = (
                processed_df["well_name"].astype(str).str.strip()
            )

        # Add derived fields
        if (
            "surface_latitude" in processed_df.columns
            and "surface_longitude" in processed_df.columns
        ):
            # Add coordinate quality flag
            processed_df["coord_quality"] = np.where(
                (processed_df["surface_latitude"].notna())
                & (processed_df["surface_longitude"].notna())
                & (processed_df["surface_latitude"] != 0)
                & (processed_df["surface_longitude"] != 0),
                "GOOD",
                "POOR",
            )

        return processed_df

    def _process_production_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process production data with PPDM standards."""
        processed_df = df.copy()

        # Convert production date
        if "production_date" in processed_df.columns:
            processed_df["production_date"] = pd.to_datetime(
                processed_df["production_date"], errors="coerce"
            )

        # Convert volume columns to numeric
        volume_columns = ["daily_volume", "monthly_volume", "cumulative_volume"]
        for col in volume_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # Standardize product types
        if "product_type" in processed_df.columns:
            product_mapping = {
                "OIL": "OIL",
                "GAS": "GAS",
                "WATER": "WATER",
                "COND": "CONDENSATE",
                "CONDENSATE": "CONDENSATE",
            }
            processed_df["product_type"] = (
                processed_df["product_type"]
                .map(product_mapping)
                .fillna(processed_df["product_type"])
            )

        return processed_df

    def _process_business_associate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process business associate data."""
        processed_df = df.copy()

        # Clean business associate names
        if "ba_name" in processed_df.columns:
            processed_df["ba_name"] = processed_df["ba_name"].astype(str).str.strip()

        # Standardize BA types
        if "ba_type" in processed_df.columns:
            ba_type_mapping = {
                "OPERATOR": "OPERATOR",
                "SERVICE_COMPANY": "SERVICE_COMPANY",
                "GOVERNMENT": "GOVERNMENT",
                "INDIVIDUAL": "INDIVIDUAL",
            }
            processed_df["ba_type"] = (
                processed_df["ba_type"]
                .map(ba_type_mapping)
                .fillna(processed_df["ba_type"])
            )

        return processed_df


def read_ppdm_csv(csv_file: Union[str, Path], data_type: str = "well") -> pd.DataFrame:
    """Load PPDM CSV data with proper formatting.

    Convenience function for loading PPDM CSV files.

    Args:
        csv_file: Path to PPDM CSV file.
        data_type: Data type ('well', 'production', 'business_associate'),
            default 'well'.

    Returns:
        DataFrame with PPDM data.

    Example:
        >>> from geosmith.workflows.ppdm import read_ppdm_csv
        >>>
        >>> well_df = read_ppdm_csv('wells.csv', data_type='well')
        >>> print(f"Loaded {len(well_df)} wells")
    """
    parser = PpdmParser()
    return parser.load_ppdm_csv(csv_file, data_type)

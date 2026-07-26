"""
Configuration utilities for data processing.

This module contains functions for loading and parsing configuration files,
and extracting feature/metadata column definitions.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw Compustat data.

    Args:
        data_path: Path to raw data file

    Returns:
        DataFrame with raw financial data
    """
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path, engine='fastparquet')
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    return df


def load_ytd_features_config(config_path: str) -> List[str]:
    """
    Load YTD features configuration from YAML file.

    Args:
        config_path: Path to YTD features YAML file

    Returns:
        List of YTD feature names (without 'y' suffix)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Remove 'y' suffix from each feature to get base names
        ytd_features = config.get('ytd_features', [])
        base_features = [feat[:-1] if feat.endswith('y') else feat for feat in ytd_features]
        return base_features
    except FileNotFoundError:
        print(f"Warning: YTD features config not found at {config_path}, using empty list")
        return []


def load_feature_sets_config(config_path: str) -> Dict:
    """
    Load feature sets configuration from YAML file.

    Args:
        config_path: Path to feature sets YAML file

    Returns:
        Dictionary containing feature set definitions
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_section_columns(feature_sets_config: Dict, feature_set_name: str, section: str) -> Set[str]:
    """
    Recursively collect columns under a specific section (e.g., cash_flow) for a feature set.
    """
    cols = set()
    fs = feature_sets_config['feature_sets'].get(feature_set_name, {})
    # Add current section
    if section in fs and isinstance(fs[section], list):
        cols.update(fs[section])
    # Include from included sets
    if 'include_sets' in fs:
        for inc in fs['include_sets']:
            cols.update(collect_section_columns(feature_sets_config, inc, section))
    return cols


def get_metadata_cols(feature_sets_config: Dict) -> Set[str]:
    """
    Get metadata columns from feature sets configuration.

    Args:
        feature_sets_config: Feature sets configuration dictionary

    Returns:
        Set of metadata column names, with firm_id and quarter ensured
    """
    metadata_cols = set(feature_sets_config.get('metadata_columns', []))

    # add the renames firm_id and quarter if not already present
    metadata_cols.update(['firm_id', 'quarter'])  # These are created during processing
    return metadata_cols


def get_feature_columns(feature_sets_config: Dict, feature_set_name: str) -> Set[str]:
    """
    Get the set of columns for a specific feature set.

    Args:
        feature_sets_config: Feature sets configuration dictionary
        feature_set_name: Name of the feature set (e.g., 'core', 'proforma')

    Returns:
        Set of column names to include
    """
    if feature_set_name not in feature_sets_config['feature_sets']:
        raise ValueError(f"Feature set '{feature_set_name}' not found in configuration")

    feature_set = feature_sets_config['feature_sets'][feature_set_name]
    columns = set()

    # Add metadata columns (always included)
    columns.update(get_metadata_cols(feature_sets_config))

    # Handle include_sets (e.g., pro_forma includes core)
    if 'include_sets' in feature_set:
        for included_set in feature_set['include_sets']:
            included_columns = get_feature_columns(feature_sets_config, included_set)
            columns.update(included_columns)

    # Add columns from each category
    for key, values in feature_set.items():
        if key in ['description', 'include_sets']:
            continue
        if isinstance(values, list):
            columns.update(values)

    return columns


def get_account_names_for_feature_set(
    feature_sets_config: Dict,
    feature_set_name: str,
    target_variables: List[str],
) -> List[str]:
    """Collect account names to use in tuple dataset for a given feature set.

    Includes:
      * All non-metadata feature names defined in the feature set (including inherited sets).
      * All target variable names from the main config.
      * All computed feature names listed under "computed" for the set and any included sets.

    Returns a sorted list of unique column names.
    """
    fs_all = feature_sets_config["feature_sets"]

    def collect_all_features(name: str, seen: Optional[Set[str]] = None) -> Set[str]:
        if seen is None:
            seen = set()
        if name in seen:
            return set()
        seen.add(name)
        fs = fs_all.get(name, {})
        cols: Set[str] = set()
        for key, vals in fs.items():
            if key in {"description", "include_sets"}:
                continue
            if isinstance(vals, list):
                cols.update(vals)
        # Recurse into included sets
        if "include_sets" in fs:
            for inc in fs["include_sets"]:
                cols.update(collect_all_features(inc, seen))

        # Add 'scale'
        cols.add("scale")
        return cols

    feature_cols = collect_all_features(feature_set_name)

    # Remove metadata columns from account names
    metadata_cols = set(feature_sets_config.get("metadata_columns", []))
    account_names = (feature_cols - metadata_cols).union(set(target_variables))
    return sorted(account_names)


def get_all_feature_names(feature_sets_config: Dict, feature_set_name: str) -> List[str]:
    """
    Get all non-metadata feature names for a feature set (for targets.all mode).

    Args:
        feature_sets_config: Feature sets configuration dictionary
        feature_set_name: Name of the feature set

    Returns:
        List of all feature names (excluding metadata)
    """
    feature_columns = get_feature_columns(feature_sets_config, feature_set_name)
    metadata_cols = get_metadata_cols(feature_sets_config)

    # Remove metadata columns and return sorted list
    feature_names = sorted(feature_columns - metadata_cols)
    return feature_names


def get_target_variables(data_config: Dict, feature_sets_config: Dict, feature_set_name: str) -> List[str]:
    """
    Get target variables from config, handling both old and new formats.

    Args:
        data_config: Data configuration dictionary
        feature_sets_config: Feature sets configuration dictionary
        feature_set_name: Name of the feature set being used

    Returns:
        List of target variable names
    """
    # Check for new format first
    targets_config = data_config.get('targets', {})

    if targets_config:
        # New format: targets.all and targets.variables
        use_all = targets_config.get('all', False)

        if use_all:
            # Use all features as targets
            target_variables = get_all_feature_names(feature_sets_config, feature_set_name)
            print(f"Using all {len(target_variables)} features as targets (targets.all=true)")
        else:
            # Use specified variables
            target_variables = targets_config.get('variables', [])
            if not target_variables:
                raise ValueError("targets.all is false but no targets.variables specified")
            print(f"Using {len(target_variables)} specified target variables")
    else:
        # Fall back to old format for backwards compatibility
        target_variables = data_config.get('target_variables', ['niq', 'fcfq'])
        print(f"Using legacy target_variables format: {target_variables}")

    return target_variables


def validate_targets_in_features(target_variables: List[str], feature_sets_config: Dict, feature_set_name: str) -> List[str]:
    """
    Validate that all target variables are included in the feature set and return valid ones.

    Args:
        target_variables: List of target variable names from main config
        feature_sets_config: Feature sets configuration dictionary
        feature_set_name: Name of the feature set being used

    Returns:
        List of validated target variables that exist in the feature set
    """
    feature_columns = get_feature_columns(feature_sets_config, feature_set_name)

    valid_targets = []
    invalid_targets = []

    for target in target_variables:
        if target in feature_columns:
            valid_targets.append(target)
        else:
            invalid_targets.append(target)

    if invalid_targets:
        print(f"Warning: Target variables not found in feature set '{feature_set_name}': {invalid_targets}")
        print(f"These targets will be ignored. Valid targets: {valid_targets}")

    if not valid_targets:
        raise ValueError(f"No valid target variables found in feature set '{feature_set_name}'. Available features: {sorted(feature_columns)}")

    return valid_targets


def dataset_suffix(feature_set: str, dataset_tag: Optional[str] = None) -> str:
    """Build the processed-file suffix for a (feature_set, dataset_tag) pair.

    When dataset_tag is set, files are named "{kind}__{feature_set}__{tag}.parquet".
    When it is missing, the legacy "{kind}__{feature_set}.parquet" naming is used.
    """
    if dataset_tag:
        return f"{feature_set}__{dataset_tag}"
    return feature_set


def resolve_dataset_path(
    data_dir: Path,
    kind: str,
    feature_set: str,
    dataset_tag: Optional[str] = None,
    ext: str = "parquet",
) -> Path:
    """Find a processed-data file for (kind, feature_set, dataset_tag).

    Falls back to the legacy untagged path if the tagged file doesn't exist,
    so old datasets keep working without a rebuild.
    """
    if dataset_tag:
        tagged = data_dir / f"{kind}__{feature_set}__{dataset_tag}.{ext}"
        if tagged.exists():
            return tagged
    return data_dir / f"{kind}__{feature_set}.{ext}"


def parse_cfg_list(lst) -> List[str]:
    """
    Robustly convert YAML-loaded value (which may be list, string, None) into list of feature names.

    Args:
        lst: YAML-loaded value (list, string, or None)

    Returns:
        List of cleaned feature names
    """
    if lst is None:
        return []
    if isinstance(lst, list):
        raw_items = lst
    elif isinstance(lst, str):
        # Split on newlines in case of malformed YAML like "-ltq" lines without spaces
        raw_items = [line.strip() for line in lst.splitlines() if line.strip()]
    else:
        raw_items = [str(lst)]
    clean = []
    for item in raw_items:
        if item.startswith('-') and len(item) > 1:
            clean.append(item[1:].strip())
        else:
            clean.append(item.strip())
    return [c for c in clean if c]


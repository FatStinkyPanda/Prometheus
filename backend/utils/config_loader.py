# backend/utils/config_loader.py

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up a basic logger for this module to report issues during the config loading process itself.
# A more robust logger will be configured using the settings loaded by this utility.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    A robust, production-ready utility for loading and managing YAML configuration files.

    This class provides static methods to:
    - Load a primary configuration file.
    - Merge multiple configuration files.
    - Apply overrides from environment variables.
    - Handle errors gracefully during the loading process.
    """

    @staticmethod
    def _find_project_root(marker: str = 'run_prometheus.py') -> Path:
        """
        Find the project root directory by searching upwards for a marker file.

        Args:
            marker (str): The name of the file or directory that marks the project root.
                          'run_prometheus.py' is a reliable marker for this project.

        Returns:
            Path: The absolute path to the project root directory.

        Raises:
            FileNotFoundError: If the project root cannot be determined.
        """
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / marker).exists():
                return parent
        raise FileNotFoundError(f"Project root marker '{marker}' not found. Could not determine project root.")

    @staticmethod
    def _deep_merge(source: Dict, destination: Dict) -> Dict:
        """
        Recursively merge two dictionaries.

        Values from the source dictionary overwrite values in the destination dictionary.
        If a key exists in both and both values are dictionaries, they are merged recursively.

        Args:
            source (Dict): The dictionary providing new values.
            destination (Dict): The dictionary to be updated.

        Returns:
            Dict: The merged dictionary.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                destination[key] = ConfigLoader._deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    @staticmethod
    def _apply_env_overrides(config: Dict, prefix: str = 'PROMETHEUS') -> Dict:
        """
        Recursively scan the configuration and apply overrides from environment variables.

        Environment variables should be in the format: PREFIX_KEY1_KEY2=value
        For example, PROMETHEUS_DATABASE_HOST would override config['database']['host'].

        Args:
            config (Dict): The configuration dictionary to process.
            prefix (str): The prefix for environment variables.

        Returns:
            Dict: The configuration dictionary with environment variables applied.
        """
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = ConfigLoader._apply_env_overrides(value, f"{prefix}_{key.upper()}")
            else:
                env_var_name = f"{prefix}_{key.upper()}"
                env_var_value = os.getenv(env_var_name)
                if env_var_value is not None:
                    original_type = type(value)
                    try:
                        if original_type == bool:
                            config[key] = env_var_value.lower() in ('true', '1', 't', 'y', 'yes')
                        elif original_type == int:
                            config[key] = int(env_var_value)
                        elif original_type == float:
                            config[key] = float(env_var_value)
                        else:
                            config[key] = env_var_value # Keep as string for other types
                        logger.info(f"Overrode '{key}' with value from environment variable '{env_var_name}'.")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error casting env var '{env_var_name}' with value '{env_var_value}' to type {original_type}: {e}. Using string value instead.")
                        config[key] = env_var_value
        return config

    @staticmethod
    def load_config(
        primary_config: str = "prometheus_config.yaml",
        merge_configs: Optional[List[str]] = None,
        config_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Loads the main configuration and optionally merges others.

        The process is as follows:
        1. Finds the project root and the 'backend/config' directory.
        2. Loads the primary configuration file (e.g., prometheus_config.yaml).
        3. Loads and merges any additional configuration files (e.g., logging_config.yaml).
        4. Applies overrides from environment variables.

        Args:
            primary_config (str): The filename of the main configuration file.
            merge_configs (Optional[List[str]]): A list of other config filenames to merge.
            config_dir (Optional[Path]): The path to the config directory. If None, it's inferred.

        Returns:
            Dict[str, Any]: The fully loaded and merged configuration dictionary.

        Raises:
            FileNotFoundError: If a specified configuration file does not exist.
            yaml.YAMLError: If a configuration file is malformed.
        """
        try:
            if config_dir is None:
                project_root = ConfigLoader._find_project_root()
                # --- THIS IS THE FIX ---
                # The config directory is inside the 'backend' folder.
                config_dir_path = project_root / 'backend' / 'config'
            else:
                config_dir_path = config_dir

            if not config_dir_path.is_dir():
                raise FileNotFoundError(f"Configuration directory not found at '{config_dir_path}'")

            # 1. Load primary configuration file
            primary_path = config_dir_path / primary_config
            logger.info(f"Loading primary configuration from: {primary_path}")
            
            with open(primary_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise TypeError(f"Primary config file '{primary_config}' did not load as a dictionary.")
            
            # 2. Load and merge additional configurations
            if merge_configs:
                for config_file in merge_configs:
                    merge_path = config_dir_path / config_file
                    logger.info(f"Merging configuration from: {merge_path}")
                    if merge_path.exists():
                        with open(merge_path, 'r', encoding='utf-8') as f:
                            to_merge = yaml.safe_load(f)
                        if isinstance(to_merge, dict):
                            config = ConfigLoader._deep_merge(to_merge, config)
                        else:
                            logger.warning(f"Config file '{config_file}' did not load as a dictionary. Skipping merge.")
                    else:
                        logger.warning(f"Merge config file not found: '{merge_path}'. Skipping.")

            # 3. Apply environment variable overrides
            logger.info("Applying environment variable overrides with prefix 'PROMETHEUS'...")
            config = ConfigLoader._apply_env_overrides(config)

            logger.info("Configuration loaded successfully.")
            return config

        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}", exc_info=True)
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
            raise
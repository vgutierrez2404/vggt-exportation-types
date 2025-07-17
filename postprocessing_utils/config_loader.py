import os
import yaml

class ConfigLoader:
    """
    Load configuration for the application from a YAML file. 
    """
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()

        self.load_type = self.config.get("load_type", "normal") 

    def _load_config(self): # underscore at start means private method 
        """Private method to load the YAML configuration file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file '{self.config_file}' not found.")

        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

    def get(self, key_path, default=None):
        """
        Get a value from the config using a dot-separated key path.
        Example: config.get("video.model.model_checkpoints")
        """
        keys = key_path.split(".")
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

    def reload(self):
        """Reload the configuration from the file."""
        self.config = self._load_config()
        self.load_type = self.config.get("load_type", "normal") 
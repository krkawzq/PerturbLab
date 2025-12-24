"""Base configuration class for models.

Provides serialization, validation, and type constraints for model configurations.
"""

import inspect
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from typing_extensions import Self


# Allowed basic types for config values
BASIC_TYPES = (int, float, str, bool, type(None))


@dataclass
class Config:
    """Base class for configurations.

    Enforces type constraints and provides serialization capabilities.

    Rules:
    - Non-private attributes (not starting with '_') must be basic types
    - Basic types: int, float, str, bool, None, or nested dict/list of these
    - Provides to_dict(), save(), and load() methods

    Examples
    --------
    >>> class MyConfig(Config):
    ...     def __init__(self, hidden_dim: int, dropout: float):
    ...         self.hidden_dim = hidden_dim
    ...         self.dropout = dropout
    ...         self._model_type = "transformer"  # Private, not serialized
    >>>
    >>> config = MyConfig(hidden_dim=128, dropout=0.1)
    >>> config.save("config.json")
    >>> loaded = MyConfig.load("config.json")
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate attribute types before setting.

        Private attributes (starting with '_') can be any type.
        Public attributes must be basic types or nested dict/list.
        """
        if not name.startswith("_"):
            self._validate_value(value, name)

        super().__setattr__(name, value)

    @staticmethod
    def _validate_value(value: Any, name: str = "value") -> None:
        """Recursively validate that value is serializable.

        Parameters
        ----------
        value : Any
            Value to validate.
        name : str
            Name of the attribute (for error messages).

        Raises
        ------
        TypeError
            If value contains non-serializable types.
        """
        # Basic types are always valid
        if isinstance(value, BASIC_TYPES):
            return

        # Lists: all elements must be valid
        if isinstance(value, list):
            for i, item in enumerate(value):
                try:
                    Config._validate_value(item, f"{name}[{i}]")
                except TypeError as e:
                    raise TypeError(f"Invalid type in {name}[{i}]: {e}")
            return

        # Dicts: all values must be valid, keys must be strings
        if isinstance(value, dict):
            for key, val in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Dict keys must be strings, got {type(key).__name__} " f"for key in {name}"
                    )
                try:
                    Config._validate_value(val, f"{name}['{key}']")
                except TypeError as e:
                    raise TypeError(f"Invalid type in {name}['{key}']: {e}")
            return

        # Invalid type
        raise TypeError(
            f"Attribute '{name}' has invalid type {type(value).__name__}. "
            f"Only basic types (int, float, str, bool, None) and nested "
            f"dict/list are allowed."
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Only includes public attributes (not starting with '_').

        Returns
        -------
        dict[str, Any]
            Dictionary representation of config.

        Examples
        --------
        >>> config = MyConfig(hidden_dim=128, dropout=0.1)
        >>> config.to_dict()
        {'hidden_dim': 128, 'dropout': 0.1}
        """
        return {
            key: self._deep_copy(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    @staticmethod
    def _deep_copy(value: Any) -> Any:
        """Deep copy value (handles nested dict/list).

        Parameters
        ----------
        value : Any
            Value to copy.

        Returns
        -------
        Any
            Copied value.
        """
        if isinstance(value, dict):
            return {k: Config._deep_copy(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [Config._deep_copy(item) for item in value]
        else:
            # Basic types are immutable, no need to copy
            return value

    def save(self, path: str | Path, update: bool = True) -> None:
        """Save config to JSON file.

        Supports incremental update mode: if the file exists, only updates
        the keys present in this config, preserving other keys in the file.

        Parameters
        ----------
        path : str or Path
            Path to save config file.
        update : bool, default=True
            If True and file exists, merge with existing keys.
            If False, overwrite the entire file.

        Examples
        --------
        >>> # Create initial config
        >>> config1 = ConfigA(param_a=1, param_b=2)
        >>> config1.save("config.json")  # {'param_a': 1, 'param_b': 2}
        >>>
        >>> # Update only some keys
        >>> config2 = ConfigB(param_c=3)
        >>> config2.save("config.json")  # {'param_a': 1, 'param_b': 2, 'param_c': 3}
        >>>
        >>> # Overwrite entirely
        >>> config2.save("config.json", update=False)  # {'param_c': 3}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get current config dict
        config_dict = self.to_dict()

        # If update mode and file exists, merge with existing data
        if update and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_dict = json.load(f)

                # Merge: existing keys are preserved, new keys are added/updated
                existing_dict.update(config_dict)
                config_dict = existing_dict
            except (json.JSONDecodeError, IOError):
                # If file is corrupt or unreadable, just save current config
                pass

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], strict: bool = True) -> Self:
        """Create config from dictionary with query-style loading.

        Inspects __init__ signature to fill missing keys with defaults.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Configuration dictionary.
        strict : bool, default=True
            If True, raise error for required parameters missing in config_dict.
            If False, skip missing required parameters (for direct attribute setting).

        Returns
        -------
        Self
            Config instance.

        Raises
        ------
        ValueError
            If strict=True and a required parameter (no default) is missing.

        Examples
        --------
        >>> # Config with defaults
        >>> class MyConfig(Config):
        ...     def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        ...         self.hidden_dim = hidden_dim
        ...         self.dropout = dropout
        >>>
        >>> # Partial dict uses defaults
        >>> config = MyConfig.from_dict({'hidden_dim': 256})
        >>> print(config.dropout)  # 0.1 (default)
        >>>
        >>> # Missing required parameter raises error
        >>> class StrictConfig(Config):
        ...     def __init__(self, required_param: int):
        ...         self.required_param = required_param
        >>>
        >>> StrictConfig.from_dict({})  # ValueError!
        """
        # Inspect __init__ signature
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            # No __init__ or can't inspect, just try direct instantiation
            return cls(**config_dict)

        # Build kwargs with defaults
        init_kwargs = {}
        missing_required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue

            if param_name in config_dict:
                # Use value from config_dict
                init_kwargs[param_name] = config_dict[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Use default value from __init__
                init_kwargs[param_name] = param.default
            else:
                # Required parameter missing
                missing_required.append(param_name)

        # Check for missing required parameters
        if strict and missing_required:
            raise ValueError(
                f"Missing required parameters for {cls.__name__}: {missing_required}. "
                f"These parameters have no default values in __init__."
            )

        # Create instance
        try:
            return cls(**init_kwargs)
        except TypeError as e:
            if strict:
                raise
            # Fallback: create empty instance and set attributes
            instance = cls.__new__(cls)
            for key, value in config_dict.items():
                setattr(instance, key, value)
            return instance

    @classmethod
    def load(cls, path: str | Path, strict: bool = True) -> Self:
        """Load config from JSON file with query-style loading.

        Reads available keys from the file and fills missing keys with
        default values from __init__. Raises error if required parameters
        (no default) are missing.

        Parameters
        ----------
        path : str or Path
            Path to config file.
        strict : bool, default=True
            If True, raise error for required parameters missing in file.
            If False, skip missing required parameters.

        Returns
        -------
        Self
            Loaded config instance.

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist.
        ValueError
            If strict=True and required parameters are missing from file.

        Examples
        --------
        >>> # Config file: {"hidden_dim": 256}
        >>> class MyConfig(Config):
        ...     def __init__(self, hidden_dim: int, dropout: float = 0.1):
        ...         self.hidden_dim = hidden_dim
        ...         self.dropout = dropout
        >>>
        >>> config = MyConfig.load("config.json")
        >>> print(config.hidden_dim)  # 256 (from file)
        >>> print(config.dropout)     # 0.1 (default value)
        >>>
        >>> # Missing required parameter
        >>> class StrictConfig(Config):
        ...     def __init__(self, required_param: int):
        ...         self.required_param = required_param
        >>>
        >>> # Config file: {}
        >>> StrictConfig.load("empty.json")  # ValueError!
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict, strict=strict)

    def update(self, **kwargs) -> None:
        """Update config attributes.

        Parameters
        ----------
        **kwargs
            Attributes to update.

        Examples
        --------
        >>> config.update(hidden_dim=256, dropout=0.2)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of config."""
        config_dict = self.to_dict()
        items = ", ".join(f"{k}={v!r}" for k, v in config_dict.items())
        return f"{self.__class__.__name__}({items})"

    def __eq__(self, other: Any) -> bool:
        """Compare configs by their dict representation."""
        if not isinstance(other, Config):
            return False
        return self.to_dict() == other.to_dict()

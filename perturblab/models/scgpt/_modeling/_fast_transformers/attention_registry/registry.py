#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#


class Registry:
    """Hold the available attention implementations and their required
    parameters."""

    def __init__(self):
        self._classes = {}
        self._class_params = {}
        self._parameters = {}

    def register(self, key, class_object, parameter_tuples):
        # register the class if the key is new
        if key in self._classes:
            raise ValueError(f"{key} is already registered")
        self._classes[key] = class_object

        # register the parameters
        for parameter, spec in parameter_tuples:
            if parameter in self._parameters and self._parameters[parameter] != spec:
                raise ValueError(
                    f"{parameter} is already registered with " f"spec {self._parameters[parameter]!r} instead of {spec!r}"
                )
            self._parameters[parameter] = spec

        # note which parameters are needed by this class
        self._class_params[key] = [p for p, s in parameter_tuples]

    def __contains__(self, key):
        return key in self._classes

    def __getitem__(self, key):
        return self._classes[key], self._class_params[key]

    @property
    def keys(self):
        return list(self._classes.keys())

    def contains_parameter(self, key):
        return key in self._parameters

    def validate_parameter(self, key, value):
        try:
            return self._parameters[key].get(value)
        except Exception as e:
            raise ValueError(f"Invalid value {value!r} for " f"parameter {key!r}") from e


AttentionRegistry = Registry()
RecurrentAttentionRegistry = Registry()
RecurrentCrossAttentionRegistry = Registry()

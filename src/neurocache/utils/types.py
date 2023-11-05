# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all
import enum


class NeurocacheType(str, enum.Enum):
    ONDEVICE = "ONDEVICE"


class TaskType(str, enum.Enum):
    DEFAULT = "DEFAULT"
    CAUSAL_LM = "CAUSAL_LM"

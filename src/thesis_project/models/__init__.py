from enum import Enum


class OutputType(str, Enum):
    CLASSIFICATION = "clf"
    REGRESSION = "reg"

from enum import Enum


class DataSplit(Enum):
    TRAIN = 0
    TEST = 2
    VALIDATION = 4


class FeatureSet(Enum):
    INPUT = 0
    TARGET = 1

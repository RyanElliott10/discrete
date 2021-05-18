from abc import ABC, abstractmethod


class Preprocessor(ABC):

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError


class ScalePreprocessor(Preprocessor):

    def __init__(self):
        pass

    def preprocess(self):
        pass

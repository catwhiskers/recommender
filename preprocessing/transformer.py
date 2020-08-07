from abc import ABC, abstractmethod

class Transformer(ABC):
    @abstractmethod
    def get_feature_vectors(self):
        pass

from abc import ABC, abstractmethod


class BasePreprocessor(ABC):

    @abstractmethod
    def fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        return self.fit(X).transform(X)

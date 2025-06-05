from abc import ABC


class Trainer(ABC):
    def train(self):
        raise NotImplementedError("Abstract base class")

    def validate(self):
        raise NotImplementedError("Abstract base class")

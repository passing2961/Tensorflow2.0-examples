from collections import namedtuple
from .models.text import TextClassifier


class Semantic(namedtuple('Semantic', ['name', 'confidence'])):
    def to_dict(self):
        return {
            'name': self.name,
            'confidence': self.confidence,
        }


class Domain(Semantic):
    pass


class Intent(Semantic):
    pass


class SemanticFinder:
    _semantic_cls = Semantic

    def __init__(self, model_filename):
        self.model = TextClassifier.load(model_filename)

    def findall(self, utterance):
        return [
            self._semantic_cls(label, confidence)
            for label, confidence in self.model.predict([utterance.morphemes])
        ]

    def find(self, utterance):
        return max(self.findall(utterance), key=lambda items: items[1])


class DomainFinder(SemanticFinder):
    _semantic_cls = Domain


class IntentFinder(SemanticFinder):
    _semantic_cls = Intent
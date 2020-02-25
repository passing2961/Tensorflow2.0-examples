class Entity:
    def __init__(self, phrase):
        self.phrase = phrase

class EntityRecognizer:
    def finditer(self, utterance):
        pass

def finditer(self, utterance):
    recognizer = EntityRecognizer()
    yield from recognizer.finditer(utterance)
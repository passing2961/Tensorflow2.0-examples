class Action:
    def act(self):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()

class TextAction(Action):
    def __init__(self, text):
        self.text = text

    def act(self):
        print('Display text:', self.text, sep='\n')

    def to_dict(self):
        return {
            'action': 'text',
            'payload': {
                'text': self.text,
            }
        }

class SpeechAction(Action):
    def __init__(self, text): # voice?
        self.text = text

    def act(self):
        print('speech text:', self.text)

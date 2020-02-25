class Response:
    def __init__(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        self.actions = actions

    def to_dict(self):
        return {
            'actions': [action.to_dict() for action in self.actions]
        }
from mecab import MeCab

_mecab = MeCab()

_TAG_PLACEHOLDERS = {
    'SF': '<SF>',  # . ! ?
    'SE': '<SE>',  # …
    'SSO': '<SSO>',  # ( [
    'SSC': '<SSC>',  # ) ]
    'SC': '<SC>',  # , · / :
    'SY': '<SY>',  # 기타 기호
    'SL': '<SL>',  # 외국어
    'SH': '<SH>',  # 한자
    'SN': '<SN>',  # 숫자
}


class Utterance:
    def __init__(self, text):
        self.domains = []
        self.intents = []

        self.text = text
        self.tagged = _mecab.pos(text)
        self.morphemes = [
            _TAG_PLACEHOLDERS.get(tag, morpheme)
            for morpheme, tag in self.tagged
        ]

    @property
    def domain(self):
        if not self.domains:
            return None
        return max(self.domains, key=lambda domain: domain.confidence)

    @property
    def intent(self):
        if not self.intents:
            return None
        return max(self.intents, key=lambda intent: intent.confidence)

    def matches(self, conditions):
        semantics = self.domains + self.intents

        for condition in conditions:
            satisfied = False

            for semantic in semantics:
                if not isinstance(semantic, type(condition)):
                    continue

                if (semantic.name == condition.name) and (semantic.confidence >= condition.confidence):
                    satisfied = True
                    break

            if not satisfied:
                return False

        return True

    def to_dict(self):
        return {
            'domains': [domain.to_dict() for domain in self.domains],
            'intents': [intent.to_dict() for intent in self.intents],
            'text': self.text,
            'morphemes': self.morphemes,
        }
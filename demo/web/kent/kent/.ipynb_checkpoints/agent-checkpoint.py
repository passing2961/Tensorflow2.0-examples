import functools

from collections import OrderedDict, namedtuple

from .responses import Response
from .actions import TextAction

Condition = namedtuple('Condition', ['key', 'value', 'threshold'])

class Agent:
    def __init__(self, domain_finder, intent_finder):
        self.domain_finder = domain_finder
        self.intent_finder = intent_finder

        self.callbacks = OrderedDict()

    def listen(self, utterance):
        # 도메인 찾고
        # 나머지는 도메인에 의존해서 전처리 ㄱㄱㄱ
        self._understand_utterance(utterance)

        return [
            Response(callback(utterance)) for callback in self._resolve(utterance)
        ]

    def on(self, semantic):
        def decorator(func):
            self.callbacks.setdefault(func, set()).add(semantic)
            return func
        return decorator

    def _understand_utterance(self, utterance):
        utterance.domains = self.domain_finder.findall(utterance)
        utterance.intents = self.intent_finder.findall(utterance)

    def _resolve(self, utterance):
        return [
            callback for callback, conditions in self.callbacks.items()
            if utterance.matches(conditions)
        ]


'''
rules = {
    domain:
    intent:
}
'''

# 데이터 만들기
# 10-fold cross validaiton -- domain 인식 성능
# entity recognition -- 평가는 정성적으로 사람들에게 조사
# entity 규칙 만들기
# ----------------------------
# overall 성능 평가

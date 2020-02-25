import json

from kent.agent import Agent
from kent.utterance import Utterance
from kent.understanding import DomainFinder, IntentFinder
from kent.understanding import Domain, Intent
from kent.actions import TextAction

agent = Agent(
    domain_finder=DomainFinder('agents/kepco/domain.pickle'),
    intent_finder=IntentFinder('agents/kepco/intent.pickle'),
)

@agent.on(Domain('전기/요금/청구서', 0.5))
@agent.on(Intent('문의하다/HOW > 카카오페이/고객센터', 0.5))
def _(utterance):
    return TextAction('카카오페이 청구서 관련한 문의는 아래 경로를 통해 하실 수 있습니다. \n- 카카오페이 고객센터 1644-7405(평일 09~18시)로 연락 \n- 웹 고객센터(http://www.kakao.com/requests?locale=ko&service=56)에 문의 접수\n- 카카오톡 > 더보기 > 카카오페이 > 설정 > 고객문의에 1:1문의 접수')

@agent.on(Domain('전기/요금/청구서', 0.5))
@agent.on(Intent('문의하다 > 카카오페이/신청/준비물', 0.5))
def _(utterance):
    return TextAction('카카오페이 청구서 신청시 "계약 고객명","고객번호"가 반드시 필요합니다. 고객번호는 기존 받으시던 청구서에서 확인하실 수 있습니다.')

@agent.on(Domain('전기/요금/청구서', 0.5))
@agent.on(Intent('문의하다 > 카카오페이/항목', 0.5))
def _(utterance):
    return TextAction('카카오페이 청구서에는 전월대비, 거주지역, 연간평균, 월별 사용량 등 사용 내역을 알기 쉽게 제공하고 있습니다.')

@agent.on(Domain('전기/요금/청구서', 0.5))
@agent.on(Intent('문의하다/CAN > 카카오페이/납부', 0.5))
def _(utterance):
    return TextAction('네, 카카오페이 청구서에서 요금 납부가 가능합니다. 결제하실 카드를 등록 후 "납부하기" 버튼 클릭하면 정상처리 됩니다.')

while True:
    message = input('In: ')
    utterance = Utterance(message)
    responses = agent.listen(utterance)

    print(utterance.to_dict())
    for response in responses:
        print(response.to_dict())

    print(json.dumps(utterance.to_dict(), ensure_ascii=False, indent=2))
    print()
import argparse
import json
import os

from kent.utterance import Utterance
from kent.models.text import MLPTextClassifier

def load_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as input_file:
        corpus = json.load(input_file)

    for item in corpus:
        for sentence in item['sentences']:
            yield {
                'utterance': Utterance(sentence).morphemes,
                'domain': item['domain'],
                'intent': f'{item["intent"]} > {item["objective"]}',
            }

parser = argparse.ArgumentParser()
parser.add_argument('corpus_filename')
parser.add_argument('agents_directory')
parser.add_argument('agent_name')
args = parser.parse_args()

print('Loading corpus...')
corpus = list(load_corpus(args.corpus_filename))

agent_directory = os.path.join(args.agents_directory, args.agent_name)
agent_filename = os.path.join(agent_directory, '__init__.py')
os.makedirs(agent_directory, exist_ok=True)

print('Building domain classifier...')
domain_clf = MLPTextClassifier()
domain_clf.fit(
    [item['utterance'] for item in corpus],
    [item['domain'] for item in corpus])
domain_clf.save(os.path.join(args.agents_directory, args.agent_name, 'domain.pickle'))

print('Building intent classifier...')
intent_clf = MLPTextClassifier()
intent_clf.fit(
    [item['utterance'] for item in corpus],
    [item['intent'] for item in corpus])
intent_clf.save(os.path.join(args.agents_directory, args.agent_name, 'intent.pickle'))

if not os.path.exists(agent_filename):
    with open(os.path.join(args.agents_directory, args.agent_name, '__init__.py'), 'w', encoding='utf-8') as output_file:
        print(f"""
import functools

from kepco.agents import load_agent
from kent.understanding import Domain as _Domain, Intent as _Intent
from kent.actions import TextAction

Domain = functools.partial(_Domain, confidence=0.5)
Intent = functools.partial(_Intent, confidence=0.5)

agent = load_agent('{args.agent_name}')

@agent.on(Domain('<domain>'))
@agent.on(Intent('<intent>'))
def _(utterance):
    return TextAction('<response>')
    """.strip(), file=output_file)
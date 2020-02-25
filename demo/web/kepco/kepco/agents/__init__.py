import os
import functools

from kent.agent import Agent
from kent.understanding import DomainFinder, IntentFinder

def load_agent(name):
    agent_directory = os.path.join(os.path.dirname(__file__), name)
    return Agent(
        domain_finder=DomainFinder(os.path.join(agent_directory, 'domain.pickle')),
        intent_finder=IntentFinder(os.path.join(agent_directory, 'intent.pickle')),
    )
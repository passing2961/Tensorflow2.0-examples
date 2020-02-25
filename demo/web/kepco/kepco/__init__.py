import importlib
import json
import os

from flask import Flask, g, jsonify, render_template, request

from kent.agent import Agent
from kent.utterance import Utterance

app = Flask(__name__)

def load_agent(name):
    if not hasattr(g, 'agents'):
        g.agents = {}

    if name not in g.agents:
        g.agents[name] = importlib.import_module(f'kepco.agents.{name}').agent

    return g.agents[name]

@app.route('/')
def show_index():
    return render_template('index.html')

@app.route('/agents/<name>', methods=['POST'])
def say(name):
    agent = load_agent(name)
    utterance = Utterance(request.form['utterance'])
    responses = agent.listen(utterance)

    return jsonify(
        text=utterance.text,
        morphemes=utterance.morphemes,
        domains=[domain.to_dict() for domain in sorted(utterance.domains, key=lambda domain: domain.confidence, reverse=True)],
        intents=[intent.to_dict() for intent in sorted(utterance.intents, key=lambda intent: intent.confidence, reverse=True)],
        responses=[response.to_dict() for response in responses],
    )
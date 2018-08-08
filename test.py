#usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from betazero.utils import *
from betazero.ai import Agent
from betazero import tic_tac_toe as game

session = game.Session()
agent = Agent(game)
agent.update_session(*session.reset())
#raw_input("pause")
agent.update_session(*session.do_action_index(0,0))
agent.update_session(*session.do_action_index(0,1))
agent.update_session(*session.do_action_index(1,0))
agent.update_session(*session.do_action_index(1,1))
agent.update_session(*session.do_action_index(0,2))
agent.update_session(*session.do_action_index(1,2))
agent.update_session(*session.do_action_index(2,2))
#print(agent.state_history[-1])
print(agent.generate_action())
agent.update_session(*session.do_action_index(2,0))
#print(agent.state_history[-1])
print(agent.generate_action())
agent.update_session(*session.do_action_index(2,1))
#print(agent.state_history[-1])
print(agent.generate_action())

for i, pdf in enumerate(agent.pdfs):
    plt.plot(pdf, label = str(i))

plt.legend()
plt.show()

#usr/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt

from beta_zero.AI import *
from beta_zero import TicTacToe as game

agent = Agent(game)
session = game.Session()
agent.update_session(*session.reset())
agent.update_session(*session.do_action_index(0,0))
agent.update_session(*session.do_action_index(0,1))
agent.update_session(*session.do_action_index(1,0))
agent.update_session(*session.do_action_index(1,1))

#print(agent.action_prediction_history[-1])
#print(agent.generate_action())

agent.update_session(*session.do_action_index(2,0))

#print(agent.state_history)
#print(agent.action_prediction_history)
#agent.update_session(*session.do_action_index(2,1))

#pdfs = [pdf/np.sum(pdf) for pdf in pdfs]

for i, pdf in enumerate(agent.pdfs):
    plt.plot(pdf, label = str(i))
#plt.plot(support, max_pdf(pdfs), marker = "*")

plt.legend()
plt.show()

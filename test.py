#usr/bin/python3
from beta_zero import AI
from beta_zero import TicTacToe as Game

agent = AI.Agent(Game)
session = Game.Session()

agent.update_session(*session.do_action_index(0,0))
agent.update_session(*session.do_action_index(0,1))
agent.update_session(*session.do_action_index(1,0))
agent.update_session(*session.do_action_index(1,1))

print(agent.generate_action())

agent.update_session(*session.do_action_index(2,0))
agent.update_session(*session.do_action_index(2,1))


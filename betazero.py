#usr/bin/python3
games = ['go', 'tic_tac_toe']

import argparse
parser = argparse.ArgumentParser(description='BetaZero App')
parser.add_argument('-g', "--game", choices=games, default = 'tic_tac_toe')
parser.add_argument(
    '-s', '--self-train', action="store_true", help='self training mode')
parser.add_argument(
    '-m', '--model', help='path to the hdf5 model file')
parser.add_argument(
    '-i',
    '--save-interval',
    type=int,
    default=1000,
    help='save model every i matches')
args = parser.parse_args()

import timeit
import numpy as np
import matplotlib.pyplot as plt
from betazero import ai

print("seleted game:", args.game)
if args.game == games[0]:
    from betazero import go as game
elif args.game == games[1]:
    from betazero import tic_tac_toe as game

if not args.model:
    args.model = args.game + "_model.h5"

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

agent = ai.Agent(game, args.model)
session = game.Session()
state, reward, reset = session.reset()

if args.self_train:
    match_count = 0
    save_count_down = args.save_interval
    save_time = timeit.default_timer()
    while True:
        agent.update_session(state, reward, reset)
        action = agent.generate_action()
        state, reward, reset = session.do_action(action)
        if reset == 1:
            print(state, state.perspective, action)
            raise ValueError("agent should not generate invalid moves")
        if reset > 2:
            match_count += 1
            save_count_down -= 1
            if save_count_down == 0:
                agent.value_model.save(args.model)
                print("model saved at match", match_count)
                print("time elapsed", timeit.default_timer() - save_time)
                save_time = timeit.default_timer()
                save_count_down = args.save_interval
                for x, y in zip(agent.x_train, agent.y_train):
                    print(x[0], y)
                continue

                for i, pdf in enumerate(agent.y_train):
                    plt.plot(pdf, label=str(i))
                plt.legend()
                plt.show()
else:
    while True:
        agent.update_session(state, reward, reset)
        action = agent.generate_action(explore=True)
        state, reward, reset = session.do_action(action)
        for action, _, action_reward, _, value_pdf, value_sample in zip(
                *agent.action_prediction_history[-1], agent.value_samples):
            print(action, action_reward, value_pdf, value_sample)
        if reset > 1:
            print(state.flip())
            if reward == 0:
                print("tie", reward)
            elif reward > 0:
                print("agent wins", reward)
            elif reward < 0:
                print("agent loses", reward)
        agent.update_session(state, reward, reset)
        while True:
            print('\n', session.state)
            try:
                move_index = tuple([
                    int(token) - 1 for token in input(
                        'your turn, enter "row col": ').split(' ')
                ])
            except ValueError:
                print("integer parsing error")
                continue
            if len(move_index) != 2:
                print("invalid index dimension")
                continue
            if (move_index[0] > game.board_size[0] or move_index[0] < 0
                    or move_index[1] > game.board_size[1] or move_index[1] < 0):
                print("invalid index range")
                continue
            state, reward, reset = session.do_action(move_index)
            if reset == 1:
                print("already occupied")
                continue
            else:
                break
        if reset > 1:
            if reward == 0:
                print("tie", reward)
            elif reward > 0:
                print("you win", reward)
            elif reward < 0:
                print("you lose", reward)

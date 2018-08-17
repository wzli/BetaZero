#usr/bin/python3
games = ['go', 'tic_tac_toe']

import argparse
parser = argparse.ArgumentParser(description='BetaZero App')
parser.add_argument('-g', "--game", choices=games, default='tic_tac_toe')
parser.add_argument(
    '-s', '--self-train', action="store_true", help='self training mode')
parser.add_argument('-m', '--model', help='path to the hdf5 model file')
parser.add_argument(
    '-i',
    '--save-interval',
    type=int,
    default=1000,
    help='save model every i matches')
args = parser.parse_args()

import timeit
import numpy as np
from betazero import ai
from betazero.utils import expected_value, parse_grid_input

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
                for i, (x, y) in enumerate(zip(agent.x_train, agent.y_train)):
                    expected, variance = expected_value(y, True)
                    expected = round(2 * game.max_value * (expected - 0.5), 3)
                    deviation = round(2 * game.max_value * (variance**0.5), 3)
                    print(x[0], "expected value", expected, "deviation",
                          deviation, "turn", i + 1)
                print("model saved at match", match_count)
                print("time elapsed", timeit.default_timer() - save_time)
                save_time = timeit.default_timer()
                save_count_down = args.save_interval
else:
    while True:
        agent.update_session(state, reward, reset)
        action = agent.generate_action(explore=True)
        state, reward, reset = session.do_action(action)

        for action_choice, _, action_reward, _, value_pdf, value_sample in sorted(zip(
                *agent.action_prediction_history[-1], agent.value_samples), key = lambda x: x[-1]):
            expected, variance = expected_value(value_pdf, True)
            expected = round(expected, 3)
            deviation = round(variance**0.5, 3)
            print('A:', action_choice, '\tR:', action_reward, '\tS:',
                  value_sample, '\tE:', expected, '  D:',
                  deviation)
        print("agent played", [i + 1 for i in action])
        if reset > 1:
            print(state.flip())
            if reward == 0:
                print("tie")
            elif reward > 0:
                print("agent wins, score", reward)
            elif reward < 0:
                print("agent loses, score", reward)
        agent.update_session(state, reward, reset)
        while True:
            print('\n', session.state)
            move_index = parse_grid_input(game.board_size)
            state, reward, reset = session.do_action(move_index)
            if reset == 1:
                print("INVALID ACTION")
                continue
            else:
                break
        if reset > 1:
            if reward == 0:
                print("tie")
            elif reward > 0:
                print("you win score", reward)
            elif reward < 0:
                print("you lose score", reward)

#usr/bin/python3
games = ['go', 'chinese_chess', 'tic_tac_toe']

import argparse
parser = argparse.ArgumentParser(description='BetaZero App')
parser.add_argument('-g', "--game", choices=games, default='tic_tac_toe')
parser.add_argument(
    '-s', '--self-train', action="store_true", help='self training mode')
parser.add_argument('-m', '--model', help='path to the hdf5 model file')
parser.add_argument(
    '-a', '--adversary', help='path to the adversary hdf5 model file')
parser.add_argument(
    '-i',
    '--save-interval',
    type=int,
    default=1000,
    help='save model every i matches')
parser.add_argument(
    '-p',
    '--plot',
    action="store_true",
    help='generate value distribution plots')
args = parser.parse_args()

import timeit
import numpy as np
import matplotlib
from betazero import ai
from betazero.utils import expected_value, parse_grid_input

if args.plot:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

print("seleted game:", args.game)
if args.game == games[0]:
    from betazero import go as game
elif args.game == games[1]:
    from betazero import chinese_chess as game
elif args.game == games[2]:
    from betazero import tic_tac_toe as game

if not args.model:
    args.model = args.game + "_model.h5"

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

agent = ai.Agent(game, args.model)
if args.adversary:
    adversary_agent = ai.Agent(game, args.adversary)
session = game.Session()
state, reward, reset = session.reset()

if args.self_train:
    match_count = 0
    save_count_down = args.save_interval
    save_time = timeit.default_timer()
    adversary_turn = False
    wins = 0
    ties = 0
    while True:
        agent.update_session(state, reward, reset)
        if reset == 1:
            raise ValueError("agent should not generate invalid moves")
        if reset > 2:
            if reward == 0:
                ties += 1
            elif reward > 0 and adversary_turn:
                wins += 1
            elif reward < 0 and not adversary_turn:
                wins += 1
            match_count += 1
            save_count_down -= 1
            if save_count_down == 0:
                agent.value_model.save(args.model)
                if args.adversary:
                    adversary_agent.value_model.save(args.adversary)
                for i, (x, y) in enumerate(zip(agent.x_train, agent.y_train)):
                    expected, variance = expected_value(y, True)
                    expected = round(2 * game.max_value * (expected - 0.5), 3)
                    deviation = round(2 * game.max_value * (variance**0.5), 3)
                    print('\n', x, "\nexpected value", expected, "deviation",
                          deviation, "turn", i + 1)
                    if args.plot:
                        plt.plot(y, label=i)
                print("\nmodel saved at match", match_count)
                print("time elapsed", timeit.default_timer() - save_time)
                if args.adversary:
                    print("win rate", wins / match_count, "tie rate",
                          ties / match_count)
                save_count_down = args.save_interval
                if args.plot:
                    plt.savefig(args.game + "_match_" + str(match_count) +
                                "_value_pdf.png")
                    plt.clf()
                save_time = timeit.default_timer()
        if args.adversary:
            adversary_agent.update_session(state, reward, reset)
        if adversary_turn:
            action = adversary_agent.generate_action()
            adversary_turn = False
        else:
            action = agent.generate_action()
            if args.adversary:
                adversary_turn = True
        state, reward, reset = session.do_action(action)
else:
    while True:
        agent.update_session(state, reward, reset)
        action = agent.generate_action(explore=False)
        state, reward, reset = session.do_action(action)

        for action_choice, _, action_reward, _, value_pdf, value_sample in sorted(
                zip(*agent.action_prediction_history[-1], agent.value_samples),
                key=lambda x: x[-1]):
            expected, variance = expected_value(value_pdf, True)
            expected = round(expected, 3)
            deviation = round(variance**0.5, 3)
            print('A:', action_choice, '\tR:', action_reward, '\tS:',
                  value_sample, '\tE:', expected, '  D:', deviation)
        print("agent played", action)
        agent.update_session(state, reward, reset)
        if reset > 1:
            print('\n', state.flip())
            if reward == 0:
                print("tie")
            elif reward > 0:
                print("agent wins, score", reward)
            elif reward < 0:
                print("agent loses, score", reward)
            state, _, _ = session.reset()
        while True:
            print('\n', state.flip())
            if args.game == games[1]:
                move_index = tuple((parse_grid_input(game.board_size),
                                    parse_grid_input(game.board_size)))
            else:
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

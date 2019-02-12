#!/usr/bin/env python3
import argparse
from betazero import ai, utils


class Player:
    def __init__(self, game, name):
        self.game = game
        self.name = name

    def update_session(self, state, reward, reset, train=False):
        self.state = state.flip()
        self.reset = reset

    def generate_action(self, explore=True, verbose=True):
        print('\n', self.state)
        if self.reset > 1:
            print('\n', self.game.Session().reset()[0])
        print(self.name, "turn:")
        return game.get_human_action()


if __name__ == '__main__':
    # create command line arguments
    games = ['go', 'chinese_chess', 'tic_tac_toe']
    parser = argparse.ArgumentParser(description='BetaZero App')
    parser.add_argument("game", choices=games, help='select game')
    parser.add_argument('-m', '--model', help='path to the hdf5 model file')
    parser.add_argument(
        '-s',
        '--save-interval',
        type=int,
        default=0,
        help='save model every i moves trained, zero disables training')
    parser.add_argument(
        '-a', '--adversary', help='path to the adversary hdf5 model file')
    parser.add_argument(
        '-d',
        '--save-directory',
        default='.',
        help='folder to save model and logs')
    parser.add_argument(
        '-f', '--first-turn', action="store_true", help='player goes first')
    # parse and process command line arguments
    args = parser.parse_args()
    if not args.model:
        args.model = args.game + "_model.h5"
    # load the module corresponding to selected game
    print("seleted game:", args.game)
    if args.game == games[0]:
        from betazero import go as game
    elif args.game == games[1]:
        from betazero import chinese_chess as game
    elif args.game == games[2]:
        from betazero import tic_tac_toe as game
    # logic to decide who is playing
    train = args.save_interval > 0
    if train:
        player1 = ai.Agent(game, "Agent", args.model, args.save_interval,
                           args.save_directory)
        if args.adversary:
            player2 = ai.Agent(game, "Adversary", args.adversary,
                               args.save_interval, args.save_directory)
        else:
            player2 = player1
    else:
        player1 = Player(game, "Player1")
        if args.adversary:
            player2 = Player(game, "Player2")
        else:
            player2 = ai.Agent(game, "Agent", args.model, args.save_interval,
                               args.save_directory)
    # start the game
    utils.Arena(
        game,
        player1,
        player2,
        print_actions=not train,
        explore=train,
        first_turn=1 if args.first_turn else -1)

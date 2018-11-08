#usr/bin/python3
import argparse
from betazero import ai


class Player:
    def __init__(self, game, name):
        self.game = game
        self.name = name

    def update_session(self, state, reward, reset):
        self.state = state.flip()
        self.reset = reset

    def generate_action(self, explore=True, verbose=True):
        print('\n', self.state)
        if self.reset > 1:
            print('\n', self.game.Session().reset()[0])
        print(self.name, "turn:")
        return game.get_human_action()


class Arena:
    def __init__(self, game, player1, player2, verbose=False):
        self.game = game
        self.players = {player1, player2}
        self.scores = {player: 0 for player in self.players}
        self.ties = 0
        self.session = self.game.Session()
        updates = self.session.reset()
        for player in self.players:
            player.update_session(*updates)
        while True:
            for player in self.players:
                self.play_turn(player, verbose)

    def play_turn(self, player, verbose):
        action = player.generate_action(explore=not verbose, verbose=verbose)
        state, reward, reset = self.session.do_action(action)
        while reset == 1:
            print(player.name, ": INVALID ACTION")
            action = player.generate_action(
                explore=not verbose, verbose=verbose)
            state, reward, reset = self.session.do_action(action)
        for each_player in self.players:
            each_player.update_session(state, reward, reset)
        if reset > 1:
            if reward == 0:
                if verbose:
                    print("tie")
                self.ties += 1
            elif reward < 0:
                if verbose:
                    print(player.name, "loses")
                self.scores[player] -= 1
                self.ties -= 1
            elif reward > 0:
                if verbose:
                    print(player.name, "wins")
                self.scores[player] += 1
            if verbose:
                print("total", self.ties + sum(self.scores.values()), "wins",
                      self.scores[player], "ties", self.ties)


if __name__ == '__main__':
    games = ['go', 'chinese_chess', 'tic_tac_toe']
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
        help='save model every i sets')
    parser.add_argument(
        '-d', '--save-directory', help='folder to save model and logs')
    args = parser.parse_args()
    print("seleted game:", args.game)
    if args.game == games[0]:
        from betazero import go as game
    elif args.game == games[1]:
        from betazero import chinese_chess as game
    elif args.game == games[2]:
        from betazero import tic_tac_toe as game
    if not args.model:
        args.model = args.game + "_model.h5"
    if not args.save_directory:
        args.save_directory = args.game + "_models"
    if args.self_train:
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
    Arena(game, player1, player2, verbose=not args.self_train)

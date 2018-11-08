#usr/bin/python3
import argparse
from betazero import ai


class Player:
    def __init__(self, game, name):
        self.is_human = True
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


class Arena:
    def __init__(self, game, player1, player):
        self.game = game
        self.session = self.game.Session()
        self.players = {player1, player2}
        self.has_human = player1.is_human or player2.is_human
        updates = self.session.reset()
        for player in self.players:
            player.update_session(*updates)
        while True:
            for player in self.players:
                self.play_turn(player)

    def play_turn(self, player):
        action = player.generate_action(
            explore=not self.has_human, verbose=self.has_human)
        state, reward, reset = self.session.do_action(action)
        while reset == 1:
            print(player.name, ": INVALID ACTION")
            action = player.generate_action(
                explore=not self.has_human, verbose=self.has_human)
            state, reward, reset = self.session.do_action(action)
        for each_player in self.players:
            each_player.update_session(
                state, reward, reset, train=not self.has_human)
        if self.has_human and reset > 1:
            if reward == 0:
                print("tie")
            elif reward < 0:
                print(player.name, "loses")
            elif reward > 0:
                print(player.name, "wins")


if __name__ == '__main__':
    # create command line arguments
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
        '-d',
        '--save-directory',
        default='.',
        help='folder to save model and logs')
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
    # start the game
    Arena(game, player1, player2)

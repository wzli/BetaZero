#usr/bin/python3
import argparse
from betazero import ai


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


class Arena:
    def __init__(self, game, player1, player2, print_actions=True):
        self.game = game
        self.session = self.game.Session()
        self.players = (
            None,
            player1,
            player2,
        )
        self.unique_players = {player1, player2}
        self.stats = [0, 0, 0]
        self.score = 0
        self.self_train = player1 == player2
        updates = self.session.reset()
        for unique_player in self.unique_players:
            unique_player.update_session(*updates)
        first_turn = 1
        player_index = first_turn
        while True:
            if self.play_turn(player_index, print_actions):
                first_turn *= -1
                player_index = first_turn
            else:
                player_index *= -1

    def play_turn(self, player_index, print_actions):
        player = self.players[player_index]
        action = player.generate_action(
            explore=not print_actions, verbose=print_actions)
        state, reward, reset = self.session.do_action(action)
        while reset == 1:
            print(player.name, ": INVALID ACTION")
            action = player.generate_action(
                explore=not print_actions, verbose=print_actions)
            state, reward, reset = self.session.do_action(action)
        for unique_player in self.unique_players:
            unique_player.update_session(
                state, reward, reset, train=self.self_train)
        if not self.self_train and reset > 1:
            if reward == 0:
                print("tie")
                self.stats[0] += 1
            elif reward < 0:
                print(player.name, "loses")
                self.stats[-player_index] += 1
            elif reward > 0:
                print(player.name, "wins")
                self.stats[player_index] += 1
            self.score += reward * player_index
            print(self.players[1].name, self.stats[1], self.players[-1].name,
                  self.stats[-1], "tie", self.stats[0], "avg",
                  self.score / sum(self.stats))
        return reset > 1


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
        help='save model every i moves trained, zero disables autosave')
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
    Arena(game, player1, player2, print_actions=not args.self_train)

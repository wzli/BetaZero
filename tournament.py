#!/usr/bin/env python3
import argparse, os, yaml
from heapq import heapify, heapreplace
from betazero import ai, utils


def scan_models(directory):
    models = set()
    for path in os.listdir(directory):
        name, ext = os.path.splitext(path)
        if ext == ".h5":
            tokens = name.split("_")
            if tokens[0] == "model" and tokens[1].isdigit():
                models.add(int(tokens[1]))
    return models


class Tournament:
    def __init__(self, game):
        self.game = game
        self.elo_stats = {}
        self.participants = []

    def add_participants(self, model_directory, models):
        # initialize participants
        for model in models:
            try:
                TournamentParticipant(self, model_directory, model)
            except Exception as e:
                print(e)

    def adjust_elo(self, winner, loser, k=30):
        rating_gap = (self.elo_stats.setdefault(winner.id, 0) -
                      self.elo_stats.setdefault(loser.id, 0)) / 400
        p_winner = 1.0 / (1.0 + 10**-rating_gap)
        p_loser = 1.0 - p_winner
        self.elo_stats[winner.id] += k * (1 - p_winner)
        self.elo_stats[loser.id] -= k * p_loser

    def playoff(self, participant1, participant2):
        arena = utils.Arena(
            self.game,
            participant1.agent,
            participant2.agent,
            explore=True,
            print_actions=False,
            matches=self.matches)
        print("")
        if arena.stats[1] > arena.stats[-1]:
            winner = participant1
            loser = participant2
        elif arena.stats[1] < arena.stats[-1]:
            winner = participant2
            loser = participant1
        elif arena.score > 0:
            winner = participant1
            loser = participant2
        else:
            winner = participant2
            loser = participant1
        self.adjust_elo(winner, loser)
        return winner, loser

    def run_once(self, matches):
        self.matches = matches
        # de-throne the champion
        heapreplace(self.participants, self.participants[0])
        # itterate tornament
        heapify(self.participants)

    def eliminate(self, n_remaining):
        if n_remaining >= len(self.participants):
            return []
        eliminated = self.participants[n_remaining:]
        self.participants = self.participants[:n_remaining]
        return eliminated


class TournamentParticipant:
    def __init__(self, tournament, model_directory, agent_id):
        self.tournament = tournament
        self.id = agent_id
        self.agent = ai.Agent(tournament.game,
                              str(agent_id),
                              os.path.join(model_directory,
                                           "model_" + str(agent_id) + ".h5"))
        tournament.participants.append(self)

    def __lt__(self, opponent):
        winner, loser = tournament.playoff(self, opponent)
        # it's a min heap, less is good
        return winner.id == self.id


if __name__ == '__main__':
    # create command line arguments
    games = ['go', 'chinese_chess', 'tic_tac_toe']
    parser = argparse.ArgumentParser(description='BetaZero Tournament App')
    parser.add_argument("game", choices=games, help='select game')
    parser.add_argument(
        '-n',
        "--n-participants",
        default=15,
        type=int,
        help=
        'number of participants in the tornament (excess will be eliminated)')
    parser.add_argument(
        '-m',
        "--matches",
        type=int,
        default=2,
        help='number of matchs per playoff')
    parser.add_argument(
        '-f',
        "--record-file",
        default='record.yaml',
        help='path to the record file')
    parser.add_argument(
        '-d',
        "--model-directory",
        default='.',
        help='directory containing models_[ts].h5 files')
    args = parser.parse_args()
    # load the module corresponding to selected game
    print("seleted game:", args.game)
    if args.game == games[0]:
        from betazero import go as game
    elif args.game == games[1]:
        from betazero import chinese_chess as game
    elif args.game == games[2]:
        from betazero import tic_tac_toe as game
    # create the tournament
    tournament = Tournament(game)
    # read/create record yaml file
    file_mode = 'r+' if os.path.exists(args.record_file) else 'w+'
    with open(args.record_file, file_mode) as record_yaml:
        # load record from yaml file
        record = yaml.load(record_yaml) or {
            "participants": [],
            "eliminated": [],
            "elo_stats": {}
        }
        # load previously saved elo stats
        tournament.elo_stats = record["elo_stats"]
        # load prevously eliminated models
        eliminated_models = set(record["eliminated"])
        # find models matching names "model_[ts].h5"
        models = scan_models(args.model_directory) - eliminated_models
        # initialize participants
        tournament.add_participants(args.model_directory, models)
        while True:
            # print current record
            print("Rankings:")
            for rank, participant in enumerate(record["participants"]):
                print("rank", rank, "\tid", participant, "\telo",
                      record["elo_stats"][participant])
            print("")
            # scan for new participants to add to the tournament
            current_models = scan_models(
                args.model_directory) - eliminated_models
            # add newly scanned models to the tournament
            tournament.add_participants(args.model_directory,
                                        current_models - models)
            models = current_models
            # run the tournament
            tournament.run_once(args.matches)
            # eliminate losers
            eliminated_participants = tournament.eliminate(args.n_participants)
            # parse remaining participants to record
            record["participants"] = [
                participant.id for participant in tournament.participants
            ]
            # parse eliminated participants to record
            eliminated_models.update(
                {participant.id
                 for participant in eliminated_participants})
            record["eliminated"] = list(eliminated_models)
            # record the record
            record_yaml.seek(0)
            record_yaml.truncate()
            yaml.dump(record, record_yaml, default_flow_style=False)

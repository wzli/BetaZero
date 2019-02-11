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
        self.participants = []

    def playoff(self, participant1, participant2):
        arena = utils.Arena(
            self.game,
            participant1.agent,
            participant2.agent,
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
        self.agent = ai.Agent(tournament.game, str(agent_id),
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
            "eliminated": set()
        }
        # find models matching names "model_[ts].h5"
        models = scan_models(args.model_directory)
        # initialize participants
        for model in models - record["eliminated"]:
            try:
                TournamentParticipant(tournament, args.model_directory, model)
            except Exception as e:
                print(e)
        while True:
            # print current record
            print("Rankings:")
            for rank, participant in enumerate(record["participants"]):
                print(rank, "\t", participant)
            for participant in record["eliminated"]:
                print("e\t", participant)
            print("")
            # scan for new participants to add to the tournament
            scanned_models = scan_models(
                args.model_directory) - record["eliminated"]
            for model in scanned_models - models:
                try:
                    TournamentParticipant(tournament, args.model_directory,
                                          model)
                except Exception as e:
                    print(e)
            models = scanned_models
            tournament.run_once(args.matches)
            eliminated_participants = tournament.eliminate(args.n_participants)
            record["participants"] = [
                participant.id for participant in tournament.participants
            ]
            record["eliminated"].update(
                {participant.id
                 for participant in eliminated_participants})
            # record the record
            record_yaml.seek(0)
            record_yaml.truncate()
            yaml.dump(record, record_yaml, default_flow_style=False)

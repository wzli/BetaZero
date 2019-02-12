#!/usr/bin/env python3
import argparse, os, yaml
from heapq import heapify, heappush, heappop
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

    def add_participants(self, model_directory, model_entries,
                         ranking_matches):
        # iterate models
        for model, elo in model_entries:
            try:
                # initialize participants
                participant = TournamentParticipant(self, model_directory,
                                                    model, elo)
                # decide whether to play initial ranking matches
                if ranking_matches == 0:
                    self.participants.append(participant)
                else:
                    self.matches = ranking_matches
                    heappush(self.participants, participant)
            except Exception as e:
                print(e)

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
        return winner, loser

    def run_once(self, matches):
        self.matches = matches
        # de-throne the champion, helps balance the heap
        champion = heappop(self.participants)
        # verify the champion's ability to climb back to the top
        heappush(self.participants, champion)
        # playoff every section of the tournament
        heapify(self.participants)

    def eliminate(self, n_remaining):
        if n_remaining >= len(self.participants):
            return []
        eliminated = self.participants[n_remaining:]
        self.participants = self.participants[:n_remaining]
        return eliminated


def adjust_elo(winner, loser, k=30):
    rating_gap = (winner.elo - loser.elo) / 400
    p_winner = 1.0 / (1.0 + 10**-rating_gap)
    p_loser = 1.0 - p_winner
    winner.elo += k * (1 - p_winner)
    loser.elo -= k * p_loser


class TournamentParticipant:
    def __init__(self, tournament, model_directory, agent_id, elo):
        self.tournament = tournament
        self.id = agent_id
        self.elo = elo
        self.model_directory = model_directory
        self.agent = None

    def create_agent(self):
        self.agent = ai.Agent(self.tournament.game,
                              str(self.id),
                              os.path.join(self.model_directory,
                                           "model_" + str(self.id) + ".h5"))

    def __lt__(self, opponent):
        if not self.agent:
            self.create_agent()
        if not opponent.agent:
            opponent.create_agent()
        winner, loser = tournament.playoff(self, opponent)
        # it's a min heap, less is good
        adjust_elo(winner, loser)
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
            "eliminated": {},
            "participants": [],
        }
        # find models matching names "model_[ts].h5"
        models = scan_models(
            args.model_directory) - record["eliminated"].keys()
        new_models = models - set(model
                                  for model, elo in record["participants"])
        # initialize participants
        model_entries = [
            participant_entry for participant_entry in record["participants"]
            if participant_entry[0] in models
        ]
        tournament.add_participants(args.model_directory, model_entries, 0)
        new_model_entries = [(new_model, 0) for new_model in new_models]
        tournament.add_participants(args.model_directory, new_model_entries,
                                    args.matches)
        while True:
            # print current record
            print("Rankings:")
            for rank, participant in enumerate(tournament.participants):
                print("rank", rank, "\tid", participant.id, "\telo",
                      participant.elo)
            print("")
            # scan for new participants to add to the tournament
            current_models = scan_models(
                args.model_directory) - record["eliminated"].keys()
            # add newly scanned models to the tournament
            model_entries = [(model, 0) for model in current_models - models]
            tournament.add_participants(args.model_directory, model_entries,
                                        args.matches)
            models = current_models
            # run the tournament
            tournament.run_once(args.matches)
            # eliminate losers
            eliminated_participants = tournament.eliminate(args.n_participants)
            # parse remaining participants to record
            record["participants"] = [[
                participant.id, participant.elo
            ] for participant in tournament.participants]
            # parse eliminated participants to record
            record["eliminated"].update({
                participant.id: participant.elo
                for participant in eliminated_participants
            })
            # record the record
            record_yaml.seek(0)
            record_yaml.truncate()
            yaml.dump(record, record_yaml, default_flow_style=False)

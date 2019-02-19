#!/usr/bin/env python3
import argparse, os, yaml
from heapq import heapify
from random import shuffle
from betazero import ai, utils
import traceback


def scan_models(directory):
    models = set()
    for path in os.listdir(directory):
        name, ext = os.path.splitext(path)
        if ext == ".h5":
            tokens = name.split("_")
            if tokens[0] == "model" and tokens[1].isdigit():
                models.add(int(tokens[1]))
    return models


def access_yaml(yaml_file, new_content=None):
    file_mode = 'r+' if os.path.exists(yaml_file) else 'w+'
    with open(yaml_file, file_mode) as record_yaml:
        # load record from yaml file
        content = yaml.load(record_yaml)
        # overwrite the record if new content
        if new_content:
            record_yaml.seek(0)
            record_yaml.truncate()
            yaml.dump(new_content, record_yaml, default_flow_style=False)
        return content


def adjust_elo(winner, loser, k=30):
    rating_gap = (winner.elo - loser.elo) / 400
    p_winner = 1.0 / (1.0 + 10**-rating_gap)
    p_loser = 1.0 - p_winner
    winner.elo += k * (1 - p_winner)
    loser.elo -= k * p_loser


class Tournament:
    def __init__(self, game):
        self.game = game
        self.participants = []

    def add_participants(self, model_directory, model_entries):
        # iterate models
        for model, elo in model_entries:
            # initialize participants
            participant = TournamentParticipant(self, model_directory, model,
                                                elo)
            self.participants.append(participant)

    def playoff(self, participant1, participant2):
        from keras import backend as K
        # create agents, lose on exception
        try:
            agent1 = participant1.create_agent()
        except Exception:
            print("create agent exception", traceback.format_exc())
            K.clear_session()
            return participant2, participant1
        try:
            agent2 = participant2.create_agent()
        except Exception:
            print("create agent2 exception", traceback.format_exc())
            K.clear_session()
            return participant1, participant2
        # create arena
        arena = utils.Arena(self.game, agent1, agent2)
        try:
            # play matches
            arena.play_matches(
                self.matches, explore=False, print_actions=False)
        except Exception as e:
            print("exception during turn ", arena.player_index,
                  traceback.format_exc())
            # if exception was during player1 turn, then player1 loses
            if arena.player_index > 0:
                return participant2, participant1
            else:
                return participant1, participant2
        else:
            # if playoff was successful
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
            results = (winner, loser)
            prev_elos = [x.elo for x in results]
            adjust_elo(*results)
            for participant, prev_elo in zip(results, prev_elos):
                print(participant.id, prev_elo, "->", participant.elo)
            return results
        finally:
            K.clear_session()

    def run_once(self, matches):
        # set number of matches per playoff
        self.matches = matches
        # face-off similar ranking participants (assuming participants are ordered by elo)
        heapify(self.participants)
        # face-off participants randomly
        shuffle(self.participants)
        heapify(self.participants)
        # sort participants in order of elo
        self.participants.sort(key=lambda x: x.elo, reverse=True)

    def eliminate(self, n_remaining):
        if n_remaining >= len(self.participants):
            return []
        eliminated = self.participants[n_remaining:]
        self.participants = self.participants[:n_remaining]
        # normalize elo
        average_elo = sum(
            participant.elo for participant in self.participants) / n_remaining
        for participant in self.participants:
            participant.elo -= average_elo
        return eliminated


class TournamentParticipant:
    def __init__(self, tournament, model_directory, agent_id, elo):
        self.tournament = tournament
        self.id = agent_id
        self.elo = elo
        self.model_directory = model_directory
        self.model_path = os.path.join(self.model_directory,
                                       "model_" + str(self.id) + ".h5")

    def create_agent(self):
        with utils.timeout(60):
            return ai.Agent(self.tournament.game, str(self.id),
                            self.model_path)

    def __lt__(self, opponent):
        # assign playoff as the less than operator
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
    record = access_yaml(args.record_file) or {
        "eliminated": {},
        "participants": []
    }
    # find models matching names "model_[ts].h5"
    models = scan_models(args.model_directory) - record["eliminated"].keys()
    new_models = models - set(participant_record["id"]
                              for participant_record in record["participants"])
    #initialize participants
    model_entries = [(participant_record["id"], participant_record["elo"])
                     for participant_record in record["participants"]
                     if participant_record["id"] in models
                     ] + [(new_model, 0) for new_model in new_models]
    tournament.add_participants(args.model_directory, model_entries)
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
        tournament.add_participants(args.model_directory, model_entries)
        models = current_models
        # run the tournament
        tournament.run_once(args.matches)
        # eliminate losers
        eliminated_participants = tournament.eliminate(args.n_participants)
        # refresh record
        record = access_yaml(args.record_file) or {"eliminated": {}}
        # parse remaining participants to record
        record["participants"] = [{
            "id": participant.id,
            "elo": participant.elo
        } for participant in tournament.participants]
        # parse eliminated participants to record
        record["eliminated"].update({
            participant.id: participant.elo
            for participant in eliminated_participants
        })
        # write new record
        access_yaml(args.record_file, record)

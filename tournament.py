#!/usr/bin/env python3
import argparse, os, yaml
from collections import deque
from betazero import ai, utils

agents = {}


def play_match(game, player1, player2, model_path, matches=2):
    for player in (player1, player2):
        if player not in agents:
            agents[player] = ai.Agent(
                game, str(player),
                os.path.join(model_path, "model_" + str(player) + ".h5"))
    arena = utils.Arena(
        game,
        agents[player1],
        agents[player2],
        print_actions=False,
        matches=matches)
    print("")
    if arena.stats[1] > arena.stats[-1]:
        winner = player1
        loser = player2
    elif arena.stats[1] < arena.stats[-1]:
        winner = player2
        loser = player1
    elif arena.score > 0:
        winner = player1
        loser = player2
    else:
        winner = player2
        loser = player1
    return winner, loser


def adjust_elo(elo_record, winner, loser, k):
    rating_gap = (
        elo_record.setdefault(winner, 0) - elo_record.setdefault(loser, 0)
    ) / 400
    p_winner = 1.0 / (1.0 + 10**-rating_gap)
    p_loser = 1.0 - p_winner
    elo_record[winner] += k * (1 - p_winner)
    elo_record[loser] -= k * p_loser


def scan_models(directory):
    models = set()
    for path in os.listdir(directory):
        name, ext = os.path.splitext(path)
        if ext == ".h5":
            tokens = name.split("_")
            if tokens[0] == "model" and tokens[1].isdigit():
                models.add(int(tokens[1]))
    return models


if __name__ == '__main__':
    # create command line arguments
    games = ['go', 'chinese_chess', 'tic_tac_toe']
    parser = argparse.ArgumentParser(description='BetaZero Tournament App')
    parser.add_argument("game", choices=games, help='select game')
    parser.add_argument(
        '-m',
        "--matches",
        default=2,
        help='number of matchs per elo adjustment')
    parser.add_argument(
        '-r',
        "--elo-record",
        default='elo_record.yaml',
        help='path to the elo record')
    parser.add_argument(
        '-d',
        "--model-directory",
        default='.',
        help='directory containing models_[ts].h5 files')
    parser.add_argument(
        '-k',
        "--elo-k",
        default=30,
        help='tune sensitivity of elo adjustments')
    args = parser.parse_args()
    # load the module corresponding to selected game
    print("seleted game:", args.game)
    if args.game == games[0]:
        from betazero import go as game
    elif args.game == games[1]:
        from betazero import chinese_chess as game
    elif args.game == games[2]:
        from betazero import tic_tac_toe as game
    # read/create elo_record yaml file
    file_mode = 'r+' if os.path.exists(args.elo_record) else 'w+'
    with open(args.elo_record, file_mode) as elo_record_yaml:
        # load stats from yaml file
        elo_record = yaml.load(elo_record_yaml) or {}
        # find models matching names "model_[ts].h5"
        models = scan_models(args.model_directory)
        # see which models have an existing record
        old_models = models & elo_record.keys()
        new_models = models - old_models
        # add old models to queue, sorted by decending elo
        challenger_queue = deque(
            sorted(old_models, key=elo_record.get, reverse=True))
        # pickout highest elo champion
        if old_models:
            champion = challenger_queue.popleft()
        # add new models to front of queue, sorted by decending timestamp
        challenger_queue.extendleft(sorted(new_models, reverse=True))
        # pickout most recent if no models have a record
        if not old_models:
            champion = challenger_queue.popleft()
        # keep playing matches, winner stays on
        while True:
            scanned_models = scan_models(args.model_directory)
            challenger_queue.extendleft(
                sorted(scanned_models - models, reverse=True))
            models = scanned_models
            challenger = challenger_queue.popleft()
            champion, loser = play_match(game, challenger, champion,
                                         args.model_directory, args.matches)
            challenger_queue.append(loser)
            adjust_elo(elo_record, champion, loser, args.elo_k)
            elo_record_yaml.seek(0)
            elo_record_yaml.truncate()
            yaml.dump(elo_record, elo_record_yaml, default_flow_style=False)
        print(elo_record)

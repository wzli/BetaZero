#!/usr/bin/env python3
import argparse, os, yaml
from urllib.request import urlopen
from urllib.error import URLError
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
            # Check the list of defined attributes.
            for name, value in attrs:
                # If href is defined, add it.
                if name == "href":
                    self.links.add(value)


def filter_model_names(file_list):
    models = set()
    for path in file_list:
        name, ext = os.path.splitext(path)
        if ext == ".h5":
            tokens = name.split("_")
            if tokens[0] == "model" and tokens[1].isdigit():
                models.add(int(tokens[1]))
    return models


def fetch_local_models(directory):
    return filter_model_names(os.listdir(directory))


def fetch_eliminated_models(record_file):
    # read record yaml file
    if not record_file:
        return set()
    with open(record_file, 'r') as record_yaml:
        # load record from yaml file
        record = yaml.load(record_yaml)
        if "eliminated" in record:
            return set(record["eliminated"])
    return set()


def fetch_remote_models(url):
    html_response = urlopen(url)
    html = str(html_response.read())
    parser = LinkParser()
    parser.feed(html)
    return filter_model_names(parser.links)


def download_models(url, directory, models):
    for model in models:
        model_file = "model_" + str(model) + ".h5"
        file_url = os.path.join(url, model_file)
        try:
            print("downloading", file_url)
            response = urlopen(file_url, timeout=5)
            with open(os.path.join(directory, model_file), 'wb') as f:
                f.write(response.read())
            break
        except URLError as e:
            print(e)


if __name__ == '__main__':
    # create command line arguments
    parser = argparse.ArgumentParser(description='BetaZero Sync Models App')
    parser.add_argument(
        'remote_url',
        help='url to remote directory containing models_[ts].h5 files')
    parser.add_argument(
        '-d',
        "--model-directory",
        default='.',
        help='directory containing models_[ts].h5 files')
    parser.add_argument('-f', "--record-file", help='path to the record file')
    args = parser.parse_args()

    eliminated_models = fetch_eliminated_models(args.record_file)
    local_models = fetch_local_models(args.model_directory)
    remote_models = fetch_remote_models(args.remote_url)
    new_models = remote_models - local_models - eliminated_models
    print("eliminated models:", eliminated_models)
    print("local models:", local_models)
    print("remote models:", remote_models)
    print("new models:", new_models)

    download_models(args.remote_url, args.model_directory, new_models)

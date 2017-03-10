import json
from spacy.en import English
import re
from collections import Counter

def main():
    parser = English()
    fn = "../train-v1.1.json"
    with open(fn) as fp:
        data = json.load(fp)

    sent_counts = Counter()
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            context = pgraph['context']
            num_sents = len(list(parser(context).sents))
            sent_counts[num_sents] += 1

    print sent_counts


if __name__ == "__main__":
    main()

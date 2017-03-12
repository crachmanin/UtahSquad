import json
from spacy.en import English
from collections import Counter

parser = English()
vocab = set()

def add_to_vocab(data):
# data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            c_parsed = parser(pgraph['context'])
            for token in c_parsed:
                vocab.add(token.orth_.lower())
            for qa in pgraph['qas']:
                q_parsed = parser(qa['question'])
                for token in q_parsed:
                    vocab.add(token.orth_.lower())

fn = "../data/train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

add_to_vocab(data)

fn = "../data/dev-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

add_to_vocab(data)

with open("../data/vocab.json", 'w') as fp:
    fp.write(json.dumps(list(vocab)))

import json
from pprint import pprint

fn = "train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

# data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
with open("answers.txt", 'w') as fp:
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            for qa in pgraph['qas']:
                for ans in qa['answers']:
                    fp.write((ans['text'] + u'\n').encode('utf-8'))

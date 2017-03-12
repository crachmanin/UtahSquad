import json
from spacy.en import English
from collections import Counter

fn = "../data/train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

parser = English()
# data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
context_len = Counter()
question_len = Counter()
with open("contexts.txt", 'w') as fp:
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            c_parsed = parser(pgraph['context'])
            context_len[len(c_parsed)] += 1
            for qa in pgraph['qas']:
                q_parsed = parser(qa['question'])
                question_len[len(q_parsed)] += 1

print context_len
print
print question_len

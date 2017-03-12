import json
from pprint import pprint
from collections import Counter

fn = "train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

counts = Counter()

data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
for topic in data['data']:
    for pgraph in topic['paragraphs']:
        l = len(pgraph['context'].split())
            # for ans in qa['answers']:
                # tokens = ans['text'].split()
        counts[l] += 1

print counts
under_15 = 0.0
total = 0.0
for c in counts:
    if c <= 15:
        under_15 += counts[c]
    total += counts[c]

print under_15/total
minn = min(counts.keys())
print minn
print counts[minn]


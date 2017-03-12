import json
from spacy.en import English
from collections import Counter
import time

fn = "../data/train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

parser = English()
# data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
sent_lens = Counter()
with open("contexts.txt", 'w') as fp:
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            c_parsed = parser(pgraph['context'])
            for sent in c_parsed.sents:
                sent_lens[len(sent)] += 1
                print sent
                print len(sent)
                print
                time.sleep(2)

print sent_lens

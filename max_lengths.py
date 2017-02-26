import json
import nltk
from collections import Counter

fn = "train-v1.1.json"
with open(fn) as fp:
    data = json.load(fp)

# data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
context_len = Counter()
question_len = Counter()
with open("contexts.txt", 'w') as fp:
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            context_tokens = nltk.word_tokenize(pgraph['context'])
            context_len[len(context_tokens)] += 1
            for qa in pgraph['qas']:
                question_tokens = nltk.word_tokenize(qa['question'])
                question_len[len(question_tokens)] += 1

print context_len
print
print question_len

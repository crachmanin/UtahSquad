import json
import nltk
import re
import os
from spacy.en import English

MAX_CLENGTH = 100
MAX_QLENGTH = 30
TRAIN_FILE = "train-v1.1.json"
DATA_DIR = "../../data"

def find_token(answer_start, token_dict, answer_text):
    closest_below = max([key for key in token_dict.keys() if key < answer_start])
    closest_above = min([key for key in token_dict.keys() if key > answer_start])
    return closest_below, closest_above

def main():
    parser = English()
    fn = os.path.join(DATA_DIR, TRAIN_FILE)
    with open(fn) as fp:
        data = json.load(fp)

    result = []

    # data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    bad_count = 0
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            token_dict = {}
            context = pgraph['context']
            c_parsed = parser(context)
            sents = list(c_parsed.sents)
            for sent_num, sent in enumerate(sents):
                for token in sent:
                    token_dict[token.idx] = (token, sent_num, token.i - sent.start)

            for qa in pgraph['qas']:
                question = qa['question']
                q_parsed = parser(question)
                q_tokens = [token.orth_ for token in q_parsed]
                # e_question = str(question).encode('utf-8')
                for ans in qa['answers']:
                    answer_start = ans['answer_start']
                    answer_text = ans['text']
                    a_parsed = parser(answer_text)
                    a_tokens = [token.orth_ for token in a_parsed]
                    if answer_start in token_dict:
                        token, sent_num, answer_idx = token_dict[answer_start]
                        sent_tokens = [token.orth_ for token in sents[sent_num]]
                        end_idx = answer_idx + len(a_tokens) - 1
                        if end_idx >= len(sent_tokens) or sent_tokens[end_idx] != a_tokens[-1] or len(a_tokens) != 1:
                            # print str((c_tokens[end_idx], a_tokens[-1])).encode('utf-8')
                            # print str(c_tokens[answer_idx:answer_idx+ 5]).encode('utf-8')
                            # print str(a_tokens).encode('utf-8')
                            # print
                            bad_count += 1
                            continue
                        # line = [e_context, e_question, str(answer_idx)]
                        if answer_idx <= MAX_CLENGTH:
                            line = [sent_tokens, q_tokens, str(answer_idx), str(end_idx)]
                            result.append(line)

    out_fn = os.path.join(DATA_DIR, "sent-train-preproc.json")
    with open(out_fn, 'w') as train_fp:
        train_fp.write(json.dumps(result))
    print bad_count


if __name__ == "__main__":
    main()

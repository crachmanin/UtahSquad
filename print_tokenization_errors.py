import json
import nltk
import re

def get_token_idxs(text, tokens):
    running_offset = 0
    token_dict = {}
    token_list = []
    for i, orig_token in enumerate(tokens):
        token = re.sub("``|''", '\"', orig_token)
        try:
            offset = text.index(token, running_offset)
        except:
            offset = text.index(orig_token, running_offset)
        token_dict[offset] = (i, token, offset)
        token_list.append((token, offset))
        running_offset = offset + len(token)
    # print token_list
    return token_dict


def find_token(answer_start, token_dict, answer_text):
    closest_below = max([key for key in token_dict.keys() if key < answer_start])
    closest_above = min([key for key in token_dict.keys() if key > answer_start])
    return closest_below, closest_above

def main():
    fn = "train-v1.1.json"
    with open(fn) as fp:
        data = json.load(fp)

    # data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    bad_answers = 0
    bad_answers2 = 0
    total_answers = 0
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            context = pgraph['context']
            tokens = nltk.word_tokenize(context)
            token_dict = get_token_idxs(context, tokens)
            for qa in pgraph['qas']:
                question = qa['question']
                for ans in qa['answers']:
                    answer_start = ans['answer_start']
                    answer_text = ans['text']
                    # print answer_text
                    total_answers += 1
                    if answer_start in token_dict:
                        answer_idx = token_dict[answer_start]
                    else:
                        answer_idx = find_token(answer_start, token_dict, answer_text)

                        print question.encode('utf-8')
                        print answer_text.encode('utf-8')
                        print context[answer_start: answer_start + 20].encode('utf-8')
                        print context[answer_start - 20: answer_start + 20].encode('utf-8')
                        print str(token_dict[answer_idx[0]]).encode('utf-8')
                        print str(token_dict[answer_idx[1]]).encode('utf-8')
                        print

                        bad_answers += 1
                        if not answer_idx:
                            bad_answers2 += 1
                            continue
    print bad_answers2
    print bad_answers, total_answers
    print 1 - float(bad_answers)/total_answers

if __name__ == "__main__":
    main()

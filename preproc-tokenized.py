import json
import nltk
import re

MAX_CLENGTH = 400

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

    result = []

    # data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    bad_count = 0
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            context = pgraph['context']
            c_tokens = nltk.word_tokenize(context)[:MAX_CLENGTH]
            # e_context = str(c_tokens).encode('utf-8')
            token_dict = get_token_idxs(context, c_tokens)
            for qa in pgraph['qas']:
                question = qa['question']
                q_tokens = nltk.word_tokenize(question)
                # e_question = str(question).encode('utf-8')
                for ans in qa['answers']:
                    answer_start = ans['answer_start']
                    answer_text = ans['text']
                    a_tokens = nltk.word_tokenize(answer_text)
                    if answer_start in token_dict:
                        answer_idx = token_dict[answer_start][0]
                        end_idx = answer_idx + len(a_tokens) - 1
                        if end_idx >= len(c_tokens) or c_tokens[end_idx] != a_tokens[-1]:
                            # print str((c_tokens[end_idx], a_tokens[-1])).encode('utf-8')
                            # print str(c_tokens[answer_idx:answer_idx+ 5]).encode('utf-8')
                            # print str(a_tokens).encode('utf-8')
                            # print
                            bad_count += 1
                            continue
                        # line = [e_context, e_question, str(answer_idx)]
                        if answer_idx <= MAX_CLENGTH:
                            line = [c_tokens, q_tokens, str(answer_idx), str(end_idx)]
                            result.append(line)

    with open("train-preproc-tokenized.json", 'w') as train_fp:
        train_fp.write(json.dumps(result))
    print bad_count


if __name__ == "__main__":
    main()

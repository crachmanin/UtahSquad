import json
from spacy.en import English
import re
import math
from collections import Counter
import numpy as np

MAX_SENTENCES = 20
word_idf_counter = Counter()
bigram_idf_counter = Counter()
parser = English()
total_passages = 0
max_word_tf_idf = 0
max_bigram_tf_idf = 0

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


def bigrams(words):
    return ['_'.join(pair) for pair in zip(words[:-1], words[1:])]


def get_idfs(data):
    global word_idf_counter
    global bigram_idf_counter
    global total_passages
    global parser
    # for topic in data['data']:
    topic = data
    for i in xrange(1):
        for pgraph in topic['paragraphs']:
            total_passages += 1
            context = pgraph['context']
            c_parsed = parser(context)
            pgraph_words = [token.orth_ for token in c_parsed]
            pgraph_bigrams = bigrams(pgraph_words)
            unique_pgraph_words = set(pgraph_words)
            unique_pgraph_bigrams = set(pgraph_bigrams)
            for word in unique_pgraph_words:
                word_idf_counter[word] += 1
            for bigram in unique_pgraph_bigrams:
                bigram_idf_counter[bigram] += 1


def word_tf_idf(word, word_tf_counter, total_words):
    global word_idf_counter
    global total_passages
    tf = word_tf_counter[word]/float(total_words)
    idf = math.log(total_passages/float(word_idf_counter[word]))
    return tf * idf


def bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams):
    global bigram_idf_counter
    global total_passages
    tf = bigram_tf_counter[bigram]/float(total_bigrams)
    idf = math.log(total_passages/float(bigram_idf_counter[bigram]))
    return tf * idf


def max_tf_idfs(data):
    global parser
    global max_word_tf_idf
    global max_bigram_tf_idf

    # for topic in data['data']:
    topic = data
    for i in xrange(1):
        for pgraph in topic['paragraphs']:
            context = pgraph['context']
            c_parsed = parser(context)
            total_words = len(c_parsed)
            total_bigrams = total_words - 1

            for qa in pgraph['qas']:
                question = qa['question']
                q_parsed = parser(question)
                question_words = [token.orth_ for token in q_parsed]
                question_bigrams = bigrams(question_words)
                unique_question_words = set(question_words)
                unique_question_bigrams = set(question_bigrams)
                # e_question = str(question).encode('utf-8')

                for sent_num, sent in enumerate(c_parsed.sents):
                    root = sent.root
                    sent_words = [token.orth_ for token in sent]
                    sent_bigrams = bigrams(sent_words)
                    word_tf_counter = Counter(sent_words)
                    bigram_tf_counter = Counter(sent_bigrams)
                    sent_word_tf_idf = sum(word_tf_idf(word, word_tf_counter, total_words)
                                            for word in sent_words if word in question_words)
                    max_word_tf_idf = max(max_word_tf_idf, sent_word_tf_idf)
                    sent_bigram_tf_idf = sum(bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams)
                                            for bigram in sent_bigrams if bigram in question_bigrams)
                    max_bigram_tf_idf = max(max_bigram_tf_idf, sent_bigram_tf_idf)


def word_tf_idf_feat_num(sent_num, tf_idf, max_tf_idf):
    word_tf_idf_divisor = max_word_tf_idf/10.0
    print max_word_tf_idf
    print tf_idf
    print math.floor(tf_idf/word_tf_idf_divisor)
    print
    return math.floor(tf_idf/word_tf_idf_divisor)


def bigram_tf_idf_feat_num(sent_num, tf_idf):
    global max_bigram_tf_idf
    bigram_tf_idf_divisor = max_bigram_tf_idf/10.0
    return MAX_SENTENCES + math.floor(tf_idf/bigram_tf_idf_divisor)


def root_match_feat_num(sent_num, match):
    return MAX_SENTENCES * 2 + sent_num


def generate_features(data):
    result = []
    global parser
    # for topic in data['data']:
    topic = data
    for i in xrange(1):
        for pgraph in topic['paragraphs']:
            token_dict = {}
            context = pgraph['context']
            c_parsed = parser(context)
            total_words = len(c_parsed)
            total_bigrams = total_words - 1

            for sent_num, sent in enumerate(c_parsed.sents):
                for token in sent:
                    token_dict[token.idx] = (token, sent_num, token.i)

            for qa in pgraph['qas']:
                question = qa['question']
                q_parsed = parser(question)
                question_root = list(q_parsed.sents)[0].root
                question_words = [token.orth_ for token in q_parsed]
                question_bigrams = bigrams(question_words)
                unique_question_words = set(question_words)
                unique_question_bigrams = set(question_bigrams)
                # e_question = str(question).encode('utf-8')

                for ans in qa['answers']:
                    answer_start = ans['answer_start']
                    answer_text = ans['text']

                    sent_word_tf_idfs = []
                    sent_bigram_tf_idfs = []
                    features = []
                    for sent_num, sent in enumerate(c_parsed.sents):
                        if sent_num >= MAX_SENTENCES:
                            break
                        root_match = sent.root == question_root
                        sent_words = [token.orth_ for token in sent]
                        sent_bigrams = bigrams(sent_words)
                        word_tf_counter = Counter(sent_words)
                        bigram_tf_counter = Counter(sent_bigrams)
                        sent_word_tf_idf = sum(word_tf_idf(word, word_tf_counter, total_words)
                                               for word in sent_words if word in question_words)
                        sent_word_tf_idfs.append(sent_word_tf_idf)
                        sent_bigram_tf_idf = sum(bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams)
                                               for bigram in sent_bigrams if bigram in question_bigrams)
                        sent_bigram_tf_idfs.append(sent_bigram_tf_idf)

                        features.append(word_tf_idf_feat_num(sent_num, sent_word_tf_idf))
                        features.append(bigram_tf_idf_feat_num(sent_num, sent_bigram_tf_idf))
                        if root_match:
                            features.append(root_match_feat_num(sent_num))

                    if answer_start in token_dict:
                        # change this for sentence vs word answers
                        # answer_idx = token_dict[answer_start][0]
                        answer_idx = token_dict[answer_start][1]
                        if answer_idx < MAX_SENTENCES:
                            libsvm_features = ["%d:1" % feat_num for feat_num in features]
                            libsvm_features = " ".join(libsvm_features)
                            result.append("%d " % answer_idx + libsvm_features)
    return result

def main():
    result = []

    fn = "../train-v1.1.json"
    with open(fn) as fp:
        data = json.load(fp)

    data = data['data'][0]

    get_idfs(data)
    max_tf_idfs(data)
    features = generate_features(data)

    with open("features.txt", 'w') as train_fp:
        for line in features:
            train_fp.write(line + '\n')


if __name__ == "__main__":
    main()

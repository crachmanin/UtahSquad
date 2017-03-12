import json
from spacy.en import English
import re
import math
from collections import Counter
import numpy as np
import sys

MAX_SENTENCES = 20
word_idf_counter = Counter()
bigram_idf_counter = Counter()
parser = English()
total_passages = 0
_DEV = False

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
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            total_passages += 1
            context = pgraph['context']
            c_parsed = parser(context)
            pgraph_words = [token.orth_.lower() for token in c_parsed]
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


def word_tf_idf_feat_num(sent_num, tf_idf, max_tf_idf):
    tf_idf_divisor = max_tf_idf/10.0
    if max_tf_idf != 0:
        value = max(0, math.floor(tf_idf/tf_idf_divisor) - 1)
    else:
        value = 0
    return MAX_SENTENCES * 0 + sent_num * 10 + value


def bigram_tf_idf_feat_num(sent_num, tf_idf, max_tf_idf):
    tf_idf_divisor = max_tf_idf/10.0
    if max_tf_idf != 0:
        value = max(0, math.floor(tf_idf/tf_idf_divisor) - 1)
    else:
        value = 0
    return MAX_SENTENCES * 10 + sent_num * 10 + value


def sent_sim_to_feat_num(sent_num, sent_sim, max_sim):
    sim_divisor = max_sim/10.0
    if max_sim != 0:
        value = max(0, math.floor(sent_sim/sim_divisor) - 1)
    else:
        value = 0
    return MAX_SENTENCES * 20 + sent_num * 10 + value


def word_wise_sent_sim_to_feat_num(sent_num, sent_sim, max_sim):
    sim_divisor = max_sim/10.0
    if max_sim != 0:
        value = max(0, math.floor(sent_sim/sim_divisor) - 1)
    else:
        value = 0
    return MAX_SENTENCES * 30 + sent_num * 10 + value


def root_match_feat_num(sent_num):
    return MAX_SENTENCES * 40 + MAX_SENTENCES * 0 + sent_num


def root_match_lemma_feat_num(sent_num):
    return MAX_SENTENCES * 40 + MAX_SENTENCES * 1 + sent_num


def sent_contains_quest_root_feat_num(sent_num):
    return MAX_SENTENCES * 40 + MAX_SENTENCES * 2 + sent_num


def quest_contains_sent_root_feat_num(sent_num):
    return MAX_SENTENCES * 40 + MAX_SENTENCES * 3 + sent_num


def year_in_sent_feat_num(sent_num):
    return MAX_SENTENCES * 50 + MAX_SENTENCES * 0 + sent_num


def date_in_sent_feat_num(sent_num):
    return MAX_SENTENCES * 50 + MAX_SENTENCES * 1 + sent_num


def person_in_sent_feat_num(sent_num):
    return MAX_SENTENCES * 50 + MAX_SENTENCES * 2 + sent_num


def location_in_sent_feat_num(sent_num):
    return MAX_SENTENCES * 50 + MAX_SENTENCES * 3 + sent_num


def number_in_sent_feat_num(sent_num):
    return MAX_SENTENCES * 50 + MAX_SENTENCES * 4 + sent_num


def year_in_question_feat_num():
    return MAX_SENTENCES * 60 + 0


def when_in_question_feat_num():
    return MAX_SENTENCES * 60 + 1


def who_in_question_feat_num():
    return MAX_SENTENCES * 60 + 2


def where_in_question_feat_num():
    return MAX_SENTENCES * 60 + 3


def many_in_question_feat_num():
    return MAX_SENTENCES * 60 + 4


def sent_entity_features(sent, sent_num, features):

    year_in_sent = False
    date_in_sent = False
    person_in_sent = False
    location_in_sent = False
    number_in_sent = False

    for word in sent:
        if word.is_digit and len(word.orth_) == 4:
            year_in_sent = True
        if word.ent_type_ == "DATE":
            date_in_sent = True
        if word.ent_type_ == "PERSON":
            person_in_sent = True
        if word.ent_type_ == "LOCATION":
            location_in_sent = True
        if word.like_num:
            number_in_sent = True

    if year_in_sent:
        features.append(year_in_sent_feat_num(sent_num))
    if date_in_sent:
        features.append(date_in_sent_feat_num(sent_num))
    if person_in_sent:
        features.append(person_in_sent_feat_num(sent_num))
    if location_in_sent:
        features.append(location_in_sent_feat_num(sent_num))
    if number_in_sent:
        features.append(number_in_sent_feat_num(sent_num))


def quest_wh_word_features(unique_question_words, features):
    year_in_question = "year" in unique_question_words
    when_in_question = "when" in unique_question_words
    who_in_question = "who" in unique_question_words
    where_in_question = "where" in unique_question_words
    many_in_question = "many" in unique_question_words

    if year_in_question:
        features.append(year_in_question_feat_num())
    if when_in_question:
        features.append(when_in_question_feat_num())
    if who_in_question:
        features.append(who_in_question_feat_num())
    if where_in_question:
        features.append(where_in_question_feat_num())
    if many_in_question:
        features.append(many_in_question_feat_num())


def root_features(sent, sent_num, question_root, unique_question_words, features):

    sent_words = set([token.orth_.lower() for token in sent])
    root_match = sent.root.orth_.lower() == question_root.orth_.lower()
    root_match_lemma = sent.root.lemma == question_root.lemma
    if root_match:
        features.append(root_match_feat_num(sent_num))
    if root_match_lemma:
        features.append(root_match_lemma_feat_num(sent_num))

    if question_root.orth_ in sent_words:
        features.append(sent_contains_quest_root_feat_num(sent_num))
    if sent.root.orth_ in unique_question_words:
        features.append(quest_contains_sent_root_feat_num(sent_num))


def sent_quest_sim_features(c_parsed, q_parsed, features):
    sent_sims = []
    word_wise_sent_sims = []
    for sent_num, sent in enumerate(c_parsed.sents):
        if sent_num >= MAX_SENTENCES:
            break

        sent_sims.append(sent.similarity(q_parsed))
        word_wise_sent_sim = 0
        for word in sent:
            for q_word in q_parsed:
                word_wise_sent_sim += word.similarity(q_word)
        word_wise_sent_sims.append(word_wise_sent_sim)

    max_sent_sim = max(sent_sims)
    max_word_wise_sent_sim = max(word_wise_sent_sims)

    for sent_num, sent_sim in enumerate(sent_sims):
        features.append(sent_sim_to_feat_num(sent_num, sent_sim, max_sent_sim))
    for sent_num, sent_sim in enumerate(word_wise_sent_sims):
        features.append(word_wise_sent_sim_to_feat_num(sent_num, sent_sim, max_word_wise_sent_sim))


def sent_tf_idf_features(c_parsed, unique_question_words, unique_question_bigrams, features):
    total_words = len(c_parsed)
    total_bigrams = total_words - 1

    sent_word_tf_idfs = []
    sent_bigram_tf_idfs = []

    for sent_num, sent in enumerate(c_parsed.sents):
        if sent_num >= MAX_SENTENCES:
            break

        sent_words = [token.orth_.lower() for token in sent]
        sent_bigrams = bigrams(sent_words)
        unique_sent_words = set(sent_words)
        unique_sent_bigrams = set(sent_bigrams)

        word_tf_counter = Counter(sent_words)
        bigram_tf_counter = Counter(sent_bigrams)
        # should be iterating through unique sent_words but reduces accuracy
        sent_word_tf_idf = sum(word_tf_idf(word, word_tf_counter, total_words)
                                for word in unique_sent_words if word in unique_question_words)
        sent_word_tf_idfs.append(sent_word_tf_idf)
        sent_bigram_tf_idf = sum(bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams)
                                for bigram in unique_sent_bigrams if bigram in unique_question_bigrams)
        sent_bigram_tf_idfs.append(sent_bigram_tf_idf)

    max_word_tf_idf = max(sent_word_tf_idfs)
    max_bigram_tf_idf = max(sent_bigram_tf_idfs)

    for sent_num, tf_idf in enumerate(sent_word_tf_idfs):
        features.append(word_tf_idf_feat_num(sent_num, tf_idf, max_word_tf_idf))
    for sent_num, tf_idf in enumerate(sent_bigram_tf_idfs):
        features.append(bigram_tf_idf_feat_num(sent_num, tf_idf, max_bigram_tf_idf))


def generate_features(data):
    result = []
    global parser
    for topic in data['data']:
        for pgraph in topic['paragraphs']:
            token_dict = {}
            context = pgraph['context']
            c_parsed = parser(context)

            for sent_num, sent in enumerate(c_parsed.sents):
                for token in sent:
                    token_dict[token.idx] = (token, sent_num, token.i)

            for qa in pgraph['qas']:
                question = qa['question']
                q_parsed = parser(question)
                question_root = list(q_parsed.sents)[0].root
                question_words = [token.orth_.lower() for token in q_parsed]
                question_bigrams = bigrams(question_words)
                unique_question_words = set(question_words)
                unique_question_bigrams = set(question_bigrams)
                # e_question = str(question).encode('utf-8')

                for ans in qa['answers']:
                    answer_start = ans['answer_start']
                    answer_text = ans['text']

                    features = []
                    sent_quest_sim_features(c_parsed, q_parsed, features)
                    sent_tf_idf_features(c_parsed, unique_question_words,
                                         unique_question_bigrams, features)
                    quest_wh_word_features(unique_question_words, features)

                    for sent_num, sent in enumerate(c_parsed.sents):
                        if sent_num >= MAX_SENTENCES:
                            break

                        sent_entity_features(sent, sent_num, features)
                        # root_features(sent, sent_num, question_root,
                                      # unique_question_words, features)

                    if answer_start in token_dict:
                        # change this for sentence vs word answers
                        # answer_idx = token_dict[answer_start][0]
                        answer_idx = token_dict[answer_start][1]
                        if answer_idx < MAX_SENTENCES:
                            libsvm_features = ["%d:1" % feat_num for feat_num in sorted(features)]
                            libsvm_features = " ".join(libsvm_features)
                            result.append("%d " % answer_idx + libsvm_features)
    return result

def main():
    global _DEV
    result = []

    fn = "../train-v1.1.json"
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        _DEV = True
        fn = "../dev-v1.1.json"

    with open(fn) as fp:
        data = json.load(fp)

    data['data'] = data['data'][:100]

    get_idfs(data)
    features = generate_features(data)

    out_fn = "ablated_features.txt"
    if _DEV:
        out_fn = "dev_features.txt"
    with open(out_fn, 'w') as fp:
        for line in features:
            fp.write(line + '\n')


if __name__ == "__main__":
    main()


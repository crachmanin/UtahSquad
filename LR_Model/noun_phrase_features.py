import json
from spacy.en import English
import re
import math
from collections import Counter
import sys
import os
from tqdm import tqdm

word_idf_counter = Counter()
bigram_idf_counter = Counter()
parser = English()
total_passages = 0

DATA_DIR = "../data/"
_TEST = False

FEATURE_DICT = {
    "UNK_SENT_SIM":  0,
    "UNK_WW_SENT_SIM":  1,
    "UNK_NP_SIM":  2,
    "UNK_WW_NP_SIM":  3,
    "UNK_WORD_SENT_TF_IDF": 4,
    "UNK_BIGRAM_SENT_TF_IDF": 5
}


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
            c_parsed = parser(context, parse=False)
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
    if word not in word_tf_counter or word not in word_idf_counter \
            or not word_tf_counter[word] or not word_idf_counter[word]:
        return 0
    tf = word_tf_counter[word]/float(total_words)
    idf = math.log(total_passages/float(word_idf_counter[word]))
    return tf * idf


def bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams):
    global bigram_idf_counter
    global total_passages
    if bigram not in bigram_tf_counter or bigram not in bigram_idf_counter \
            or not bigram_tf_counter[bigram] or not bigram_idf_counter[bigram]:
        return 0
    tf = bigram_tf_counter[bigram]/float(total_bigrams)
    idf = math.log(total_passages/float(bigram_idf_counter[bigram]))
    return tf * idf


def get_feat_num(feat_str):
    global FEATURE_DICT

    if feat_str not in FEATURE_DICT:
        if _TEST:
            return None
        FEATURE_DICT[feat_str] = len(FEATURE_DICT)

    return FEATURE_DICT[feat_str]


def normalize_list(l1):
    result = []
    max_val = max(l1)
    divisor = max_val/10.0
    for unnorm in l1:
        if max_val != 0:
            value = max(0, math.floor(unnorm/divisor) - 1)
        else:
            value = 0
        result.append(value)
    return result


def wh_ent_features(np, q_parsed, features):
    for i, q_word in enumerate(q_parsed[:3]):
        q_word = q_word.orth_
        features.append(get_feat_num("Q_WORD_NP_ENT_%d_%s_%s" % (i, q_word, np.label_)))


def span_pos_features(np, q_parsed, features):
    for i, q_word in enumerate(q_parsed[:3]):
        q_word = q_word.orth_
        for j, token in enumerate(list(np)):
            features.append(get_feat_num("Q_WORD_NP_POS_%d_%d_%s_%s" %
                                         (i, j, q_word, token.pos_)))


def lexicalized_lemma_features(np, q_parsed):
    result = set()
    for q_word in q_parsed:
        for i, np_token in enumerate(np):
            result.add(get_feat_num("LEX_LEMMA_%s_%s" %
                                         (q_word.lemma_, np_token.lemma_)))
            head1 = np.root.head.lemma_
            dep1 = np.root.dep_
            head2 = np.root.head.head.lemma_
            dep2 = np.root.head.dep_
            result.add(get_feat_num("LEX_LEMMA_DEP1%s_%s->%s" %
                                         (q_word.lemma_, head1, dep1)))
            result.add(get_feat_num("LEX_LEMMA_DEP2%s_%s->%s" %
                                         (q_word.lemma_, head2, dep2)))

    return result


def pos_path_to_root(token):
    result = []
    curr = token
    while True:
        result.append("%s_%s" % (curr.pos_, curr.dep_))
        if str(curr.dep_) == "ROOT":
            return "->".join(result)
        curr = curr.head


def ne_path_to_root(token):
    result = []
    curr = token
    while True:
        result.append("%s_%s" % (curr.ent_type_, curr.dep_))
        if str(curr.dep_) == "ROOT":
            return "->".join(result)
        curr = curr.head


def dependency_path_features(np, sent_num, sents, unique_question_words, q_parsed):
    result = set()
    right_pos_path = pos_path_to_root(np.root)
    right_ne_path = ne_path_to_root(np.root)
    for i, q_word in enumerate(q_parsed[:3]):
        q_word = q_word.orth_
        for word in sents[sent_num]:
            if word.orth_.lower() in unique_question_words:
                left_pos_path = pos_path_to_root(word)
                left_ne_path = ne_path_to_root(word)
                result.add(get_feat_num("POS_DEP_PATH_%d_%s_%s_%s" %
                                        (i, q_word, left_pos_path, right_pos_path)))
                result.add(get_feat_num("NE_DEP_PATH_%d_%s_%s_%s" %
                                        (i, q_word, left_ne_path, right_ne_path)))
    return result


def span_length_features(np, features):
    features.append(get_feat_num("NP_LEN_%d" % (len(np))))


def root_features(sents, sent_num, question_root, unique_question_words, features):
    sent = sents[sent_num]

    sent_words = set([token.orth_.lower() for token in sent])
    root_match = sent.root.orth_.lower() == question_root.orth_.lower()
    root_match_lemma = sent.root.lemma == question_root.lemma
    if root_match:
        root_match_feat_num = get_feat_num("ROOT_MATCH")
        features.append(root_match_feat_num)
    if root_match_lemma:
        root_match_lemma_feat_num = get_feat_num("ROOT_MATCH_LEMMA")
        features.append(root_match_lemma_feat_num)

    if question_root.orth_.lower() in sent_words:
        sent_contains_quest_root_feat_num = get_feat_num("SENT_CONTAINS_QUEST_ROOT")
        features.append(sent_contains_quest_root_feat_num)
    if sent.root.orth_.lower() in unique_question_words:
        quest_contains_sent_root_feat_num = get_feat_num("QUEST_CONTAINS_SENT_ROOT")
        features.append(quest_contains_sent_root_feat_num)


def left_right_in_question(np, sent_num, sent_boundaries, c_parsed,
                           unique_question_words, unique_question_bigrams, features):

    left_words = [c_parsed[np.start - i].orth_.lower() for i in xrange(1, 6)
                  if np.start - i >= sent_boundaries[sent_num][0]]
    for i, token in enumerate(reversed(left_words)):
        if token in unique_question_words:
            features.append(get_feat_num("LEFT_WORD_%d_IN_QUESTION" % i))

    left_bigrams = bigrams(left_words)
    for i, bigram in enumerate(reversed(left_bigrams)):
        if bigram in unique_question_words:
            features.append(get_feat_num("LEFT_BIGRAM_%d_IN_QUESTION" % i))

    right_words = [c_parsed[np.end + i].orth_.lower() for i in xrange(5)
                   if np.end + i < sent_boundaries[sent_num][1]]
    for i, token in enumerate(right_words):
        if token in unique_question_bigrams:
            features.append(get_feat_num("RIGHT_WORD_%d_IN_QUESTION" % i))

    right_bigrams = bigrams(right_words)
    for i, bigram in enumerate(right_bigrams):
        if bigram in unique_question_bigrams:
            features.append(get_feat_num("RIGHT_BIGRAM_%d_IN_QUESTION" % i))


def left_right_context(np, sent_num, sent_boundaries, c_parsed, q_parsed, features):
    # adding POS info reduced accuracy by around .4
    # using lemmas in addition also did not help
    # also expanding to the whole question reduced accuracy
    left_words = [c_parsed[np.start - i] for i in xrange(1, 6)
                  if np.start - i >= sent_boundaries[sent_num][0]]
    for i, q_word in enumerate(q_parsed[:3]):
        q_word = q_word.orth_
        for j, token in enumerate(reversed(left_words)):
            features.append(get_feat_num("LEFT_WORD_CONTEXT_%d_%s_%d_%s" %
                                         (i, q_word, j, token.orth_.lower())))

    right_words = [c_parsed[np.end + i] for i in xrange(5)
                   if np.end + i < sent_boundaries[sent_num][1]]
    for i, q_word in enumerate(q_parsed[:3]):
        q_word = q_word.orth_
        for j, token in enumerate(right_words):
            features.append(get_feat_num("RIGHT_WORD_CONTEXT_%d_%s_%d_%s"
                                         % (i, q_word, j, token.orth_.lower())))


def np_words_in_question(np, unique_question_words, unique_question_bigrams):
    result = set()
    words = [token.orth_.lower() for token in np]
    grams = bigrams(words)

    for token in words:
        if token in unique_question_words:
            result.add(get_feat_num("NP_WORD_IN_QUESTION"))

    for gram in grams:
        if gram in unique_question_words:
            features.append(get_feat_num("NP_BIGRAM_IN_QUESTION"))

    return result


def np_contains_stop(np):
    result = set()

    for token in np:
        if token.is_stop:
            result.add(get_feat_num("NP_CONTAINS_STOP"))

    return result


def calc_sent_quest_sim(c_parsed, q_parsed):
    sent_sims = []
    word_wise_sent_sims = []
    for sent_num, sent in enumerate(c_parsed.sents):
        sent_sims.append(sent.similarity(q_parsed))
        word_wise_sent_sim = 0
        for word in sent:
            for q_word in q_parsed:
                word_wise_sent_sim += word.similarity(q_word)
        word_wise_sent_sims.append(word_wise_sent_sim)

    return normalize_list(sent_sims), normalize_list(word_wise_sent_sims)


def calc_np_quest_sim(all_nps, q_parsed):
    np_sims = []
    word_wise_np_sims = []
    for np in all_nps:
        np_sims.append(np.similarity(q_parsed))
        word_wise_np_sim = 0
        for word in np:
            for q_word in q_parsed:
                word_wise_np_sim += word.similarity(q_word)
        word_wise_np_sims.append(word_wise_np_sim)

    return normalize_list(np_sims), normalize_list(word_wise_np_sims)


def np_sim_features(np_num, sims, features):
    np_sims, word_wise_np_sims = sims
    np_sim = np_sims[np_num]
    ww_np_sim = word_wise_np_sims[np_num]

    np_sim_feat_num = get_feat_num("NP_SIM_%d" % np_sim)
    if np_sim_feat_num:
        features.append(np_sim_feat_num)
    else:
        features.append(get_feat_num("UNK_NP_SIM"))

    ww_np_sim_feat_num = get_feat_num("WW_NP_SIM_%d" % ww_np_sim)
    if ww_np_sim_feat_num:
        features.append(ww_np_sim_feat_num)
    else:
        features.append(get_feat_num("UNK_WW_NP_SIM"))


def sent_sim_features(sent_num, sims, features):
    sent_sims, word_wise_sent_sims = sims
    sent_sim = sent_sims[sent_num]
    ww_sent_sim = word_wise_sent_sims[sent_num]

    sent_sim_feat_num = get_feat_num("SENT_SIM_%d" % sent_sim)
    if sent_sim_feat_num:
        features.append(sent_sim_feat_num)
    else:
        features.append(get_feat_num("UNK_SENT_SIM"))

    ww_sent_sim_feat_num = get_feat_num("WW_SENT_SIM_%d" % ww_sent_sim)
    if ww_sent_sim_feat_num:
        features.append(ww_sent_sim_feat_num)
    else:
        features.append(get_feat_num("UNK_WW_SENT_SIM"))


def calc_sent_tf_idf(c_parsed, unique_question_words, unique_question_bigrams):
    sent_word_tf_idfs = []
    sent_bigram_tf_idfs = []

    for sent_num, sent in enumerate(c_parsed.sents):
        sent_words = [token.orth_.lower() for token in sent]
        sent_bigrams = bigrams(sent_words)

        total_words = len(sent_words)
        total_bigrams = total_words - 1

        unique_sent_words = set(sent_words)
        unique_sent_bigrams = set(sent_bigrams)

        word_tf_counter = Counter(sent_words)
        bigram_tf_counter = Counter(sent_bigrams)

        sent_word_tf_idf = sum(word_tf_idf(word, word_tf_counter, total_words)
                                for word in unique_sent_words if word in unique_question_words)
        sent_word_tf_idfs.append(sent_word_tf_idf)

        sent_bigram_tf_idf = sum(bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams)
                                for bigram in unique_sent_bigrams if bigram in unique_question_bigrams)
        sent_bigram_tf_idfs.append(sent_bigram_tf_idf)

    return normalize_list(sent_word_tf_idfs), normalize_list(sent_bigram_tf_idfs)


def sent_tf_idf_features(sent_num, tf_idfs, features):
    sent_word_tf_idfs, sent_bigram_tf_idfs = tf_idfs

    word_tf_idf = sent_word_tf_idfs[sent_num]
    bigram_tf_idf = sent_bigram_tf_idfs[sent_num]

    word_tf_idf_feat_num = get_feat_num("WORD_SENT_TF_IDF_%d" % word_tf_idf)
    if word_tf_idf_feat_num:
        features.append(word_tf_idf_feat_num)
    else:
        features.append(get_feat_num("UNK_WORD_SENT_TF_IDF"))

    bigram_tf_idf_feat_num = get_feat_num("BIGRAM_SENT_TF_IDF_%d" % bigram_tf_idf)
    if bigram_tf_idf_feat_num:
        features.append(bigram_tf_idf_feat_num)
    else:
        features.append(get_feat_num("UNK_BIGRAM_SENT_TF_IDF"))


def get_sent_boundaries(c_parsed):
    return sorted([(sent.start, sent.end) for sent in c_parsed.sents])


def binary_search(start, sent_boundaries):
    lo = 0
    hi = len(sent_boundaries) - 1
    while lo <= hi:
        mid = (lo + hi)/2
        mid_span = sent_boundaries[mid]
        if start >= mid_span[0] and start < mid_span[1]:
            return mid
        if start > mid_span[0]:
            lo = mid + 1
        else:
            hi = mid - 1


def get_sent_num(np, sent_boundaries):
    return binary_search(np.start, sent_boundaries)


def get_all_nps(c_parsed):
    all_nps = list(c_parsed.noun_chunks) + list(c_parsed.ents)
    seen_spans = set()
    filtered_nps = []
    for np in all_nps:
        if (np.start, np.end) in seen_spans:
            continue
        else:
            filtered_nps.append(np)
            seen_spans.add((np.start, np.end))
    return filtered_nps


def calc_np_tf_idfs(all_nps):
    word_tf_idfs = []
    bigram_tf_idfs = []
    for np in all_nps:
        np_words = [token.orth_.lower() for token in np]
        np_bigrams = bigrams(np_words)

        total_words = len(np_words)
        total_bigrams = total_words - 1

        unique_np_words = set(np_words)
        unique_np_bigrams = set(np_bigrams)

        word_tf_counter = Counter(np_words)
        bigram_tf_counter = Counter(np_bigrams)

        w_tf_idf = sum(word_tf_idf(word, word_tf_counter, total_words)
                          for word in unique_np_words)
        word_tf_idfs.append(w_tf_idf)
        b_tf_idf = sum(bigram_tf_idf(bigram, bigram_tf_counter, total_bigrams)
                            for bigram in unique_np_bigrams)
        bigram_tf_idfs.append(b_tf_idf)

    return normalize_list(word_tf_idfs), normalize_list(bigram_tf_idfs)


def np_tf_idf_features(np_num, np_tf_idfs, features):
    np_word_tf_idfs, np_bigram_tf_idfs = np_tf_idfs
    np_word_tf_idf = np_word_tf_idfs[np_num]
    np_bigram_tf_idf = np_bigram_tf_idfs[np_num]
    features.append(get_feat_num("NP_WORD_TF_IDF_%d" % np_word_tf_idf))
    features.append(get_feat_num("NP_BIGRAM_TF_IDF_%d" % np_bigram_tf_idf))


def libsvm_features(label, features):
    features = [feat for feat in sorted(features) if feat]
    libsvm_features = " ".join(["%d:1" % feat_num for feat_num in features])
    return "%d " % label + libsvm_features + '\n'


def generate_features(data, feat_fn):
    result = []
    ids = []
    fp = open(feat_fn, 'w')
    global parser
    for topic in tqdm(data['data']):
        for pgraph in topic['paragraphs']:
            token_dict = {}
            context = pgraph['context']
            c_parsed = parser(context)
            # maybe try enumerating all possible spans instead of nps
            all_nps = get_all_nps(c_parsed)
            np_tf_idfs = calc_np_tf_idfs(all_nps)
            sents = list(c_parsed.sents)
            sent_boundaries = get_sent_boundaries(c_parsed)

            for qa in pgraph['qas']:
                question = qa['question']
                question_id = qa['id']
                q_parsed = parser(question)
                question_root = list(q_parsed.sents)[0].root
                question_words = [token.orth_.lower() for token in q_parsed]
                question_bigrams = bigrams(question_words)
                unique_question_words = set(question_words)
                unique_question_bigrams = set(question_bigrams)

                ans = qa['answers'][0]
                answer_start = ans['answer_start']
                answer_text = ans['text']

                sent_sims = calc_sent_quest_sim(c_parsed, q_parsed)
                tf_idfs = calc_sent_tf_idf(c_parsed, unique_question_words,
                                            unique_question_bigrams)

                for np_num, np in enumerate(all_nps):
                    features = []
                    label = 0
                    # mult is number of times item appears in train file
                    # somewhat arbitrary but improves accuracy
                    # try sampling negative instances
                    # also maybe check if words are correct but in wrong position
                    if not _TEST:
                        mult = 1
                        if np.start_char <= answer_start and np.end_char > answer_start:
                            label = 1
                            mult = 4
                        if np.start_char == answer_start and np.orth_ == answer_text:
                            mult = 20

                    sent_num = get_sent_num(np, sent_boundaries)

                    # tf-idf and similarity features
                    sent_sim_features(sent_num, sent_sims, features)
                    sent_tf_idf_features(sent_num, tf_idfs, features)
                    np_tf_idf_features(np_num, np_tf_idfs, features)

                    # sentence root match features
                    root_features(sents, sent_num, question_root,
                                    unique_question_words, features)

                    # whether or not context words appear in question
                    left_right_in_question(np, sent_num, sent_boundaries,
                                            c_parsed, unique_question_words,
                                            unique_question_bigrams, features)

                    # whether or not np words appear in question
                    features += np_contains_stop(np)

                    # whether or not np words appear in question
                    features += np_words_in_question(np, unique_question_words,
                                                     unique_question_bigrams)

                    # lexicalized features from paper
                    features += lexicalized_lemma_features(np, q_parsed)

                    # dependency paths of pos paired wiht question words
                    features += dependency_path_features(np, sent_num, sents,
                                                         unique_question_words,
                                                         q_parsed)

                    # named entities in np paired with question words
                    wh_ent_features(np, q_parsed, features)

                    # pos tags in np paired with question words
                    span_pos_features(np, q_parsed, features)

                    # context words paired with question words
                    left_right_context(np, sent_num, sent_boundaries, c_parsed,
                                       q_parsed, features)

                    # feature for dependency link to question root
                    # feature for dependency link to question entities
                    # add phi and omega?
                    # add unk entries for all features?
                    # add joint span context with first 3 words/ redundant with lexicalized?
                    # question root distance to span
                    # question root matches context
                    # question entities match context
                    # question entities dependency path to span
                    # dependency paths with actual words instead of pos
                    # question to span similarity, didn't work
                    # span length features appear not to work

                    if _TEST:
                        ids.append("%s\t%s" % (question_id, np.orth_.replace('\n', "\\n")))
                        fp.write(libsvm_features(label, features))
                    else:
                        for i in xrange(mult):
                            fp.write(libsvm_features(label, features))
    fp.close()
    return ids


def write_feature_key():
    with open("features.key", 'w') as fp:
        for feat_str, idx in sorted([item for item in FEATURE_DICT.items()],
                                    key=lambda x: x[1]):
            fp.write(("%d\t%s\n" % (idx, feat_str)).encode("utf-8"))


def main():
    global _TEST
    result = []

    train_fn = os.path.join(DATA_DIR, "train-v1.1.json")
    dev_fn = os.path.join(DATA_DIR, "dev-v1.1.json")

    train_feat_fn = "np_train_features.txt"
    with open(train_fn) as fp:
        train_data = json.load(fp)
    with open(dev_fn) as fp:
        dev_data = json.load(fp)

    get_idfs(train_data)
    get_idfs(dev_data)

    # train_data['data'] = train_data['data'][:10]
    ids = generate_features(train_data, train_feat_fn)

    # write_feature_key()

    _TEST = True

    dev_feat_fn = "np_dev_features.txt"

    ids = generate_features(dev_data, dev_feat_fn)

    id_fn = "np_dev_ids.txt"
    with open(id_fn, 'w') as fp:
        for line in ids:
            fp.write(line.encode("utf-8") + '\n')

if __name__ == "__main__":
    main()


from __future__ import division, print_function

import os, sys
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import codecs


"""
Scoring programme for WMT'20 Task 2 HTER **word-level**
"""

# -------------PREPROCESSING----------------
def list_of_lists(a_list):
    '''
    check if <a_list> is a list of lists
    '''
    if isinstance(a_list, (list, tuple, np.ndarray)) and \
            len(a_list) > 0 and \
            all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False


def check_word_tag(words_seq, tags_seq, dataset_name='', gap_tags=False):
    '''
    check that two lists of sequences have the same number of elements
    :gaps: --- checking number of gaps between words (has to be words+1)
    '''
    for idx, (words, tags) in enumerate(zip(words_seq, tags_seq)):
        if gap_tags:
            words.append(' ')
        assert(len(words) == len(tags)), "Numbers of words and tags don't match in sequence %d of %s: %i and %i" % (
            idx,
            dataset_name,
            len(words),
            len(tags)
        )


def check_words(ref_words, pred_words):
    '''
    check that words in reference and prediction match
    '''
    assert(len(ref_words) == len(pred_words)), \
        "Number of word sequences doesn't match in reference and hypothesis: %d and %d" % (len(ref_words),
                                                                                           len(pred_words))
    for idx, (ref, pred) in enumerate(zip(ref_words, pred_words)):
        ref_str = ' '.join(ref).lower()
        pred_str = ' '.join(pred).lower()
        assert(ref_str == pred_str), \
            "Word sequences don't match in reference and hypothesis at line %d:\n\t%s\n\t%s\n" % (idx,
                                                                                                  ref_str,
                                                                                                  pred_str)


def parse_submission(ref_txt_file, ref_tags_file, submission_tags_file, gap_tags=False):
    LP_ID = 0
    SYSNAME = 1
    TYPE = 2
    SENT_ID = 3
    WORD_IDX = 4
    WORD = 5
    TAG = 6

    tag_map = {'GOOD': 1, 'OK': 1, 'BAD': 0, 'good': 1, 'ok': 1, 'bad': 0}

    ref_words = []
    for line in codecs.open(ref_txt_file, 'r', encoding='utf-8'):
        ref_words.append(line.strip().split())

    ref_tags = [[] for i in range(len(ref_words))]
    for idx, line in enumerate(codecs.open(ref_tags_file, 'r', encoding='utf-8')):
        chunks = line.strip().split('\t')
        cur_seq = int(chunks[SENT_ID])
        ref_tags[cur_seq].append(tag_map[chunks[TAG]])

    check_word_tag(ref_words, ref_tags, dataset_name='reference', gap_tags=gap_tags)

    submission_tags = [[] for i in range(len(ref_words))]
    submission_words = [[] for i in range(len(ref_words))]
    for idx, line in enumerate(codecs.open(submission_tags_file, 'r', encoding='utf-8')):
        chunks = line.strip().split('\t')
        cur_seq = int(chunks[SENT_ID])
        submission_words[cur_seq].extend(chunks[WORD].strip().split())
        submission_tags[cur_seq].append(tag_map[chunks[TAG]])

    if not gap_tags:
        check_words(ref_words, submission_words)
        check_word_tag(submission_words, submission_tags, dataset_name='submission', gap_tags=gap_tags)

    return (ref_tags, submission_tags)


def flatten(lofl):
    '''
    convert list of lists into a flat list
    '''
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)

    f1_all_scores = f1_score(flat_true, flat_pred, average=None, pos_label=None)

    # Matthews correlation coefficient (MCC)
    # true/false positives/negatives
    tp = tn = fp = fn = 0
    for pred_tag, gold_tag in zip(flat_pred, flat_true):
        if pred_tag == 1:
            if pred_tag == gold_tag:
                tp += 1
            else:
                fp += 1
        else:
            if pred_tag == gold_tag:
                tn += 1
            else:
                fn += 1

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator

    return np.append(f1_all_scores, mcc)


def evaluate(ref_txt_file, ref_tags_file, submission, gap_tags=False):
    (true_tags, test_tags) = parse_submission(ref_txt_file, ref_tags_file, submission, gap_tags)
    return compute_scores(true_tags, test_tags)


def main(ref_dir: 'dir containing all ref files', submit_dir: 'dir containing all predicitiosn files'):

    if not os.path.isdir(ref_dir):
        print("%s doesn't exist" % ref_dir)

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    # target scores
    ref_mt_txt = os.path.join(ref_dir, "goldlabels_mt.txt")
    ref_mt_tags = os.path.join(ref_dir, "goldlabels_mt.tags")
    submit_mt_tags = os.path.join(submit_dir, "predictions_mt.txt")

    f1_bad_tg, f1_good_tg, mcc_tg = evaluate(ref_mt_txt, ref_mt_tags, submit_mt_tags)
    f1_multi_tg = f1_bad_tg * f1_good_tg

    # source scores
    ref_src_txt = os.path.join(ref_dir, "goldlabels_src.txt")
    ref_src_tags = os.path.join(ref_dir, "goldlabels_src.tags")
    submit_src_tags = os.path.join(submit_dir, "predictions_src.txt")

    if os.path.isfile(submit_src_tags):
        f1_bad_src, f1_good_src, mcc_src = evaluate(ref_src_txt, ref_src_tags, submit_src_tags)
        f1_multi_src = f1_bad_src * f1_good_src

    else:
        f1_bad_src, f1_good_src, f1_multi_src, mcc_src = 0.0, 0.0, 0.0, 0.0

    ref_gaps_tags = os.path.join(ref_dir, "goldlabels_gaps.tags")
    submit_gaps_tags = os.path.join(submit_dir, "predictions_gaps.txt")
    if os.path.isfile(submit_gaps_tags):
        f1_bad_gap, f1_good_gap, mcc_gap = evaluate(ref_mt_txt, ref_gaps_tags, submit_gaps_tags, gap_tags=True)
        f1_multi_gap = f1_bad_gap * f1_good_gap

    else:
        f1_bad_gap, f1_good_gap, f1_multi_gap, mcc_gap = 0.0, 0.0, 0.0, 0.0

    print('mt_mcc: {:.4}'.format(mcc_tg))
    print("mt_f1-bad: {:.4}".format(f1_bad_tg))
    print('mt_f1-good: {:.4}'.format(f1_good_tg))
    print('mt_f1-multi: {:.4}'.format(f1_multi_tg))
    print('---')

    print("gaps_mcc: {:.4}".format(mcc_gap))
    print("gaps_f1-bad: {:.4}".format(f1_bad_gap))
    print('gaps_f1-good: {:.4}'.format(f1_good_gap))
    print('gaps_f1-multi: {:.4}'.format(f1_multi_gap))
    print('---')

    print('src_mcc: {:.4}'.format(mcc_src))
    print("src_f1-bad: {:.4}".format(f1_bad_src))
    print('src_f1-good: {:.4}'.format(f1_good_src))
    print('src_f1-multi: {:.4}'.format(f1_multi_src))


CATS = ['conditions_this_may_prevent', 'side_effects', 'used_to_treat']
from collections import defaultdict as dd
import numpy as np
import multiprocessing as mp

# def read_train_labels(filename):
#     ret = []
#     for line in open(filename):
#         inputs = line.strip().split()
#         ret.append(inputs[2])
#     return ret

# def read_test_labels(filename):
#     ret = []
#     for line in open(filename):
#         ret.append(line.strip().replace(' ', '_'))
#     return ret

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return 1.0

    if len(source) - len(target) > len(source) * 0.2: return 1.0

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1] * 1.0 / len(source)

def func(args):
    start, delta, test_cov, corpus = tuple(args)
    max_len = len(test_cov)
    sim_dict = dd(list)
    for i in range(start, min(max_len, start + delta)):
        if i < delta:
            print "{} out of {}".format(i, delta)
        for j in range(len(corpus)):
            t1, t2 = test_cov[i], corpus[j]
            t1s = t1.split('@')
            t2s = t2.split('@')
            # yf: later part is generally shorter than the medicine.
            dist2 = levenshtein(t1s[1],t2s[1])
            if dist2 <= 0.2:
                dist1 = levenshtein(t1s[0],t2s[0])
                if dist1 <= 0.2:
                    sim_dict[t1].append(t2)
                    print t1, t2
    return sim_dict

def read_corpus(filename):
    corpus = set()
    for line in open(filename):
        corpus.add(line.strip().split()[-1])
    return list(corpus)

def read_test_cov(filename):
    test_cov = {}
    for cat in CATS:
        test_cov[cat] = []
    for line in open(filename):
        inputs = line.strip().split()
        # test_cov[CATS[int(inputs[1]) - 1]].append(inputs[0])
        test_cov[inputs[1]].append(inputs[2])
    return test_cov

def compute_sim():
    # dir = '../diel'
    # corpus = read_corpus(dir + '/hasItem.cfacts')
    # yf: <list> all the items in the lists.
    corpus = read_corpus('/remote/curtis/baidu/SSL-pipeline/database/unit0_nlm_para_200/step2_gpid_cfacts/graph/hasItem_aug.cfacts')
    test_cov = set()
    # for i in range(10):
    #     t_test_cov = read_test_cov("{}/{}/coverage_eva_multiAdded".format(dir, i))
    #     for cat in CATS:
    #         test_cov.update(set(t_test_cov[cat]))
    t_test_cov = read_test_cov('/remote/curtis/baidu/SSL-pipeline/database/unit1_seed_nlm_para/seed_matched_secondString/all.cfacts')
    for cat in CATS:
        test_cov.update(set(t_test_cov[cat]))
    # yf: <list> all the items in the seeds.
    test_cov = list(test_cov)

    sim_dict = dd(list)

    num_threads = 64
    p = mp.Pool(num_threads)
    delta = len(test_cov) / num_threads + 1
    args = [(delta * i, delta, test_cov, corpus) for i in range(num_threads)]
    sim_dicts = p.map(func, args)

    fout = open('sim.dict', 'w')
    for sim_dict in sim_dicts:
        for k, v in sim_dict.iteritems():
            for vv in v:
                fout.write("{} {}".format(k, vv) + "\n")
    fout.close()


compute_sim()

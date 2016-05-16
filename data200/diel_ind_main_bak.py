
from collections import defaultdict as dd
import diel_ind_model
import argparse
import numpy as np
from scipy import sparse

CATS = ['conditions_this_may_prevent', 'side_effects', 'used_to_treat']
FEATURE_SAVED = False

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'], dtype = np.float32)

def read_features(filename):
    print 'reading features'
    features, f_num = {}, 0
    for line in open(filename):
        inputs = line.strip().split()
        features[inputs[0]] = []
        for t in inputs[1:]:
            tt = int(t)
            f_num = max(f_num, tt + 1)
            features[inputs[0]].append(tt)
    print "    TEST: len_list={},f_num={}".format(len(features.keys()),f_num)
    return features, f_num

def read_cites(filename):
    print 'reading cites'
    cites, s_graph = [], dd(list)
    for i, line in enumerate(open(filename)):
        if i % 100000 == 0:
            print 'reading cites {}'.format(i)
        inputs = line.strip().split()
        cites.append((inputs[1], inputs[2]))
        s_graph[inputs[2]].append(inputs[1])
        s_graph[inputs[1]].append(inputs[2])
    print "    TEST: len_cites={},total_nodes={}".format(len(cites),len(s_graph.keys()))
    return cites, s_graph

def read_sim_dict(filename):
    print 'reading sim_dict'
    sim_dict = dd(list)
    for i, line in enumerate(open(filename)):
        inputs = line.strip().split()
        sim_dict[inputs[0]].append(inputs[1])
    return sim_dict

def read_train_labels(filename):
    ret = []
    cat = []
    # i = 0
    for line in open(filename):
        inputs = line.strip().split()
        ret.append(inputs[2])
        cat.append(inputs[1])
        # i += 1
        # print "    TEST: i = {}, item={},cat={}".format(i, inputs[2],inputs[1])
    return ret, cat

# def read_test_labels(filename):
#     ret = []
#     for line in open(filename):
#         ret.append(line.strip().replace(' ', '_'))
#     return ret

def add_index(index, cnt, key):
    if key in index: return cnt
    index[key] = cnt
    return cnt + 1

def construct_graph(train_id, test_id, cites):
    id2index, cnt = {}, 0
    for id in train_id:
        # print id
        cnt = add_index(id2index, cnt, id)
    for id in test_id:
        cnt = add_index(id2index, cnt, id)
    graph = dd(list)
    for id1, id2 in cites:
        cnt = add_index(id2index, cnt, id1)
        cnt = add_index(id2index, cnt, id2)
        i, j = id2index[id1], id2index[id2]
        graph[i].append(j)
        graph[j].append(i)
    return graph

# train_list, in_labels, features, f_num
def construct_x_y(ents, in_labels, features, f_num):
    row, col = [], []
    for i, ent in enumerate(ents):
        for f_ind in features[ent]:
            row.append(i)
            col.append(f_ind)
    data = np.ones(len(row), dtype = np.float32)
    x = sparse.coo_matrix((data, (row, col)), shape = (len(ents), f_num), dtype = np.float32).tocsr()
    y = np.zeros((len(ents), len(CATS)), dtype = np.int32)
    for i, ent in enumerate(ents):
        for j, cat in enumerate(CATS):
            if ent in in_labels[cat]:
                # yf: possible that y[i, j1] = y[i, j2]=1
                y[i, j] = 1
    return x, y

def read_test_cov(filename):
    print 'reading test cov'
    test_cov = {}
    for cat in CATS:
        test_cov[cat] = []
    for line in open(filename):
        inputs = line.strip().split()
        test_cov[inputs[1]].append(inputs[2])
    return test_cov
# yf:
# features  -<dict>:  list name -> 1' pos of features.
# f_num:              dimension of features.
# cites     -<list>:  (list name, item name)
# s_graph   -<dict>:  list name -> <list> of items, item name -> <list> of lists
# sim_dict  -<dict>:  test item name -> <list> of item names
def run(run_num, args, features, f_num, cites, s_graph, sim_dict):
    # yf: train_list-<set>: lists which contain the test items
    # in_labels-<dict>: catergory -> related lists
    train_list, in_labels = set(), dd(set)
#     for cat in CATS:
#         print "processing {}".format(cat)
#         # train_item = read_train_labels("{}/{}/{}_devel_50p_proppr_seed_forTrainList".format(dir, run_num, cat))
#         train_item = read_train_labels("../seeds/{}/devel".format(run_num))        
#         for item in train_item:
#             for l in s_graph[item]:
#                 train_list.add(l)
#                 in_labels[cat].add(l)
    print 'reading into training seeds'
    train_item, cat = read_train_labels("../seeds/{}/devel".format(run_num))        
    for i, item in enumerate(train_item):
        # print i, cat[i]
        for l in s_graph[item]:
            train_list.add(l)
            in_labels[cat[i]].add(l)
    # yf: test_list-<set>: lists which do not contain the test items
    test_list = set()
    for l, _ in cites:
        if l not in train_list:
            test_list.add(l)
    train_list = list(train_list)
    test_list = list(test_list)

    # print len(train_list), len(test_list)

    if not FEATURE_SAVED:
        print 'constructing training lists'
        x, y = construct_x_y(train_list, in_labels, features, f_num)
        print 'constructing test lists'
        tx, ty = construct_x_y(test_list, in_labels, features, f_num) # ty are all zeros.
        
        print 'saving'
        save_sparse_csr("{}.x".format(run_num), x)
        save_sparse_csr("{}.tx".format(run_num), tx)
        np.save("{}.y".format(run_num), y)
        np.save("{}.ty".format(run_num), ty)
    else:
        print 'loading'
        x = load_sparse_csr("{}.x.npz".format(run_num))
        tx = load_sparse_csr("{}.tx.npz".format(run_num))
        y = np.load("{}.y.npy".format(run_num))
        ty = np.load("{}.ty.npy".format(run_num))

    print x.shape, y.shape
    print tx.shape, ty.shape

    allx = sparse.vstack([x, tx], format = 'csr')

    print 'constructing graph'
    # yf: graph-<dict>: index -> index of connected vertices
    graph = construct_graph(train_list, test_list, cites)

    m = diel_ind_model.model(args)
    m.add_data(x, y, tx, ty, allx, graph)
    m.build()
    m.graph_train()

    test_cov = read_test_cov("../seeds/{}/eva".format(run_num))

    max_recall = 0.0
    for epoch in range(100):
        print 'training epoch {}'.format(epoch)
        # yf: tpy-predicted y for test lists.
        tpy, loss = m.step_train(args.train_step)
        tpy_ind = np.argmax(tpy, axis = 1)
        st_dict = dd(float)
        for i, l in enumerate(test_list):
            # yf: cat-predicted label for test list l.
            j = tpy_ind[i]
            cat = CATS[j]
            for item in s_graph[l]:
                # yf: st_dict[(item, cat)]-max prob of item in cat.
                cur = st_dict[(item, cat)]
                if tpy[i, j] > cur:
                    st_dict[(item, cat)] = tpy[i, j]
        # yf: st_dict-<list>: [(item,cat)->prob] in the order of descending prob. 
        st_dict = sorted(st_dict.items(), key = lambda x: x[1], reverse = True)

        len_st_dict = len(st_dict)
        pred_labels = dd(set)
        for k, _ in st_dict[: 240000]:
            item, cat = k
            pred_labels[cat].add(item)
        pred_num = 0
        for cat in CATS:
            pred_num += len(pred_labels[cat])
        tot, cor = 0, 0
        for cat in CATS:
            for test_item in test_cov[cat]:
                tot += 1
                for item in sim_dict[test_item]:
                    if item in pred_labels[cat]:
                        cor += 1
                        break
        recall = 1.0 * cor / tot
        max_recall = max(max_recall, recall)
        print loss, pred_num, len_st_dict, recall, max_recall
    return max_recall

# def external_dataset(run_num):
#     dir = '../diel'
#     cites, s_graph = read_cites(dir + '/hasItem.cfacts')
#     sim_dict = read_sim_dict('sim.dict')
# 
#     train_list, in_labels = set(), dd(set)
#     for cat in CATS:
#         print "processing {}".format(cat)
#         train_item = read_train_labels("{}/{}/{}_devel_50p_proppr_seed_forTrainList".format(dir, run_num, cat))
#         for item in train_item:
#             for l in s_graph[item]:
#                 train_list.add(l)
#                 in_labels[cat].add(l)
# 
#     test_list = set()
#     for l, _ in cites:
#         if l not in train_list:
#             test_list.add(l)
#     train_list = list(train_list)
#     test_list = list(test_list)
# 
#     y = np.load("{}.y.npy".format(run_num))
# 
#     graph = construct_graph(train_list, test_list, cites)
# 
#     return y, graph


# def external_train(run_num, m):
#     dir = '../diel'
#     cites, s_graph = read_cites(dir + '/hasItem.cfacts')
#     sim_dict = read_sim_dict('sim.dict')
# 
#     train_list, in_labels = set(), dd(set)
#     for cat in CATS:
#         print "processing {}".format(cat)
#         train_item = read_train_labels("{}/{}/{}_devel_50p_proppr_seed_forTrainList".format(dir, run_num, cat))
#         for item in train_item:
#             for l in s_graph[item]:
#                 train_list.add(l)
#                 in_labels[cat].add(l)
# 
#     test_list = set()
#     for l, _ in cites:
#         if l not in train_list:
#             test_list.add(l)
#     train_list = list(train_list)
#     test_list = list(test_list)
# 
#     print 'loading'
#     x = load_sparse_csr("{}.x.npz".format(run_num))
#     tx = load_sparse_csr("{}.tx.npz".format(run_num))
#     y = np.load("{}.y.npy".format(run_num))
#     ty = np.load("{}.ty.npy".format(run_num))
# 
#     print x.shape, y.shape
#     print tx.shape, ty.shape
# 
#     allx = sparse.vstack([x, tx], format = 'csr')
# 
#     print 'constructing graph'
#     graph = construct_graph(train_list, test_list, cites)
# 
#     m.add_data(x, y, tx, ty, allx, graph)
#     m.build()
# 
#     test_cov = read_test_cov("{}/{}/coverage_eva_multiAdded".format(dir, run_num))
# 
#     max_recall = 0.0
#     for epoch in range(20):
#         print 'training epoch {}'.format(epoch)
#         tpy, loss, g_loss = m.step_train()
#         tpy_ind = np.argmax(tpy, axis = 1)
#         st_dict = dd(float)
#         for i, l in enumerate(test_list):
#             j = tpy_ind[i]
#             cat = CATS[j]
#             for item in s_graph[l]:
#                 cur = st_dict[(item, cat)]
#                 if tpy[i, j] > cur:
#                     st_dict[(item, cat)] = tpy[i, j]
#         st_dict = sorted(st_dict.items(), key = lambda x: x[1], reverse = True)
# 
#         pred_labels = dd(set)
#         for k, _ in st_dict[: 240000]:
#             item, cat = k
#             pred_labels[cat].add(item)
#         pred_num = 0
#         for cat in CATS:
#             pred_num += len(pred_labels[cat])
#         tot, cor = 0, 0
#         for cat in CATS:
#             for test_item in test_cov[cat]:
#                 tot += 1
#                 for item in sim_dict[test_item]:
#                     if item in pred_labels[cat]:
#                         cor += 1
#                         break
#         recall = 1.0 * cor / tot
#         max_recall = max(max_recall, recall)
#         print loss, g_loss, pred_num, recall, max_recall
#     return max_recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seeds', help = 'percentage of seeds', type = float, default = 20)
    parser.add_argument('--learning_rate', help = 'learning rate', type = float, default = 1.0) # 0.1
    parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
    parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
    parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
    parser.add_argument('--batch_size', help = 'the size of batch for training instances', type = int, default = 200)
    parser.add_argument('--g_batch_size', help = 'the batch size for graph', type = int, default = 150) #zl: 50
    parser.add_argument('--g_sample_size', help = 'the sample size from label information', type = int, default = 1000)
    parser.add_argument('--neg_samp', help = 'negative sampling rate', type = int, default = 5)
    parser.add_argument('--g_learning_rate', help = 'learning rate for graph', type = float, default = 1e-1)
    parser.add_argument('--train_step', type = int, default = 6) # 100
    args = parser.parse_args()

    # dir = '../diel'

    if not FEATURE_SAVED:
        features, f_num = read_features('list_features.txt')
    else:
        features, f_num = None, None
    cites, s_graph = read_cites('/remote/curtis/baidu/SSL-pipeline/database/unit0_nlm_para_200/step2_gpid_cfacts/graph/hasItem_aug.cfacts')
    sim_dict = read_sim_dict('sim.dict')
    run(1, args, features, f_num, cites, s_graph, sim_dict)
    # for i in range(1, 10):
    #     max_recall = run(dir, i, args, features, f_num, cites, s_graph, sim_dict)
    #     fout = open('diel.results.txt', 'a')
    #     fout.write("{} {}\n".format(i, max_recall))
    #     fout.close()


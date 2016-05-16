
import lasagne
from theano import sparse
import theano.tensor as T
import theano
import layers
import numpy as np
import random
import copy
from numpy import linalg as lin
from collections import defaultdict as dd
import cPickle

EMB = False
PRINT_PERIOD = 1
EMB_ONLY = True
EXP_SOFTMAX = True
EMBEDDING_FILE = 'embedding.cache.txt'
JOINT = False

class model:

    def __init__(self, args):
        self.embedding_size = args.embedding_size
        self.window_size = args.window_size
        self.path_size = args.path_size
        self.batch_size = args.batch_size
        self.g_batch_size = args.g_batch_size
        self.g_sample_size = args.g_sample_size
        self.learning_rate = args.learning_rate
        self.neg_samp = args.neg_samp
        self.g_learning_rate = args.g_learning_rate

        lasagne.random.set_rng(np.random)
        np.random.seed(13)

        random.seed(13)

        self.inst_generator = self.gen_train_inst()
        self.graph_generator = self.gen_graph()
        self.label_generator = self.gen_label_graph()

    def add_data(self, x, tx, y, ty, graph):
        self.x, self.tx, self.y, self.ty, self.graph = x, tx, y, ty, graph

    def build(self):
        print 'start building'

        x_sym = sparse.csr_matrix('x', dtype = 'float32')
        y_sym = T.imatrix('y')
        g_sym = T.imatrix('g')
        gy_sym = T.vector('gy')
        ind_sym = T.ivector('ind')
        self.x_sym, self.y_sym, self.ind_sym = x_sym, y_sym, ind_sym

        l_x_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = x_sym)
        l_g_in = lasagne.layers.InputLayer(shape = (None, 2), input_var = g_sym)
        l_ind_in = lasagne.layers.InputLayer(shape = (None, ), input_var = ind_sym)
        l_gy_in = lasagne.layers.InputLayer(shape = (None, ), input_var = gy_sym)

        reg_weights_l1, reg_weights_l2 = {}, {}

        num_ver = max(self.graph.keys()) + 1
        l_emb_in = lasagne.layers.SliceLayer(l_g_in, indices = 0, axis = 1)
        l_emb_in = lasagne.layers.EmbeddingLayer(l_emb_in, input_size = num_ver, output_size = self.embedding_size)
        self.embedding = l_emb_in.W
        l_emb_out = lasagne.layers.SliceLayer(l_g_in, indices = 1, axis = 1)
        if self.neg_samp > 0:
            l_emb_out = lasagne.layers.EmbeddingLayer(l_emb_out, input_size = num_ver, output_size = self.embedding_size)

        l_emd_f = lasagne.layers.EmbeddingLayer(l_ind_in, input_size = num_ver, output_size = self.embedding_size, W = l_emb_in.W)
        l_emb_joint = l_emd_f
        l_x_hid = layers.SparseLayer(l_x_in, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        if EMB:
            reg_weights_l1[l_x_hid] = 1e-4
            l_emd_f = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
            self.W = l_emd_f.W
            reg_weights_l1[l_emd_f] = 1e-4
            l_y = lasagne.layers.ConcatLayer([l_x_hid, l_emd_f], axis = 1)
            l_y = layers.DenseLayer(l_y, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
            reg_weights_l1[l_y] = 1e-4
        elif EMB_ONLY:
            l_y = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
            reg_weights_l1[l_y] = 1e-4
        else:
            l_y = layers.SparseLayer(l_x_in, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
            reg_weights_l1[l_y] = 1e-4

        py_sym = lasagne.layers.get_output(l_y)
        loss = lasagne.objectives.categorical_crossentropy(py_sym, y_sym).mean()
        if EMB:
            hid_sym = lasagne.layers.get_output(l_x_hid)
            loss += lasagne.objectives.categorical_crossentropy(hid_sym, y_sym).mean()
            emd_sym = lasagne.layers.get_output(l_emd_f)
            loss += lasagne.objectives.categorical_crossentropy(emd_sym, y_sym).mean()

        loss += lasagne.regularization.regularize_layer_params_weighted(reg_weights_l1, lasagne.regularization.l1)
        loss += lasagne.regularization.regularize_layer_params_weighted(reg_weights_l2, lasagne.regularization.l2)

        tpy_sym = lasagne.layers.get_output(l_y, deterministic = True)

        if self.neg_samp == 0:
            l_gy = layers.DenseLayer(l_emb_in, num_ver, nonlinearity = lasagne.nonlinearities.softmax)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = lasagne.objectives.categorical_crossentropy(pgy_sym, lasagne.layers.get_output(l_emb_out)).sum()
        else:
            l_gy = lasagne.layers.ElemwiseMergeLayer([l_emb_in, l_emb_out], T.mul)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = - T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis = 1) * gy_sym)).sum()

        if EMB:
            params = [l_emd_f.W, l_emd_f.b, l_x_hid.W, l_x_hid.b, l_y.W, l_y.b]
            if JOINT:
                params.append(l_emb_joint.W)
        else:
            params = [l_y.W, l_y.b]
        updates = lasagne.updates.sgd(loss, params, learning_rate = self.learning_rate)

        self.train_fn = theano.function([x_sym, y_sym, g_sym, gy_sym, ind_sym], loss, updates = updates, on_unused_input = 'warn')

        acc = T.mean(T.eq(T.argmax(tpy_sym, axis = 1), T.argmax(y_sym, axis = 1)))

        self.test_fn = theano.function([x_sym, y_sym, ind_sym], acc, on_unused_input = 'warn')
        self.ret_y = tpy_sym

        self.l = l_gy
        g_params = lasagne.layers.get_all_params(l_gy, trainable = True)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate = self.g_learning_rate)
        self.g_fn = theano.function([g_sym, gy_sym], g_loss, updates = g_updates, on_unused_input = 'warn')

    def gen_train_inst(self):
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < ind.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                # print self.x[ind[i: j]]
                yield self.x[ind[i: j]], self.y[ind[i: j]], ind[i: j]
                i = j

    def gen_label_graph(self):
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1: continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                for _ in range(self.neg_samp):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append( - 1.0)
            yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)

    def gen_graph(self):

        num_ver = max(self.graph.keys()) + 1

        while True:
            ind = np.random.permutation(num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0: continue
                    path = [k]
                    for _ in range(self.path_size):
                        path.append(random.choice(self.graph[path[-1]]))
                    for l in range(len(path)):
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, num_ver - 1)])
                                gy.append(- 1.0)
                yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)
                i = j

    def store_params(self):
        fout = open(EMBEDDING_FILE, 'w')
        params = lasagne.layers.get_all_param_values(self.l)
        cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
        fout.close()

    def load_params(self):
        fin = open(EMBEDDING_FILE)
        params = cPickle.load(fin)
        lasagne.layers.set_all_param_values(self.l, params)
        fin.close()

    def store_emb(self):
        t_embedding = self.embedding.get_value()
        np.save('embedding.npy', t_embedding)

    def train(self):
        print 'start training'

        test_ind = np.arange(self.x.shape[0], self.x.shape[0] + self.tx.shape[0], dtype = np.int32)
        max_acc = 0.0

        if EMB or EMB_ONLY:
            # self.load_params()
            for i in range(2000):
                g, gy = next(self.label_generator)
                loss = self.g_fn(g, gy)
                print 'g_label_fn', i, loss
            for i in range(70):
                g, gy = next(self.graph_generator)
                loss = self.g_fn(g, gy)
                print 'g_fn', i, loss
            # self.store_params()
            self.store_emb()

        iter = 0
        while True:
            # x, y, train_ind = next(self.gen_train_inst())
            x, y, train_ind = next(self.inst_generator)
            loss = self.train_fn(x, y, None, None, train_ind)
            if np.isnan(loss):
                quit()
            if iter % PRINT_PERIOD == 0:
                acc = self.test_fn(self.tx, self.ty, test_ind)
                max_acc = max(max_acc, acc)
                print iter, max_acc, acc, loss
            iter += 1

    def joint_train(self):
        print 'start joint training'

        test_ind = np.arange(self.x.shape[0], self.x.shape[0] + self.tx.shape[0], dtype = np.int32)
        max_acc = 0.0

        for i in range(0):
            g, gy = next(self.label_generator)
            loss = self.g_fn(g, gy)
            print 'g_label_fn', i, loss
        for i in range(1400):
            g, gy = next(self.graph_generator)
            loss = self.g_fn(g, gy)
            print 'g_fn', i, loss

        iter = 0
        label_loss = 0.0
        while True:
            for _ in range(1):
                g, gy = next(self.graph_generator)
                g_loss = self.g_fn(g, gy)
            for _ in range(10):
                x, y, train_ind = next(self.gen_train_inst())
                loss = self.train_fn(x, y, None, None, train_ind)
            # if iter % 10 == 0:
            #     g, gy = next(self.label_generator)
            #     label_loss = self.g_fn(g, gy)
            acc = self.test_fn(self.tx, self.ty, test_ind)
            max_acc = max(acc, max_acc)
            print iter, max_acc, acc, loss, g_loss, label_loss
            iter += 1

    def graph_train(self, iter_1 = 0, iter_2 = 20000):
        if EMB or EMB_ONLY:
            # self.load_params()
            for i in range(iter_1):
                g, gy = next(self.label_generator)

                # continue ######

                loss = self.g_fn(g, gy)
                print 'g_label_fn', i, loss
            for i in range(iter_2):
                g, gy = next(self.graph_generator)

                # if i < 50: continue ######

                loss = self.g_fn(g, gy)
                print 'g_fn', i, loss
            # self.store_params()

    def step_train(self, iter):
        test_ind = np.arange(self.x.shape[0], self.x.shape[0] + self.tx.shape[0], dtype = np.int32)
        for i in range(iter):
            x, y, train_ind = next(self.inst_generator)
            loss = self.train_fn(x, y, None, None, train_ind)
        if not EMB_ONLY:
            return self.ret_y.eval({self.x_sym: self.tx, self.ind_sym: test_ind}), loss
        else:
            return self.ret_y.eval({self.ind_sym: test_ind}), loss



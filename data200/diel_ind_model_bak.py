
from theano import sparse
import theano.tensor as T
import lasagne
import layers
import theano
import numpy as np
import random

class model:

    def __init__(self, args):
        self.embedding_size = args.embedding_size
        self.learning_rate = args.learning_rate
        self.g_learning_rate = args.g_learning_rate
        self.batch_size = args.batch_size
        self.g_batch_size = args.g_batch_size
        self.window_size = args.window_size
        self.path_size = args.path_size
        self.neg_samp = args.neg_samp

        lasagne.random.set_rng(np.random)
        np.random.seed(13)

        random.seed(13)

        self.inst_generator = self.gen_train_inst()
        self.graph_generator = self.gen_graph()

    def add_data(self, x, y, tx, ty, allx, graph):
        self.x, self.y, self.tx, self.ty, self.allx, self.graph = x, y, tx, ty, allx, graph
        print self.x.shape, self.tx.shape, self.allx.shape
        self.num_ver = max(self.graph.keys()) + 1

    def build(self):
        x_sym = sparse.csr_matrix('x', dtype = 'float32')
        self.x_sym = x_sym
        y_sym = T.imatrix('y')
        gx_sym = sparse.csr_matrix('gx', dtype = 'float32')
        gy_sym = T.ivector('gy')
        gz_sym = T.vector('gz')

        l_x_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = x_sym)
        l_gx_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = gx_sym)
        l_gy_in = lasagne.layers.InputLayer(shape = (None, ), input_var = gy_sym)

        l_x_1 = layers.SparseLayer(l_x_in, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        l_x_2 = layers.SparseLayer(l_x_in, self.embedding_size)
        W = l_x_2.W
        l_x_2 = lasagne.layers.DenseLayer(l_x_2, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        l_x = lasagne.layers.ConcatLayer([l_x_1, l_x_2], axis = 1)
        l_x = lasagne.layers.DenseLayer(l_x, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)

        l_gx = layers.SparseLayer(l_gx_in, self.embedding_size, W = W)
        l_gy = lasagne.layers.EmbeddingLayer(l_gy_in, input_size = self.num_ver, output_size = self.embedding_size)
        l_gx = lasagne.layers.ElemwiseMergeLayer([l_gx, l_gy], T.mul)
        pgy_sym = lasagne.layers.get_output(l_gx)
        g_loss = - T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis = 1) * gz_sym)).sum()

        py_sym = lasagne.layers.get_output(l_x)
        self.ret_y = py_sym
        loss = lasagne.objectives.categorical_crossentropy(py_sym, y_sym).mean()

        # params = lasagne.layers.get_all_params(l_x)
        params = [l_x_1.W, l_x_1.b, l_x_2.W, l_x_2.b, l_x.W, l_x.b]
        updates = lasagne.updates.sgd(loss, params, learning_rate = self.learning_rate)
        self.train_fn = theano.function([x_sym, y_sym], loss, updates = updates)

        g_params = lasagne.layers.get_all_params(l_gx)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate = self.g_learning_rate)
        self.g_fn = theano.function([gx_sym, gy_sym, gz_sym], g_loss, updates = g_updates)

        acc = T.mean(T.eq(T.argmax(py_sym, axis = 1), T.argmax(y_sym, axis = 1)))
        self.test_fn = theano.function([x_sym, y_sym], acc)

    def gen_train_inst(self):
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < self.x.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]]
                i = j

    def gen_graph(self):
        while True:
            ind = np.random.permutation(self.num_ver)
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
                        # yf: gx_sym only considers list type vertices.
                        if path[l] >= self.allx.shape[0]: continue
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, self.num_ver - 1)])
                                gy.append(- 1.0)
                g = np.array(g, dtype = np.int32)
                yield self.allx[g[:, 0]], g[:, 1], gy
                i = j

#     def train(self):
#         max_acc, g_loss = 0, 0.0
#         for i in range(400):
#             gx, gy, gz = next(self.graph_generator)
#             g_loss = self.g_fn(gx, gy, gz)
#             print i, g_loss
#         while True:
#             x, y = next(self.inst_generator)
#             loss = self.train_fn(x, y)
#             # gx, gy = next(self.graph_generator)
#             # g_loss = self.g_fn(gx, gy)
#             acc = self.test_fn(self.tx, self.ty)
#             max_acc = max(max_acc, acc)
#             print max_acc, acc, loss, g_loss

    def graph_train(self):
        fout = open('emb_loss.dat', 'w')
        for i in range(1000): # 100 #20000
            gx, gy, gz = next(self.graph_generator)
            g_loss = self.g_fn(gx, gy, gz)
            print i, g_loss, len(gz)
            fout.write("{}".format(g_loss) + "\n")
        fout.close()

    def step_train(self, iter):
        for i in range(iter):
            x, y = next(self.inst_generator)
            loss = self.train_fn(x, y)
        return self.ret_y.eval({self.x_sym: self.tx}), loss


import tensorflow as tf
import scipy.sparse as sp
import numpy as np

seed = 42


def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype=np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


class LACE:
    def __init__(self, args):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.X = tf.SparseTensor(*sparse_feeder(args.X))
        self.N, self.D = args.X.shape
        self.L = args.embedding_dim
        self.n_hidden = [512]

        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])

        self.__create_model()

        if not args.is_all:
            self.val_edges = args.val_edges
            self.val_ground_truth = args.val_ground_truth
            self.val_u_i_embedding = tf.gather(self.embedding, self.val_edges[:, 0])
            self.val_u_j_embedding = tf.gather(self.embedding, self.val_edges[:, 1])
            self.neg_val_energy = tf.reduce_sum(self.val_u_i_embedding * self.val_u_j_embedding, axis=1)
            self.val_set = True
        else:
            self.val_set = False

        self.u_i_embedding = tf.gather(self.embedding, self.u_i)
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.gather(self.embedding, self.u_j)
        else:
            self.u_j_embedding = tf.gather(self.context_embedding, self.u_j)
        self.similarity = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.similarity))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def __create_model(self):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_emb = tf.get_variable(name='W_emb', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_emb = tf.get_variable(name='b_emb', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.embedding = tf.matmul(encoded, W_emb) + b_emb

        # context embedding for second-order proximity
        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_ctx{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b_ctx{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                ctx_encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                ctx_encoded = tf.matmul(ctx_encoded, W) + b

            ctx_encoded = tf.nn.relu(ctx_encoded)

        W_emb = tf.get_variable(name='W_ctx_emb', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_emb = tf.get_variable(name='b_ctx_emb', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.context_embedding = tf.matmul(ctx_encoded, W_emb) + b_emb


class GLACE:
    def __init__(self, args):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.X = tf.SparseTensor(*sparse_feeder(args.X))
        self.N, self.D = args.X.shape
        self.L = args.embedding_dim
        self.n_hidden = [512]

        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])

        self.__create_model(args.proximity)

        if not args.is_all:
            self.val_edges = args.val_edges
            self.val_ground_truth = args.val_ground_truth
            self.neg_val_energy = -self.energy_kl(self.val_edges[:, 0], self.val_edges[:, 1], args.proximity)
            self.val_set = True
        else:
            self.val_set = False

        # softmax loss
        self.energy = -self.energy_kl(self.u_i, self.u_j, args.proximity)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.energy))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def __create_model(self, proximity):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.embedding = tf.matmul(encoded, W_mu) + b_mu

        W_sigma = tf.get_variable(name='W_sigma', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=tf.float32, initializer=w_init())
        log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
        self.sigma = tf.nn.elu(log_sigma) + 1 + 1e-14

        if proximity == 'second-order':
            for i in range(1, len(sizes)):
                W = tf.get_variable(name='W_ctx{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                    initializer=w_init())
                b = tf.get_variable(name='b_ctx{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

                if i == 1:
                    encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
                else:
                    encoded = tf.matmul(encoded, W) + b

                encoded = tf.nn.relu(encoded)

            W_mu = tf.get_variable(name='W_mu_ctx', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
            b_mu = tf.get_variable(name='b_mu_ctx', shape=[self.L], dtype=tf.float32, initializer=w_init())
            self.ctx_mu = tf.matmul(encoded, W_mu) + b_mu

            W_sigma = tf.get_variable(name='W_sigma_ctx', shape=[sizes[-1], self.L], dtype=tf.float32,
                                      initializer=w_init())
            b_sigma = tf.get_variable(name='b_sigma_ctx', shape=[self.L], dtype=tf.float32, initializer=w_init())
            log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
            self.ctx_sigma = tf.nn.elu(log_sigma) + 1 + 1e-14

    def energy_kl(self, u_i, u_j, proximity):
        mu_i = tf.gather(self.embedding, u_i)
        sigma_i = tf.gather(self.sigma, u_i)

        if proximity == 'first-order':
            mu_j = tf.gather(self.embedding, u_j)
            sigma_j = tf.gather(self.sigma, u_j)
        elif proximity == 'second-order':
            mu_j = tf.gather(self.ctx_mu, u_j)
            sigma_j = tf.gather(self.ctx_sigma, u_j)

        sigma_ratio = sigma_j / sigma_i
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)

        ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        sigma_ratio = sigma_i / sigma_j
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

        ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        kl_distance = 0.5 * (ij_kl + ji_kl)

        return kl_distance

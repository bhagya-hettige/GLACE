import tensorflow as tf
import argparse
from model import LACE, GLACE
from utils import DataUtils, score_link_prediction
import pickle
import time
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='cora_ml')
    parser.add_argument('model', default='glace', help='lace or glace')
    parser.add_argument('--suf', default='')
    parser.add_argument('--proximity', default='first-order', help='first-order or second-order')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--num_batches', type=int, default=100000)
    parser.add_argument('--is_all', default=False)  # train with all edges; no validation or test set
    args = parser.parse_args()
    args.is_all = True if args.is_all == 'True' else False
    train(args)


def train(args):
    graph_file = '/Users/bhagya/PycharmProjects/Old data/line-master data/%s/%s.npz' % (args.name, args.name)
    graph_file = graph_file.replace('.npz', '_train.npz') if not args.is_all else graph_file
    data_loader = DataUtils(graph_file, args.is_all)

    suffix = args.proximity
    args.X = data_loader.X if args.suf != 'oh' else sp.identity(data_loader.X.shape[0])
    if not args.is_all:
        args.val_edges = data_loader.val_edges
        args.val_ground_truth = data_loader.val_ground_truth

    m = args.model
    name = m + '_' + args.name
    if m == 'lace':
        model = LACE(args)
    elif 'glace' == m:
        model = GLACE(args)

    with tf.Session() as sess:
        print('-------------------------- ' + m + ' --------------------------')
        if model.val_set:
            print('batches\tloss\tval_auc\tval_ap\tsampling time\ttraining_time\tdatetime')
        else:
            print('batches\tloss\tsampling time\ttraining_time\tdatetime')

        tf.global_variables_initializer().run()
        sampling_time, training_time = 0, 0

        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label}
            t2 = time.time()
            sampling_time += t2 - t1

            loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)

            training_time += time.time() - t2

            if model.val_set:
                if b % 50 == 0:
                    val_energy = sess.run(model.neg_val_energy)
                    val_auc, val_ap = score_link_prediction(data_loader.val_ground_truth, val_energy)
                    print('%d\t%f\t%f\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, val_auc, val_ap, sampling_time, training_time,
                                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    sampling_time, training_time = 0, 0
            else:
                if b % 50 == 0:
                    print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    sampling_time, training_time = 0, 0

            if b % 50 == 0 or b == (args.num_batches - 1):
                if m == 'glace':
                    mu, sigma = sess.run([model.embedding, model.sigma])
                    pickle.dump({'mu': data_loader.embedding_mapping(mu),
                                 'sigma': data_loader.embedding_mapping(sigma)},
                                open('emb/%s%s_embedding_%s.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                    # if model.val_set:
                    #     r = kl_link_pred(mu, sigma, test_edges)
                    #     print('{:.4f}, {:.4f}'.format(r[0], r[1]))
                else:
                    embedding = sess.run(model.embedding)
                    pickle.dump(data_loader.embedding_mapping(embedding),
                                open('emb/%s%s_embedding_%s.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                    # if model.val_early_stopping:
                    #     r = link_prediction(test_edges, embedding)
                    #     print('{:.4f}, {:.4f}'.format(r[0], r[1]))


if __name__ == '__main__':
    main()

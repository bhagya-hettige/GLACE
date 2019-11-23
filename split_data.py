import numpy as np
import argparse
from utils import train_val_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='cora_ml')
    parser.add_argument('--p_val', type=float, default=0.05)
    parser.add_argument('--p_test', type=float, default=0.15)
    args = parser.parse_args()
    split(args)


def split(args):
    graph_file = 'data/%s/%s.npz' % (args.name, args.name)
    A, X, labels, val_edges, val_ground_truth, test_edges, test_ground_truth = train_val_test_split(graph_file, p_val=args.p_val, p_test=args.p_test)
    np.savez('data/%s/%s_train.npz' % (args.name, args.name), adj_data=A.data, adj_indices=A.indices,
             adj_indptr=A.indptr, adj_shape=A.shape, attr_data=X.data, attr_indices=X.indices,
             attr_indptr=X.indptr, attr_shape=X.shape, labels=labels, val_edges=val_edges,
             val_ground_truth=val_ground_truth, test_edges=test_edges,
             test_ground_truth=test_ground_truth)
    # test_edges = np.vstack((data_loader.test_edges.T, data_loader.test_ground_truth)).T.astype(np.int32)
    # np.savez('data/%s/%s_train_test.npz' % (args.name, args.name), train_edges=test_edges, test_edges=test_edges)
    print('%s train data saved' % args.name)


if __name__ == '__main__':
    main()

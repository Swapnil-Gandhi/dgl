import dgl
import numpy as np
import torch as th
import argparse
import time
from pyinstrument import Profiler
from load_graph import load_reddit, load_ogb
import os.path as osp

# Sample Command :
# python3 partition_graph.py --dataset law-uk-union --num_parts 4 --part_method random --balance_train --balance_edges

# OGB Metadata : https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/master.csv

def load_custom(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    meta_info = {}
    dir_name = '_'.join(name.split('-'))
    dir_name = dir_name + '_dgl'
    dir_name = osp.join('dataset', dir_name)
    meta_info['dir_path'] = dir_name
    print('dir_path is : ', str(dir_name))

    meta_info['add_inverse_edge'] = False
    meta_info['additional edge files'] = 'None'
    meta_info['additional node files'] = 'None'
    meta_info['binary'] = False
    meta_info['download_name'] = 'uk-union'
    meta_info['eval metric'] = 'acc'
    meta_info['has_edge_attr'] = False
    meta_info['has_node_attr'] = False
    meta_info['is hetero'] = False
    meta_info['num classes'] = 172
    meta_info['num tasks'] = 1
    meta_info['split'] = 'time'
    meta_info['task type'] = 'multiclass classification'
    meta_info['url'] = None
    meta_info['version'] = 1


    print('load', name)
    data = DglNodePropPredDataset(name=name, meta_dict=meta_info)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit,ogb-protein,ogb-mag, ogb-product, ogb-paper100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    argparser.add_argument('--part_id', type=int, default=0,
                           help='subgraph partition to be generated')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogb-paper100M':
        g, _ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'ogb-protein':
        g, _ = load_ogb('ogbn-proteins')
    elif args.dataset == 'ogb-mag':
        g, _ = load_ogb('ogbn-mag')
    else:
        g, _ = load_custom(args.dataset)
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))
    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    in_feats = g.ndata['features'].shape[1]
    print('#Features', in_feats)

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
        print('Constructed Bidirected Graph:')
        print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))

    profiler = Profiler()
    profiler.start()
    if args.num_parts == 1 or args.part_method == 'metis' or args.part_method == 'random':
        dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    reshuffle=True,
                                    num_hops=1,
                                    feature_override=False,
                                    feature_len=128,
                                    part_override=False,
                                    partition_id=args.part_id)
    else:
        print('Not implemented')
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=True))

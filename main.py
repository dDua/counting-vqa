import argparse
import torch
import torch.nn as nn
import numpy as np
from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils
from base_model import TwoWayModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='two-way')
    parser.add_argument('--reload_exp', type=str, default='/home/ddua/workspace/bottom-up-attention-vqa/saved_models/two_regression')
    parser.add_argument('--output', type=str, default='saved_models/two_classifier')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--debug', default=False, action="store_true", help='random seed')
    parser.add_argument('--test_mode', default=False, action="store_true", help='random seed')
    parser.add_argument('--loss_fn', type=str, default='mse')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    
    eval_dset = VQAFeatureDataset('val', dictionary, debug=args.debug)
    train_dset = VQAFeatureDataset('train', dictionary, debug=args.debug)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    if args.model == "two-way":
        model = TwoWayModel(args.num_hid, train_dset.v_dim, \
                train_dset.v_numfeats, train_dset.dictionary.ntoken, \
                train_dset.num_ans_candidates)
    else:
        model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    if args.test_mode:
        model.load_state_dict(torch.load(args.reload_exp + "/model.pth"))

    if args.cuda:
        model = model.cuda()
    #model = nn.DataParallel(model).cuda()
#     model = model.cpu()
    
    print("Finished loading data")
    if not args.test_mode:
        train(model, train_dset, eval_dset, args.epochs, args.output, batch_size, args.loss_fn)
    else:
        eval_score, bound, acc = evaluate(model, eval_dset, batch_size, 0.7)
        print('eval score: %.2f (%.2f) acc: %.2f' % (100 * eval_score, 100 * bound, acc))
    

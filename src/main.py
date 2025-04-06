import time
# time.clock()默认单位为s
# 获取开始时间
start = time.clock()
import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)
parser = argparse.ArgumentParser()

# default settings for movie
'''
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings') # quatCNNEX也有
parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate') # quatCNNEX也有
parser.add_argument('--batch_size', type=int, default=1024, help='batch size') # quatCNNEX也有4358, 16115
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs') # quatCNNEX也有
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''
# quatCNNEX独有
# scoring_technique 
# label_smoothing  
# optim # Choose optimizer: Adam or RMSprop 
# decay_rate
# train_plus_valid
# input_dropout
# gamma
# hidden_dropout
# feature_map_dropout
# num_of_output_channels
# kernel_size
# path_dataset_folder
# num_workers
parser.add_argument('--scoring_technique', default='KvsAll',
                        help="KvsAll technique or Negative Sampling. For Negative Sampling, use any positive integer as input parameter")
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optim', type=str, default='RMSprop', help='Choose optimizer: Adam or RMSprop')
parser.add_argument('--decay_rate', default=None)
parser.add_argument('--train_plus_valid', default=False)
parser.add_argument('--input_dropout', type=float, default=0.1) #用了
parser.add_argument('--gamma', type=float, default=12.0, help='Distance parameter')
parser.add_argument('--hidden_dropout', type=float, default=0.1)
parser.add_argument('--feature_map_dropout', type=float, default=0.1) #用了
parser.add_argument('--num_of_output_channels', type=int, default=32) #用了
parser.add_argument('--kernel_size', type=int, default=3) #用了
parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS/')
parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
# default settings for Book-Crossing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')

# default settings for FM
'''
# parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.035, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=64, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''

parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
args = parser.parse_args()

show_loss = False
data_info = load_data(args)
train(args, data_info, show_loss)

# 获取结束时间
end = time.clock()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
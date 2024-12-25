from models.general.LightGCN import LGCNEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import threading
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.BaseModel import BaseModel 

from helpers import *
from time import time
from utility.batch_test import *
from utility.helper import *
from argparse import Namespace

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# cpus = ['/device:CPU:2']
print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

def generate_new_A(users,pos_items,neg_items,A, batch_size=None):
    """
    生成匹配的 new_A 矩阵，用于训练。

    参数：
    - data: Data 类的实例，包含用户、物品信息，以及 sample 方法等。
    - A: 稀疏矩阵，表示用户和物品之间的交互关系。
    - batch_size: 每批训练样本的大小（可选）。

    返回：
    - new_A: 用于训练的稀疏矩阵，具有匹配关系的用户和物品对。
    """


    # 创建一个新的稀疏矩阵，用于存储训练数据
    # 只需为当前批次的样本创建矩阵，因此新矩阵的形状是 len(users) x 2（正负样本）
    new_A = sp.dok_matrix((len(users), 2), dtype=np.float32)  # 形状为(batch_size, 2)

    # 为每个用户，添加对应的正样本和负样本
    for idx, (u, pos_item, neg_item) in enumerate(zip(users, pos_items, neg_items)):
        new_A[idx, 0] = 1.  # 正样本
        new_A[idx, 1] = -1.  # 负样本

    return new_A

def generate_new_A_test(users,pos_items,neg_items, A, batch_size=None):
    """
    生成匹配的 new_A 测试矩阵，用于测试集。

    参数：
    - data: Data 类的实例，包含用户、物品信息，以及 sample_test 方法等。
    - A: 稀疏矩阵，表示用户和物品之间的交互关系。
    - batch_size: 每批训练样本的大小（可选）。

    返回：
    - new_A_test: 用于测试的稀疏矩阵，具有匹配关系的用户和物品对。
    """
    # 获取 sample_test 方法生成的用户、正样本和负样本
    # users, pos_items, neg_items = data.sample_test()  # 默认使用 sample_test 方法



    # 创建一个新的稀疏矩阵，用于存储测试数据
    # 只需为当前批次的样本创建矩阵，因此新矩阵的形状是 len(users) x 2（正负样本）
    new_A_test = sp.dok_matrix((len(users), 2), dtype=np.float32)  # 形状为(batch_size, 2)

    # 为每个用户，添加对应的正样本和负样本
    for idx, (u, pos_item, neg_item) in enumerate(zip(users, pos_items, neg_items)):
        new_A_test[idx, 0] = 1.  # 正样本
        new_A_test[idx, 1] = -1.  # 负样本

    return new_A_test

def train(model, data_generator, args):
    # 初始化 TensorBoard
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    
    print(f"device:{device}")
    run_time = 1
    while True:
        
        if not os.path.exists(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)):
            os.makedirs(tensorboard_model_path + model.log_dir + '/run_' + str(run_time))
            run_time += 1
            print(f"run_time:{run_time}")
        # else:
        #     break

        train_writer = SummaryWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time))

        loss_logger, rec_logger, pre_logger, ndcg_logger, hit_logger = [], [], [], [], []
        stopping_step = 0
        should_stop = False

        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用 Adam 优化器
        criterion = nn.CrossEntropyLoss()  # 示例：使用交叉熵损失，可以根据需要调整

        for epoch in range(1, args.epoch + 1):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            loss_test, mf_loss_test, emb_loss_test, reg_loss_test = 0., 0., 0., 0.

            # 并行采样
            sample_last = sample_thread(data_generator)
            sample_last.start()
            sample_last.join()

            for idx in range(n_batch):
                # 启动训练线程
                train_cur = train_thread(model, optimizer, sample_last, criterion,device=device)
                sample_next = sample_thread(data_generator)

                train_cur.start()
                sample_next.start()

                sample_next.join()
                train_cur.join()
# (total_loss.item(), mf_loss.item(), emb_loss.item(), reg_loss.item())
                users, pos_items, neg_items = sample_last.data
                batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = train_cur.data
                sample_last = sample_next

                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            # TensorBoard 记录训练损失
            train_writer.add_scalar('Loss/train_loss', loss, epoch)
            train_writer.add_scalar('Loss/mf_loss', mf_loss, epoch)
            train_writer.add_scalar('Loss/emb_loss', emb_loss, epoch)

            if np.isnan(loss):
                print('ERROR: loss is NaN.')
                sys.exit()

            if (epoch % 5) != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[{loss:.5f}={mf_loss:.5f} + {emb_loss:.5f}]'
                    print(perf_str)
                continue

            # Test the model
            users_to_test = list(data_generator.train_items.keys())
            ret = test(model, users_to_test, drop_flag=True)

            # TensorBoard 记录测试准确率
            train_writer.add_scalar('Test/recall_first', ret['recall'][0], epoch)
            train_writer.add_scalar('Test/recall_last', ret['recall'][-1], epoch)
            train_writer.add_scalar('Test/ndcg_first', ret['ndcg'][0], epoch)
            train_writer.add_scalar('Test/ndcg_last', ret['ndcg'][-1], epoch)

            t2 = time()

            # Log performance
            loss_logger.append(loss)
            rec_logger.append(ret['recall'])
            pre_logger.append(ret['precision'])
            ndcg_logger.append(ret['ndcg'])

            if args.verbose > 0:
                perf_str = f'Epoch {epoch} [{t2 - t1:.1f}s + {time() - t2:.1f}s]: test==[{loss_test:.5f}={mf_loss_test:.5f} + {emb_loss_test:.5f} + {reg_loss_test:.5f}], ' \
                       f'recall=[{", ".join([f"{r:.5f}" for r in ret["recall"]])}], ' \
                       f'precision=[{", ".join([f"{r:.5f}" for r in ret["precision"]])}], ' \
                       f'ndcg=[{", ".join([f"{r:.5f}" for r in ret["ndcg"]])}]'
                print(perf_str)

            # Early stopping check
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

            if should_stop:
                break

            # Save model if performance improves
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                model.save(f"{args.weights_path}/weights_epoch_{epoch}.pt")
                print(f'Save the weights in path: {args.weights_path}/weights_epoch_{epoch}.pt')

        # Log final performance
        recs = np.array(rec_logger)
        pres = np.array(pre_logger)
        ndcgs = np.array(ndcg_logger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = f"Best Iter=[{idx}]@[{time() - t1:.1f}]\trecall=[{', '.join([f'{r:.5f}' for r in recs[idx]])}], " \
                 f"precision=[{', '.join([f'{r:.5f}' for r in pres[idx]])}], " \
                 f"ndcg=[{', '.join([f'{r:.5f}' for r in ndcgs[idx]])}]"
        print(final_perf)

        save_path = f'{args.proj_path}output/{args.dataset}/{model.model_type}.result'
        ensureDir(save_path)
        with open(save_path, 'a') as f:
            f.write(
                f'embed_size={args.embed_size}, lr={args.lr:.4f}, layer_size={args.layer_size}, '
                f'node_dropout={args.node_dropout}, mess_dropout={args.mess_dropout}, regs={args.regs}, '
                f'adj_type={args.adj_type}\n\t{final_perf}\n'
            )


def early_stopping(cur_best, cur_best_pre_0, stopping_step, expected_order='acc', flag_step=5):
    """
    Early stopping mechanism based on performance.
    :param cur_best: current best performance (e.g., recall[0])
    :param cur_best_pre_0: previous best performance
    :param stopping_step: current step of early stopping
    :param expected_order: 'acc' or 'dec' (whether the metric should increase or decrease)
    :param flag_step: number of steps to wait before stopping
    :return: new cur_best, stopping_step, should_stop (boolean)
    """
    if expected_order == 'acc':
        if cur_best > cur_best_pre_0:
            cur_best_pre_0 = cur_best
            stopping_step = 0
        else:
            stopping_step += 1
    else:
        if cur_best < cur_best_pre_0:
            cur_best_pre_0 = cur_best
            stopping_step = 0
        else:
            stopping_step += 1

    should_stop = stopping_step > flag_step
    return cur_best_pre_0, stopping_step, should_stop

# 假设你已经有一个 test() 函数用于在模型上进行评估
def report_performance(args, model, data_generator):
    if args.report == 1:
        # 检查测试标志，确保进行的是完整的评估
        assert args.test_flag == 'full'
        
        # 获取不同稀疏性级别的用户拆分
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))  # 添加 'all' 用户
        split_state.append('all')

        # 创建报告路径
        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        
        # 打开文件写入报告
        with open(report_path, 'w') as f:
            f.write(
                'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
                % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type)
            )

            # 对每个稀疏度级别进行测试
            for i, users_to_test in enumerate(users_to_test_list):
                model.eval()  # 切换到评估模式
                with torch.no_grad():  # 禁用梯度计算
                    ret = test(model, users_to_test, drop_flag=True)  # 测试，具体的 test 函数需要根据情况调整
                    
                    # 格式化输出性能指标
                    final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                                 (', '.join(['%.5f' % r for r in ret['recall']]),
                                  ', '.join(['%.5f' % r for r in ret['precision']]),
                                  ', '.join(['%.5f' % r for r in ret['ndcg']]))
                    
                    # 写入报告文件
                    f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        
        # 退出程序
        exit()


class ExtendedLightGCN(LGCNEncoder):
    def __init__(self, args, corpus, data_config, pretrain_data):
        super(ExtendedLightGCN, self).__init__(data_config['n_users'],data_config['n_items'], args.emb_size, data_config['norm_adj'])
        self.model_type = 'LightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 256
        # self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        # self.emb_dim = args.emb_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.corpus=corpus
        # Embedding layers
        # self.user_embedding = self.embedding_dict['user_embed']
        # self.item_embedding = self.embedding_dict['item_embed']

        if self.pretrain_data is not None:
            self.embedding_dict['user_emb'].data = torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32)
            self.embedding_dict['item_emb'].data = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32)
            # self.u_g_embeddings = self.embedding_dict['user_emb'][users]
            # self.pos_i_g_embeddings = self.embedding_dict['item_emb'][pos_items]
            # self.neg_i_g_embeddings = self.embedding_dict['item_emb'][neg_items]
        else:
            users, pos_items, neg_items = self.corpus.sample()
            users=self.corpus.exist_users
            items=self.corpus.train_items
            all_items = []
            for user, item in items.items():
                all_items.extend(item)  # 将每个用户的物品列表添加到总列表
            # self.u_g_embeddings = torch.tensor(users, dtype=torch.float32)
            # self.pos_i_g_embeddings = torch.tensor(pos_items, dtype=torch.float32)
            # self.neg_i_g_embeddings = torch.tensor(neg_items, dtype=torch.float32)
            print(len(users))
            print(len(items))
            # Ensure pos_items and neg_items are tensors
            # pos_items = torch.tensor(pos_items, dtype=torch.long)
            # neg_items = torch.tensor(neg_items, dtype=torch.long)
            # user_items = torch.tensor(users, dtype=torch.long)

            # Add a new dimension to distinguish between positive and negative items
            # pos_items = pos_items.unsqueeze(1)  # Shape (batch_size, 1)
            # neg_items = neg_items.unsqueeze(1)  # Shape (batch_size, 1)

            # # Concatenate pos_items and neg_items along the second dimension
            # items = torch.cat([pos_items, neg_items], dim=1)  # Shape (batch_size, 2)
            # print(items.shape)
            # print(user_items.shape)
            self.embedding_dict['user_emb'].data = torch.tensor(users, dtype=torch.float32)
            self.embedding_dict['item_emb'].data = torch.tensor(all_items, dtype=torch.float32)
            
        # Layer weights
        self.weights = self._init_weights()
        self.log_dir = self.create_model_str()
        # Create adjacency matrix if needed
        self.A_fold_hat = self._split_A_hat(self.norm_adj)
    def create_model_str(self):
        log_dir = '/' + self.alg_type + '/layers_' + str(self.n_layers) + '/dim_' + str(self.emb_size)
        log_dir += '/' + args.dataset + '/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir
    def _init_weights(self):
        weights = {}
        for k in range(self.n_layers):
            weights[f'W_gc_{k}'] = nn.Parameter(torch.randn(self.emb_size, self.emb_size) * 0.01)
            weights[f'b_gc_{k}'] = nn.Parameter(torch.randn(self.emb_size) * 0.01)

            weights[f'W_bi_{k}'] = nn.Parameter(torch.randn(self.emb_size, self.emb_size) * 0.01)
            weights[f'b_bi_{k}'] = nn.Parameter(torch.randn(self.emb_size) * 0.01)

        return weights

    def _split_A_hat(self, X):
        fold_len = (self.user_count + self.item_count) // self.n_fold
        A_fold_hat = []
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = (i_fold + 1) * fold_len if i_fold != self.n_fold - 1 else self.n_users + self.n_items
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to('cuda'))
        return A_fold_hat
    def forward(self, users, pos_items, neg_items):
        # Compute embeddings for users and items
        neg_items = neg_items.to(torch.long)
        u_g_embeddings = self.embedding_dict['user_emb'].data[users]
        pos_i_g_embeddings = self.embedding_dict['item_emb'].data[pos_items]
        neg_i_g_embeddings = self.embedding_dict['item_emb'].data[neg_items]
        new_A=generate_new_A(u_g_embeddings,pos_i_g_embeddings,neg_i_g_embeddings,self.A_fold_hat)
        # 在进行矩阵乘法之前，确保 u_g_embeddings 在 CPU 上
        u_g_embeddings = u_g_embeddings.cpu()  # 将 u_g_embeddings 移动到 CPU
        new_u_g_embeddings = u_g_embeddings.unsqueeze(1)  # 形状变为 (512, 1)
        # 如果 new_A 是稀疏矩阵（如 dok_matrix），可以转换为稠密矩阵
        new_A_dense = new_A.toarray()  # 将稀疏矩阵转换为稠密矩阵

        # 转换为 PyTorch 张量
        new_A_tensor = torch.tensor(new_A_dense, dtype=torch.float32)
        # Compute the LightGCN embeddings
        for k in range(self.n_layers):
            temp_embed = []
            print(f"newA:{new_A.shape}")  # 打印 A_fold_hat[f] 的形状
            print(f"new_u:{new_u_g_embeddings.shape}")  # 打印 u_g_embeddings 的形状
            for f in range(self.n_fold):
                temp_embed.append(new_A_tensor.t() @ new_u_g_embeddings)

            side_embeddings = torch.cat(temp_embed, 0)
            new_u_g_embeddings = side_embeddings
        u_g_embeddings = u_g_embeddings.to('cuda')
        print(f"use:{u_g_embeddings.shape}")
        print(f"pos:{pos_i_g_embeddings.shape}")
        print(f"neg:{neg_i_g_embeddings.shape}")
        pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1)
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=1)

        # Compute regularization loss
        reg_loss = self.decay * (u_g_embeddings.norm(2) + pos_i_g_embeddings.norm(2) + neg_i_g_embeddings.norm(2))

        # BPR Loss
        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))
        emb_loss = reg_loss

        total_loss = mf_loss + emb_loss
        return total_loss, mf_loss, emb_loss, reg_loss

    def create_bpr_loss1(self, users, users1, pos_items, pos_items1, neg_items, neg_items1):
        neg_items = neg_items.to(torch.long)
        neg_items1 = neg_items1.to(torch.long)
        u_g_embeddings = self.embedding_dict['user_emb'].data[users]
        pos_i_g_embeddings = self.embedding_dict['item_emb'].data[pos_items]
        neg_i_g_embeddings = self.embedding_dict['item_emb'].data[neg_items]
        u_g_embeddings1 = self.embedding_dict['user_emb'].data[users1]
        pos_i_g_embeddings1 = self.embedding_dict['item_emb'].data[pos_items1]
        neg_i_g_embeddings1 = self.embedding_dict['item_emb'].data[neg_items1]
        # u_g_embeddings = self.u_g_embeddings[users]
        # pos_i_g_embeddings = self.pos_i_g_embeddings[pos_items]
        # neg_i_g_embeddings = self.neg_i_g_embeddings[neg_items]
        # u_g_embeddings1 = self.u_g_embeddings[users1]
        # pos_i_g_embeddings1 = self.pos_i_g_embeddings[pos_items1]
        # neg_i_g_embeddings1 = self.neg_i_g_embeddings[neg_items1]
        print(f"u:::{u_g_embeddings.shape}")  # 查看 u_g_embeddings 的形状
        print(f"i:::{neg_i_g_embeddings.shape}")  # 查看 neg_i_g_embeddings 的形状

        pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=0)
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=0)
        pos_scores1 = torch.sum(u_g_embeddings1 * pos_i_g_embeddings1, dim=0)
        neg_scores1 = torch.sum(u_g_embeddings1 * neg_i_g_embeddings1, dim=0)

        pos_scores = self.Lambda * pos_scores + (1 - self.Lambda) * pos_scores1
        neg_scores = self.Lambda * neg_scores + (1 - self.Lambda) * neg_scores1
        # pos_scores = pos_scores.unsqueeze(1)  # (64, 1)
        # neg_scores = neg_scores.view(-1, 64)  # 假设每个用户有多个负样本
        # Regularizer loss
        reg_loss = self.decay * (u_g_embeddings.norm(2) + pos_i_g_embeddings.norm(2) + neg_i_g_embeddings.norm(2) +
                                 u_g_embeddings1.norm(2) + pos_i_g_embeddings1.norm(2) + neg_i_g_embeddings1.norm(2))

        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))
        emb_loss = reg_loss

        total_loss = mf_loss + emb_loss
        return total_loss, mf_loss, emb_loss, reg_loss
def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='0',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--save_final_results', type=int, default=1,
						help='To save the final validation and test results or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	return parser

# 确保创建文件夹
def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


# Sample Thread for training
class sample_thread(threading.Thread):
    def __init__(self, data_generator):
        threading.Thread.__init__(self)
        self.data_generator = data_generator

    def run(self):
        self.data = self.data_generator.sample()

# Sample Thread for testing
class samplethreadtest(threading.Thread):
    def __init__(self, data_generator):
        threading.Thread.__init__(self)
        self.data_generator = data_generator

    def run(self):
        self.data = self.data_generator.sample_test()

# Train Thread (GPU/CPU)
class train_thread(threading.Thread):
    def __init__(self, model, optimizer, sample, criterion, device):
        threading.Thread.__init__(self)
        self.model = model        # 模型
        self.optimizer = optimizer  # 优化器
        self.sample = sample      # 数据采样器（提供数据）
        self.criterion = criterion  # 损失函数
        self.device = device       # 设备（CPU 或 GPU）
        self.data = None            # 训练数据

    def run(self):
        # data_set=self.sample.data
        # 从数据生成器获取用户、正样本和负样本数据
        users, pos_items, neg_items = self.sample.data
        data_set = list(zip(users, pos_items, neg_items))
        # 将数据转化为张量，并将它们移动到正确的设备上（CPU 或 GPU）
        users = torch.tensor(users).to(self.device)
        pos_items = torch.tensor(pos_items).to(self.device)
        neg_items = torch.tensor(neg_items).to(self.device)
        
        # 生成 Lambda 用于正则化
        Lambda = torch.abs(torch.tensor(np.random.beta(0.5, 0.5, (len(users), 1))) - 0.5) + 0.5
        Lambda = Lambda.to(self.device)
        self.model.Lambda=Lambda
        if len(pos_items.shape) == 1:
            pos_items = pos_items.unsqueeze(1)  # 把 items 转换为二维张量 (batch_size, 1)
        # 清零梯度
        accumulation_steps = 4
        self.optimizer.zero_grad()
        # dataloader=DataLoader(data_set, batch_size=16, shuffle=True)
        # for step, batch in enumerate(dataloader):
        #     if (step + 1) % accumulation_steps == 0:
        #         self.optimizer.step()
        #         self.optimizer.zero_grad()
        #     feed_dict = {
        #         'user_id': users,         # 用户 ID
        #         'item_id': pos_items,     # 正样本物品 ID
        #         'batch_size': len(users), # 批次大小
        #     }
        # # 前向传播（计算预测值）
        # # loss, mf_loss, emb_loss, reg_loss = self.model(users, pos_items, neg_items, Lambda)
        # # loss, mf_loss, emb_loss, reg_loss = self.model.forward(feed_dict)
        #     total_loss,mf_loss, emb_loss, reg_loss=self.model.create_bpr_loss1(users, users, pos_items, pos_items, neg_items, neg_items)
        #     prediction = self.model.forward(users, pos_items,neg_items)
        #     # loss=mf_loss+emb_loss+reg_loss
        #     # 计算总损失
        #     # total_loss = loss + mf_loss + emb_loss + reg_loss

        #     # 反向传播（计算梯度）
        #     total_loss.backward()

        # # 更新模型参数
        # self.optimizer.step()


        self.optimizer.step()
        self.optimizer.zero_grad()
        feed_dict = {
            'user_id': users,         # 用户 ID
            'item_id': pos_items,     # 正样本物品 ID
            'batch_size': len(users), # 批次大小
        }
        # 前向传播（计算预测值）
        # loss, mf_loss, emb_loss, reg_loss = self.model(users, pos_items, neg_items, Lambda)
        # loss, mf_loss, emb_loss, reg_loss = self.model.forward(feed_dict)
        total_loss,mf_loss, emb_loss, reg_loss=self.model.create_bpr_loss1(users, users, pos_items, pos_items, neg_items, neg_items)
        prediction = self.model.forward(users, pos_items,neg_items)
            # loss=mf_loss+emb_loss+reg_loss
            # 计算总损失
            # total_loss = loss + mf_loss + emb_loss + reg_loss

            # 反向传播（计算梯度）
        total_loss.backward()

        # 更新模型参数
        self.optimizer.step()
        # 将训练结果保存到线程中
        self.data = (total_loss.item(), mf_loss.item(), emb_loss.item(), reg_loss.item())

# Train Thread for testing (GPU/CPU)
class train_thread_test(threading.Thread):
    def __init__(self, model, sample, device):
        threading.Thread.__init__(self)
        self.model = model
        self.sample = sample
        self.device = device

    def run(self):
        users, pos_items, neg_items = self.sample.data
        idx = np.random.permutation(len(users))  # Shuffle for randomness
        users = torch.tensor(users[idx]).to(self.device)
        pos_items = torch.tensor(pos_items[idx]).to(self.device)
        neg_items = torch.tensor(neg_items[idx]).to(self.device)

        Lambda = torch.abs(torch.tensor(np.random.beta(0.5, 0.5, (len(users), 1))) - 0.5) + 0.5
        Lambda = Lambda.to(self.device)

        # Running the model
        loss, mf_loss, emb_loss = self.model.forward_test(users, pos_items, neg_items, Lambda)
        
        self.data = (loss, mf_loss, emb_loss)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()  # 这里仅加载预训练数据
        # corpus = Data(path=args.data_path+args.dataset, batch_size=args.batch_size)  # 加载实际的训练数据集
    elif args.pretrain == 0:
        pretrain_data = None
        # corpus = Data(path=args.data_path+args.dataset, batch_size=args.batch_size)
    # args = Namespace()  # 创建一个 Namespace 实例
    # args.model_path = './saved_model'  # 手动设置模型路径
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device 属性
    model = ExtendedLightGCN(args=args, corpus=data_generator,data_config=config, pretrain_data=pretrain_data).to(device)

    # model = LightGCN(data_config=config, pretrain_data=pretrain_data)
    # model = LightGCN(args=args,corpus=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    # saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        model.save_model(model_path=weights_save_path)



    """
    *********************************************************
    Reload the pretrained model parameters.
    """

    # Assuming your model is defined as 'model' and on the selected device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrain == 1:
        # 根据传入的参数生成路径
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                    str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        # 检查是否存在预训练模型
        pretrained_model_path = os.path.join(pretrain_path, 'model.pth')

        if os.path.exists(pretrained_model_path):
            # 加载预训练模型
            model.load_state_dict(torch.load(pretrained_model_path))
            model = model.to(device)
            print(f'Loaded pretrained model parameters from: {pretrained_model_path}')

            # *********************************************************
            # 使用预训练模型进行评估
            if args.report != 1:
                model.eval()  # 切换到评估模式
                with torch.no_grad():  # 在评估阶段禁用梯度计算
                    users_to_test = list(data_generator.test_set.keys())
                    ret = test(model, users_to_test, drop_flag=True)
                    cur_best_pre_0 = ret['recall'][0]

                    pretrain_ret = 'pretrained model recall=[%s], precision=[%s], ' \
                                'ndcg=[%s]' % (
                                   ', '.join(['%.5f' % r for r in ret['recall']]),
                                   ', '.join(['%.5f' % r for r in ret['precision']]),
                                   ', '.join(['%.5f' % r for r in ret['ndcg']])
                               )
                    print(pretrain_ret)

        else:
            print('Pretrained model not found. Initializing model from scratch.')
            cur_best_pre_0 = 0.

    else:
        # 没有预训练模型时，直接初始化
        print('Without pretraining.')
        cur_best_pre_0 = 0.


    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    report_performance(args, model, data_generator)
    """
    *********************************************************
    Train.
    """
    train(model,data_generator=data_generator,args=args)



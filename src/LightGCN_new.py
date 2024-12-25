# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from helpers import *
from models.general import *
from models.general.LightGCN import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils
from helpers.BaseRunner import BaseRunner
# from helpers.BaseReader import BaseReader
import numpy as np
from typing import Dict, List

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
	parser.add_argument('--normalization',type=str,default="Adjacency_matrix_with_self-loops",
					   help='Normalization method for the adjacency matrix')
	parser.add_argument('--activation_function',type=str,default="",
					   help='Activation function for the model')
	parser.add_argument('--result_file',type=str,default="",
						help='File to save the final results')
	parser.add_argument('--hidden_layer_size',type=int,default=0,
					   help='Size of the hidden layer')
	return parser


class NewEncoder(LGCNEncoder):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, hidden_size=128, dropout_rate=0.2,activation=nn.LeakyReLU()):
        # 调用父类的初始化方法
        super(NewEncoder, self).__init__(user_count, item_count, emb_size, norm_adj, n_layers)
        
        # 添加额外的隐藏层
        self.hidden_layer = nn.Linear(emb_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, users, items):
        # 使用父类的方法获取基本的用户和物品嵌入
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        # NGCF 特有的特征交互逻辑
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(self.encoder.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        # 聚合多层嵌入
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        # 隐藏层处理（添加非线性变换和 dropout）
        all_embeddings = self.activation(self.hidden_layer(all_embeddings))
        all_embeddings = self.dropout(all_embeddings)

        # 分割用户和物品嵌入
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings



class NewLightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=True):
		rows,cols,data=[],[],[]
		for user,items in train_mat.items():
			rows.extend([user]*len(items))
			cols.extend(items)
			data.extend([1]*len(items))
		R=sp.coo_matrix((data,(rows,cols)),shape=(user_count,item_count),dtype=np.float32)
		adj_mat_upper=sp.hstack([sp.coo_matrix((user_count, user_count)), R])
		adj_mat_lower=sp.hstack([R.T, sp.coo_matrix((item_count, item_count))])
		adj_mat = sp.vstack([adj_mat_upper, adj_mat_lower]).tocoo()
		def normalized_adj_single(adj):
			normalization = args.normalization
			if normalization=="Adjacency_matrix_with_self-loops":
				bi_lap = adj
			elif normalization=="Row-normalized_adjacency_matrix":
				rowsum = np.array(adj.sum(1)) + 1e-10
				d_inv_sqrt = np.power(rowsum, -1.0).flatten()
				d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
				d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
				bi_lap = d_mat_inv_sqrt.dot(adj)
			elif normalization=="Symmetrically_normalized_adjacency_matrix":
				# D^-1/2 * A * D^-1/2
				rowsum = np.array(adj.sum(1)) + 1e-10

				d_inv_sqrt = np.power(rowsum, -0.5).flatten()
				d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
				d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

				bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			else:
				raise ValueError(f"Unknown normalization type: {normalization}")
			return bi_lap

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat) + sp.eye(adj_mat.shape[0])
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.activation_function = args.activation_function
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		if args.hidden_layer_size==0:
			self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
		else:
			print("*******************************************************************************************newencoder**********************************************************************************************************")
			logging.info(f"hidden_layer_size:{args.hidden_layer_size}")
			self.encoder = NewEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers,activation=self.activation_function)

	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)
		if self.activation_function is not None:
			u_embed, i_embed = self.activation_function(u_embed), self.activation_function(i_embed)
		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

# class CustomReader(BaseReader):
# 	def __init__(self, args):
# 		super().__init__(args)

# 	def _read_data(self):
# 		logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
# 		self.data_df = dict()
# 		for key in ['train', 'dev', 'test']:
# 			self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id'])
# 			self.data_df[key] = utils.eval_list_columns(self.data_df[key])

# 		logging.info('Counting dataset statistics...')
# 		key_columns = ['user_id','item_id']
# 		if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
# 			key_columns.append('label')
# 		self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
# 		self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
# 		for key in ['dev', 'test']:
# 			if 'neg_items' in self.data_df[key]:
# 				neg_items = np.array(self.data_df[key]['neg_items'].tolist())
# 				assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
# 		logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
# 			self.n_users - 1, self.n_items - 1, len(self.all_df)))
# 		if 'label' in key_columns:
# 			positive_num = (self.all_df.label==1).sum()
# 			logging.info('"# positive interaction": {} ({:.1f}%)'.format(
# 				positive_num, positive_num/self.all_df.shape[0]*100))

class CustomRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
        evaluations = dict()
		# sort_idx = (-predictions).argsort(axis=1)
		# gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
		# ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
        gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		# if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
		# 	predictions_rnd = predictions.copy()
		# 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
		# 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
        print(metrics)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                elif metric == 'RECALL':
                    evaluations[key] = hit.sum() / len(gt_rank)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations




class CustomLightGCN(GeneralModel, NewLightGCNBase):
	runner = 'CustomRunner'
	reader = 'BaseReader'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = NewLightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		# GeneralModel.__init__(self, args, corpus)
		super().__init__(args,corpus)
		super()._base_init(args, corpus)
		self.hidden_size=args.hidden_layer_size
		self.activation_function=args.activation_function
		if self.activation_function=="ReLU":
			self.activation=nn.ReLU()
		elif self.activation_function=="Tanh":
			self.activation=nn.Tanh()
		elif self.activation_function=="Sigmoid":
			self.activation=nn.Sigmoid()
		elif self.activation_function=="LeakyReLU":
			self.activation=nn.LeakyReLU()
		elif self.activation_function == "None" or self.activation_function is None or self.activation_function is "":
			self.activation = None
		else :
			raise ValueError("Unknown activation function: {}".format(self.activation_function))

	def forward(self, feed_dict):
		user, items = feed_dict['user_id'], feed_dict['item_id']
    	# 使用父类的方法获取基本的用户和物品嵌入
		ego_embeddings = torch.cat([self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		# NGCF 特有的特征交互逻辑
		for k in range(len(self.encoder.layers)):
			ego_embeddings = torch.sparse.mm(self.encoder.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		# 聚合多层嵌入
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

    	# 隐藏层处理（添加非线性变换和 dropout）
		if args.hidden_layer_size > 0:
			all_embeddings = self.activation(self.encoder.hidden_layer(all_embeddings))
			all_embeddings = self.dropout(all_embeddings)

    	# 获取最终用户和物品嵌入
		user_all_embeddings = all_embeddings[:self.encoder.user_count, :]
		item_all_embeddings = all_embeddings[self.encoder.user_count:, :]
		u_embed = user_all_embeddings[user, :]
		i_embed = item_all_embeddings[items, :]

        # 点积预测
		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
	




class LightGCNWithIMix(CustomLightGCN):
	def __init__(self, args, corpus):
		super().__init__(args,corpus=corpus)
		self.alpha=args.alpha if hasattr(args,'alpha') else 0.8
		self.imix_noise=args.imix_noise if hasattr(args,'imix_noise') else 0.1

	def apply_imix(self,u_embed,i_embed):
		perm_user=torch.randperm(u_embed.size(0)).to(u_embed.device)
		perm_item=torch.randperm(i_embed.size(0)).to(i_embed.device)
		u_embed_perm=u_embed[perm_user]
		i_embed_perm=i_embed[perm_item]
		lambda_mix=torch.rand(u_embed.size(0),1).to(u_embed.device)*self.alpha
		u_embed_mixed=lambda_mix*u_embed+(1-lambda_mix)*u_embed_perm
		lambda_mix=torch.rand(i_embed.size(0),1).to(i_embed.device)*self.alpha
		i_embed_mixed=lambda_mix*i_embed+(1-lambda_mix)*i_embed_perm
		noise=self.imix_noise*torch.randn_like(u_embed).to(u_embed.device)
		# u_embed_mixed+=noise
		noise=self.imix_noise*torch.randn_like(i_embed).to(i_embed.device)
		# i_embed_mixed+=noise
		return u_embed_mixed,i_embed_mixed

	def forward(self, feed_dict):
		user, items = feed_dict['user_id'], feed_dict['item_id']
    	# 使用父类的方法获取基本的用户和物品嵌入
		ego_embeddings = torch.cat([self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		# NGCF 特有的特征交互逻辑
		for k in range(len(self.encoder.layers)):
			ego_embeddings = torch.sparse.mm(self.encoder.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]
			user_emb = ego_embeddings[:self.encoder.user_count, :]
			item_emb = ego_embeddings[self.encoder.user_count:, :]
			user_emb_mixed, item_emb_mixed = self.apply_imix(user_emb, item_emb)
            # 合并混合的嵌入
			ego_embeddings = torch.cat([user_emb_mixed, item_emb_mixed], dim=0)
			all_embeddings.append(ego_embeddings)
		# 聚合多层嵌入
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		# 隐藏层处理（添加非线性变换和 dropout）
		if args.hidden_layer_size > 0:
			all_embeddings = self.activation(self.encoder.hidden_layer(all_embeddings))
			all_embeddings = self.dropout(all_embeddings)

    	# 获取最终用户和物品嵌入
		user_all_embeddings = all_embeddings[:self.encoder.user_count, :]
		item_all_embeddings = all_embeddings[self.encoder.user_count:, :]
		u_embed = user_all_embeddings[user, :]
		i_embed = item_all_embeddings[items, :]

        # 点积预测
		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}


class NGCF(CustomLightGCN):
    def __init__(self, args, corpus):
        super().__init__(args, corpus)

    def forward(self, feed_dict):
        """
        自定义的前向传播，基于 NGCF 的特征交互和非线性激活
        """
        # 获取用户和物品 ID
        user, items = feed_dict['user_id'], feed_dict['item_id']
        
        # 初始化嵌入
        ego_embeddings = torch.cat([self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']], dim=0)
        all_embeddings = [ego_embeddings]

        # 消息传播过程
        for _ in range(len(self.encoder.layers)):
            side_embeddings = torch.sparse.mm(self.encoder.encoder.sparse_norm_adj, ego_embeddings)  # 线性传播
            bi_embeddings = ego_embeddings * side_embeddings  # 特征交互（逐元素乘法）
            ego_embeddings = F.leaky_relu(side_embeddings + bi_embeddings)  # 非线性激活
            ego_embeddings = self.encoder.dropout(ego_embeddings)  # Dropout
            all_embeddings.append(ego_embeddings)

        # 聚合多层嵌入
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

		# 隐藏层处理（添加非线性变换和 dropout）
        if args.hidden_layer_size > 0:
            all_embeddings = self.activation(self.encoder.hidden_layer(all_embeddings))
            all_embeddings = self.dropout(all_embeddings)

        # 获取用户和物品嵌入
        user_all_embeddings = all_embeddings[:self.encoder.user_count, :]
        item_all_embeddings = all_embeddings[self.encoder.user_count:, :]
        u_embed = user_all_embeddings[user, :]
        i_embed = item_all_embeddings[items, :]

        # 点积预测
        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)

        return {'prediction': prediction}

class NGCFWithIMix(NGCF):
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.alpha=args.alpha if hasattr(args,'alpha') else 1
		self.imix_noise=args.imix_noise if hasattr(args,'imix_noise') else 0.1

	def apply_imix(self,u_embed,i_embed):
		perm_user=torch.randperm(u_embed.size(0)).to(u_embed.device)
		perm_item=torch.randperm(i_embed.size(0)).to(i_embed.device)
		u_embed_perm=u_embed[perm_user]
		i_embed_perm=i_embed[perm_item]
		lambda_mix=torch.rand(u_embed.size(0),1).to(u_embed.device)*self.alpha
		u_embed_mixed=lambda_mix*u_embed+(1-lambda_mix)*u_embed_perm
		lambda_mix=torch.rand(i_embed.size(0),1).to(i_embed.device)*self.alpha
		i_embed_mixed=lambda_mix*i_embed+(1-lambda_mix)*i_embed_perm
		noise=self.imix_noise*torch.randn_like(u_embed).to(u_embed.device)
		# u_embed_mixed+=noise
		noise=self.imix_noise*torch.randn_like(i_embed).to(i_embed.device)
		# i_embed_mixed+=noise
		return u_embed_mixed,i_embed_mixed
	def forward(self, feed_dict):
		user, items = feed_dict['user_id'], feed_dict['item_id']

        # 初始化嵌入
		ego_embeddings = torch.cat([self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']], dim=0)
		all_embeddings = [ego_embeddings]

        # 消息传播过程
		for layer in range(len(self.encoder.layers)):
            # 基于 NGCF 的特征交互
			side_embeddings = torch.sparse.mm(self.encoder.sparse_norm_adj, ego_embeddings)
			bi_embeddings = ego_embeddings * side_embeddings  # 特征交互
			ego_embeddings = F.leaky_relu(side_embeddings + bi_embeddings)  # 非线性激活
            
            # **IMix 数据增强**：在每一层传播后，混合用户和物品嵌入
			user_emb = ego_embeddings[:self.encoder.user_count, :]
			item_emb = ego_embeddings[self.encoder.user_count:, :]
			user_emb_mixed, item_emb_mixed = self.apply_imix(user_emb, item_emb)

            # 合并混合的嵌入
			ego_embeddings = torch.cat([user_emb_mixed, item_emb_mixed], dim=0)
			all_embeddings.append(ego_embeddings)

        # 聚合多层嵌入
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)
		# 隐藏层处理（添加非线性变换和 dropout）
		if args.hidden_layer_size > 0:
			all_embeddings = self.activation(self.encoder.hidden_layer(all_embeddings))
			all_embeddings = self.dropout(all_embeddings)
        # 获取最终用户和物品嵌入
		user_all_embeddings = all_embeddings[:self.encoder.user_count, :]
		item_all_embeddings = all_embeddings[self.encoder.user_count:, :]
		u_embed = user_all_embeddings[user, :]
		i_embed = item_all_embeddings[items, :]

        # 点积预测
		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

def main():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
			   'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))

	# Random seed
	utils.init_seed(args.random_seed)

	# GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data
	corpus_path = os.path.join(args.path, args.dataset, model_name.reader+args.data_appendix+ '.pkl')
	print(f"corpus_path: {corpus_path}")
	if not args.regenerate and os.path.exists(corpus_path):
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
	else:
		corpus = reader_name(args)
		for phase in ['train', 'dev', 'test']:
			df=corpus.data_df[phase]
			df['user_id'] = df['user_id'].astype('int64')
			df['item_id'] = df['item_id'].astype('int64')
			corpus.data_df[phase] = df
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model
	model = model_name(args, corpus).to(args.device)
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)

	# Define dataset
	data_dict = dict()
	for phase in ['train', 'dev', 'test']:
		data_dict[phase] = model_name.Dataset(model, corpus, phase)
		data_dict[phase].prepare()

	# Run model
	runner = runner_name(args)
	logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
	if args.normalization is not None:
		logging.info('normalization:'+args.normalization+"*************************************************************************************************************************************")
	if args.activation_function is not None:
		logging.info('activation:'+args.activation_function+"***********************************************************************************************************************************************")
	if args.load > 0:
		model.load_model()
	if args.train > 0:
		runner.train(data_dict)
	result_file=args.result_file
	# Evaluate final results
	eval_res = runner.print_res(data_dict['dev'])
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	eval_res = runner.print_res(data_dict['test'])
	logging.info(os.linesep + 'Test After Training: ' + eval_res)

	with open(result_file, "a") as f:
		f.write("Test After Training: "+" layers: "+str(args.n_layers) +" weight: "+str(args.l2) +" "+ eval_res + "\n")

	if args.save_final_results==1: # save the prediction results
		save_rec_results(data_dict['dev'], runner, 100)
		save_rec_results(data_dict['test'], runner, 100)
	model.actions_after_train()
	logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def save_rec_results(dataset, runner, topk):
	model_name = '{0}{1}'.format(init_args.model_name,init_args.model_mode)
	result_path = os.path.join(runner.log_path,runner.save_appendix, 'rec-{}-{}.csv'.format(model_name,dataset.phase))
	utils.check_dir(result_path)

	if init_args.model_mode == 'CTR': # CTR task 
		logging.info('Saving CTR prediction results to: {}'.format(result_path))
		predictions, labels = runner.predict(dataset)
		users, items= list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			items.append(info['item_id'][0])
		rec_df = pd.DataFrame(columns=['user_id', 'item_id', 'pCTR', 'label'])
		rec_df['user_id'] = users
		rec_df['item_id'] = items
		rec_df['pCTR'] = predictions
		rec_df['label'] = labels
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['TopK','']: # TopK Ranking task
		logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, rec_items, rec_predictions = list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			item_scores = zip(info['item_id'], predictions[i])
			sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
			rec_items.append([x[0] for x in sorted_lst])
			rec_predictions.append([x[1] for x in sorted_lst])
		rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
		rec_df['user_id'] = users
		rec_df['rec_items'] = rec_items
		rec_df['rec_predictions'] = rec_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['Impression','General','Sequential']: # List-wise reranking task: Impression is reranking task for general/seq baseranker. General/Sequential is reranking task for rerankers with general/sequential input.
		logging.info('Saving all recommendation results to: {}'.format(result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, pos_items, pos_predictions, neg_items, neg_predictions= list(), list(), list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			pos_items.append(info['pos_items'])
			neg_items.append(info['neg_items'])
			pos_predictions.append(predictions[i][:dataset.pos_len])
			neg_predictions.append(predictions[i][:dataset.neg_len])
		rec_df = pd.DataFrame(columns=['user_id', 'pos_items', 'pos_predictions', 'neg_items', 'neg_predictions'])
		rec_df['user_id'] = users
		rec_df['pos_items'] = pos_items
		rec_df['pos_predictions'] = pos_predictions
		rec_df['neg_items'] = neg_items
		rec_df['neg_predictions'] = neg_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	else:
		return 0
	logging.info("{} Prediction results saved!".format(dataset.phase))

if __name__ == '__main__':
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='CustomLightGCN', help='Choose a model to run.')
	init_parser.add_argument('--model_mode', type=str, default='', 
							 help='Model mode(i.e., suffix), for context-aware models to select "CTR" or "TopK" Ranking task;\
            						for general/seq models to select Normal (no suffix, model_mode="") or "Impression" setting;\
                  					for rerankers to select "General" or "Sequential" Baseranker.')
	init_args, init_extras = init_parser.parse_known_args()
	

	if init_args.model_name == "CustomLightGCN":
		model_name = CustomLightGCN  # 直接引用自定义类
	elif init_args.model_name == "LightGCNWithIMix":
		model_name = LightGCNWithIMix  # 直接引用自定义类
	elif init_args.model_name == "NGCF":
		model_name = NGCF  # 直接引用自定义类
	elif init_args.model_name == "NGCFWithIMix":
		model_name = NGCFWithIMix  # 直接引用自定义类
	else:
		model_name = eval('{0}.{0}{1}'.format(init_args.model_name, init_args.model_mode))
	# model_name = eval('{0}.{0}{1}'.format(init_args.model_name,init_args.model_mode))
	if model_name.reader=="CustomReader":
		reader_name = CustomReader  # 映射到自定义 Reader
	else:
		reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
	if model_name.runner == "CustomRunner":
		runner_name = CustomRunner  # 映射到自定义 Runner
	else:
		runner_name = eval('{0}.{0}'.format(model_name.runner))
	# runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	
	args.data_appendix = '' # save different version of data for, e.g., context-aware readers with different groups of context
	if 'Context' in model_name.reader:
		args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
										args.include_situation_features)

	# Logging configuration
	log_args = [init_args.model_name+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')
	if args.log_file == '':
		args.log_file = '../log/{}/{}.txt'.format(init_args.model_name+init_args.model_mode, log_file_name)
	if args.model_path == '':
		args.model_path = '../model/{}/{}.pt'.format(init_args.model_name+init_args.model_mode, log_file_name)

	utils.check_dir(args.log_file)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	logging.info(init_args)

	main()

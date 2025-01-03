异构图增强对比学习：基于ReChorus框架的复现实验
介绍
本项目为中山大学人工智能学院2022级机器学习课程的作业，主要工作为利用ReChorus框架对异构图增强对比学习（Heterogeneous Graph Contrastive Learning，HGCL）进行复现。

主要工作及运行方法
代码修改
在复现过程中，我们进行了以下代码修改：

增加新数据集文件夹及数据读取函数：位于 ReChorus/data/dataset_HGCL/ 目录下。
增加模型代码：位于 ReChorus/src/models/general/HGCL.py 。
增加数据读取类：位于 ReChorus/src/helpers/HGCLReader.py。
增加数据运行类：位于 ReChorus/src/helpers/HGCLRunner.py。
运行方法
要在ReChorus框架中运行HGCL，请执行以下命令：

cd ReChorus
python src/main.py --model_name HGCL --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset CiaoDVD
环境配置及数据处理
环境配置
创建并激活名为ReChorus的conda环境，并安装必要的依赖包：

conda create -n ReChorus python=3.10.4
conda activate ReChorus
编辑requirements.txt文件，删除torch==1.12.1和pickle（因为pickle是Python 3.10自带的），并将numpy版本从1.22.3调整为1.23.5以适配scipy库。然后安装依赖：

conda install --file requirements.txt
pip install torch==1.12.1
安装全局依赖原件

cd ReChorus
pip install -e .
数据处理
由于ReChorus框架中提供的数据集缺少描述用户之间信任关系的数据，我们使用了CiaoDVD和Epinions两个数据集，并基于BaseReader类重写了数据导入逻辑。

处理流程包括：

读取原始数据，生成训练集和测试集，同时生成信任矩阵和类别矩阵，最终保存所有数据到data.pkl。
读取data.pkl，计算用户之间的信任关系和项目类别的相似性，并将结果保存为UserDistance_mat.pkl和ItemDistance_mat.pkl。基于项目的类别信息生成项目距离矩阵，并将结果保存为ICI.pkl。
基于上述数据生成train.csv，test.csv和dev.csv。
运行结果及关键代码
为了验证模型的有效性，我们在CiaoDVD和Epinions数据集上进行了实验，并与ReChorus框架内的其他模型进行了对比分析。实验结果表明，HGCL在HR和NDCG评价指标上表现优异。

关键算法
GNN的前向传播
MLP的前向传播
HGCL的前向传播
创新：基于动态关系建模的改进思路
在推荐系统中，用户—项目交互、用户间社会关系以及项目间关联关系通常是动态变化的。为此，我们提出了一种基于动态关系建模的方法，利用时序异构图模型来捕捉随时间变化的关系，并通过递归公式更新节点嵌入。

动态关系建模算法
Compute Relation Matrix
Update Embedding
Dynamic Aggregate
以上算法协同工作，增强了推荐系统的动态适应能力。

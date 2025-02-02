# ReChorus-Master

## 项目简介
本项目为中山大学人工智能学院2022级机器学习课程的大作业。主要目标为利用 ReChorus 框架复现论文《Graph Convolution Network based Recommender Systems: Learning Guarantee and Item Mixture Powered Strategy》，验证 LightGCN 在推荐系统上的泛用性，并评估 IMix 方法的实用性。

## 环境配置
安装依赖：

pip install -r requirements.txt

## 代码更改
我们在src中添加了LightGCN_new.py 文件，其中添加了CustomLightGCN类，作为我们实验使用的LightGCN，并添加了带有IMIX的LightGCN，NGCF，带有IMIX的NGCF，该文件只用于运行这个四个类，效果与main相当。

我们还在src中添加了run.py 文件，该文件用于运行多种参数的命令。

## 运行方法
以下命令用于在 ReChorus 框架中运行 LightGCN 模型：

创建名为ReChorus_master的conda虚拟环境，且python==3.7

cd ReChorus_master

python src/LightGCN_new.py --model_name=LightGCN --n_layers=3 --metric=NDCG,RECALL --dataset=ML_1MTOPK --topk=20,40,60,80,100

### 运行示例
运行带有 IMix 的 LightGCN 模型：
python src/LightGCN_new.py --model_name=LightGCNWithIMix --n_layers=3 --metric=NDCG,RECALL --dataset=ML_1MTOPK --topk=20,40,60,80,100

## 主要功能
- **LightGCN 和 NGCF 的复现**
- **IMix 的实验集成**
- **命令行交互便捷运行**

## 结果
实验结果以 Top-K 推荐指标展示，支持 NDCG 和 Recall 的多层次评估。

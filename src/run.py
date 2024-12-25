import os

def t1():
    n_layers_list = [1, 2, 3, 4, 5]
    # weight_decay_list = [0.001, 0.01, 0.1, 1.0]
    weight_decay_list = [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0,1.3,1.6]
    for n_layers in n_layers_list:
        cmd = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --model_name=CustomLightGCN --n_layers={n_layers} --metric=NDCG,RECALL --dataset=gowalla --topk=20,40,60,80,100 --result_file=D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/results/result_layernumber.txt"
        os.system(cmd)
    for weight_decay in weight_decay_list:
        cmd = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --model_name=CustomLightGCN --l2={weight_decay} --metric=NDCG,RECALL --dataset=gowalla --topk=20,40,60,80,100 --result_file=D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/results/result_l2.txt"
        os.system(cmd)


# --emb_size=32 --batch_size=128 
def t2():
    normalizations=["Adjacency_matrix_with_self-loops","Row-normalized_adjacency_matrix","Symmetrically_normalized_adjacency_matrix"]
    for normalization in normalizations:
        cmd = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=gowalla --topk=20,40,60,80,100 --normalization={normalization} --result_file='D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/results/result_normalization.txt'"
        os.system(cmd)
def t3():
    activation=['ReLU','Tanh','Sigmoid','LeakyReLU',""]
    for activation in activation:
        cmd = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=gowalla --topk=20,40,60,80,100 --activation_function={activation} --result_file='D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/results/result_activation.txt'"
        os.system(cmd)
def t4():
    # activation=['ReLU','Tanh','Sigmoid','LeakyReLU',""]
    # for activation in activation:
        # cmd1 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=gowalla --topk=20  --result_file='D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/results/result_activation.txt'"
        cmd2 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=LightGCNWithIMix --metric=NDCG,RECALL --dataset=gowalla --topk=20  "
        cmd3 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=LightGCNWithIMix --metric=NDCG,RECALL --dataset=yelp2018 --topk=20  "
        cmd4 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=gowalla --topk=20  "
        cmd5 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --test_epoch=5 --early_stop=200 --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=yelp2018 --topk=20  "
        # os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)
        os.system(cmd5)
def t5():
    feature_sizes=[32,64,128]
    for feature_size in feature_sizes:
        cmd1 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --emb_size={feature_size}   --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=gowalla --topk=20"
        cmd2 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --emb_size={feature_size}   --model_name=LightGCNWithIMix --metric=NDCG,RECALL --dataset=gowalla --topk=20"
        cmd3 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --emb_size={feature_size}   --model_name=CustomLightGCN --metric=NDCG,RECALL --dataset=yelp_review --topk=20"
        cmd4 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --emb_size={feature_size}   --model_name=LightGCNWithIMix --metric=NDCG,RECALL --dataset=yelp_review --topk=20"
        if not feature_size == 32:
            os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)
def t6():
    hidden_layer_sizes=[8,16,32,64,128]
    for hidden_layer_size in hidden_layer_sizes:
        cmd1 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --hidden_layer_size={hidden_layer_size}   --model_name=NGCF --metric=NDCG,RECALL --dataset=gowalla --topk=20 "
        cmd2 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --hidden_layer_size={hidden_layer_size}   --model_name=NGCFWithIMix --metric=NDCG,RECALL --dataset=gowalla --topk=20 "
        cmd3 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --hidden_layer_size={hidden_layer_size}   --model_name=NGCF --metric=NDCG,RECALL --dataset=yelp_review --topk=20 "
        cmd4 = f"python D:/Users/Wind/Desktop/机器学习大作业/ReChorus-master/src/LightGCN_new.py --hidden_layer_size={hidden_layer_size}   --model_name=NGCFWithIMix --metric=NDCG,RECALL --dataset=yelp_review --topk=20 "

        os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)

# t2()
# t3()
t4()
# t5()
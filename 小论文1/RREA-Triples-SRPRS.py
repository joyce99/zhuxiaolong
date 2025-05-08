import warnings

from evaluate import evaluate

warnings.filterwarnings('ignore')

import os
import random
import keras
from tqdm import *
import numpy as np
from utils import *
from CSLS import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import NR_GraphAttention, Classfier, Classfier1

import torch
import torch.nn as nn
import torch.optim as optim
import gc

from scipy import spatial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

lang = 'zh'
train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,triples1,triples2,noisy1,noisy2,entityid1,entityid2,triples_pos1,triples_pos2,entity1,entity2,triplesinitial = load_data1('data/en_fr_15k_V1/',train_ratio=0.30)
triples=triples1+triples2
noisy=noisy1+noisy2
triplespos1=1*triples1;triplespos2=1*triples2;triplespos=triplespos1+triplespos2
triples_pos=triples_pos1+triples_pos2
postivenumber=0
number=len(triples_pos)
train_pair_initial=1*train_pair
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
batch_size = node_size


dict={};dictail={}


def updatedictsample(dict,dictail,adj_matrixup):
    for i in range(node_size):
        dict[i]=[]
        dictail[i]=[]

    for j in range(len(adj_matrixup)):
        dict[adj_matrixup[j][0]].append(adj_matrixup[j][1])
        dictail[adj_matrixup[j][1]].append(adj_matrixup[j][0])

updatedictsample(dict,dictail,adj_matrix)


def negativasample(triples):
    triples_sample=[]
    for i in range(len(triples)):
        j=triples[i]
        if j in entityid1:
            k=random.choice(entityid1)
            while k in dict[j]:
                k = random.choice(entityid1)
            triples_sample.append(k)
        else:
           k = random.choice(entityid2)
           while k in dict[j]:
                k = random.choice(entityid2)
           triples_sample.append(k)
    return triples_sample

def negativasampletail(triples):
    triples_sample=[]
    for i in range(len(triples)):
        j=triples[i]
        if j in entityid1:
            k=random.choice(entityid1)
            while k in dictail[j]:
                k = random.choice(entityid1)
            triples_sample.append(k)
        else:
           k = random.choice(entityid2)
           while k in dictail[j]:
                k = random.choice(entityid2)
           triples_sample.append(k)
    return triples_sample

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


def get_embedding():
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    return get_emb.predict_on_batch(inputs)

def get_embedding1(index_a,index_b,vec = None):
    if vec is None:
        inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]
        vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True)+1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True)+1e-5)
    return Lvec,Rvec

def test(wrank=None):
    vec = get_embedding()
    return get_hits(vec, dev_pair, wrank=wrank)


def CSLS_test(thread_number=16, csls=10, accurate=True):
    vec = get_embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    return None


def get_train_set(batch_size=batch_size):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set);
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set


def get_trgat(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3,
              lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    encoder = NR_GraphAttention(node_size, activation="relu",
                                rel_size=rel_size,
                                depth=depth,
                                attn_heads=n_attn_heads,
                                triple_size=triple_size,
                                attn_heads_reduction='average',
                                dropout_rate=dropout_rate)

    out_feature = Concatenate(-1)([encoder([ent_feature] + opt), encoder([rel_feature] + opt)])
    out_feature = Dropout(dropout_rate)(out_feature)

    alignment_input = Input(shape=(None, 4))
    find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
        [out_feature, alignment_input])

    def align_loss(tensor):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        def l1(ll, rr):
            return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)

        def l2(ll, rr):
            return K.sum(K.square(ll - rr), axis=-1, keepdims=True)

        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))
        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)

    loss = Lambda(align_loss)(find)

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model


# %%
model, get_emb = get_trgat(dropout_rate=0.30, node_size=node_size, rel_size=rel_size, n_attn_heads=1, depth=1, gamma=3,
                           node_hidden=100, rel_hidden=100, triple_size=triple_size)

model.summary(); initial_weights = model.get_weights()
evaluater = evaluate(dev_pair)
triples_1=[];triples_2=[]
noisy_set_1 =[];noisy_set_2=[];noisy_set_rel=[]

def updateposid():
    triples_1.clear()
    triples_2.clear()
    num=len(triplespos)
    for i in range(num):
        triples_1.append(triplespos[i][0])
        triples_2.append(triplespos[i][2])

updateposid()


def updatenoisy():
    noisy_set_1.clear()
    noisy_set_2.clear()
    noisy_set_rel.clear()
    if noisy!=[]:
        num = len(noisy)
        for i in range(num):
            noisy_set_1.append(noisy[i][0])
            noisy_set_rel.append(noisy[i][1])
            noisy_set_2.append(noisy[i][2])

updatenoisy()


rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

net = Classfier1(1200,600,300, 150,75);loss_function = nn.BCELoss();optimizer = optim.Adam(net.parameters(), lr=1e-3);

print("当前三元组的个数："+str(len(triples)))
epoch=300
for i in trange(epoch):
    train_set = get_train_set()
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    model.train_on_batch(inputs, np.zeros((1, 1)))
    if i % 300 == 299:
        CSLS_test()
        vec=get_embedding()
    vec=get_embedding()
    if i==epoch-1:
        net = Classfier1(800,400,200,100,50);loss_function = nn.BCELoss();optimizer = optim.Adam(net.parameters(), lr=1e-3);
        net.to(device);loss_function.to(device);
        print("鉴别开始。。。")
        Lvecpos = np.array([vec[e] for e in triples_1]);Rvecpos = np.array([vec[e] for e in triples_2]);
        lenclassify0 = Lvecpos.shape[0];
        martix = np.hstack([Lvecpos, Rvecpos]);martix = torch.tensor(martix,device=device)
        y = torch.ones((lenclassify0, 1),device=device);y1 = torch.zeros((lenclassify0, 1),device=device);

        triples_sample = negativasample(triples_1);triples_sample2 = negativasample(triples_1);triples_sample3 = negativasample(triples_1);
        SRvec1 = np.array([vec[e] for e in triples_sample]);SRvec2 = np.array([vec[e] for e in triples_sample2]);SRvec3 = np.array([vec[e] for e in triples_sample3])
        martix_ne = np.hstack([Lvecpos, SRvec1]);martix_ne = torch.tensor(martix_ne,device=device)
        martix_ne2 = np.hstack([Lvecpos, SRvec2]);martix_ne2 = torch.tensor(martix_ne2,device=device)
        martix_ne3 = np.hstack([Lvecpos, SRvec3]);martix_ne3 = torch.tensor(martix_ne3,device=device)

        triples_samplet = negativasampletail(triples_2);triples_samplet2 = negativasampletail(triples_2);triples_samplet3 = negativasampletail(triples_2);
        SRvect1 = np.array([vec[e] for e in triples_samplet]);SRvect2 = np.array([vec[e] for e in triples_samplet2]);SRvect3 = np.array([vec[e] for e in triples_samplet3])
        martix_net = np.hstack([SRvect1,Rvecpos]);martix_net = torch.tensor(martix_net,device=device)
        martix_net2 = np.hstack([SRvect2,Rvecpos]);martix_net2 = torch.tensor(martix_net2,device=device)
        martix_net3 = np.hstack([SRvect3,Rvecpos]);martix_net3 = torch.tensor(martix_net3,device=device)
        for i in range(400):
            net.train()
            pred = net(martix);pred1 = net(martix_ne);pred2 = net(martix_ne2);pred3 = net(martix_ne3);
            predt1 = net(martix_net);predt2 = net(martix_net2);predt3 = net(martix_net3);
            loss = loss_function(pred, y);loss1 = loss_function(pred1, y1);loss2 = loss_function(pred2, y1);loss3 = loss_function(pred3, y1);
            losst1 = loss_function(predt1, y1);losst2 = loss_function(predt2, y1);losst3 = loss_function(predt3, y1);
            loss_all = loss + loss1 + loss2 + loss3 + losst1+losst2+losst3
            optimizer.zero_grad();loss_all.backward();optimizer.step()
        print(loss_all)
        Lvecnos = np.array([vec[e] for e in noisy_set_1]);Rvecnos = np.array([vec[e] for e in noisy_set_2])
        net.eval()
        len2 = Lvecnos.shape[0];len3 = Lvecnos.shape[1]
        martixnos = np.hstack([Lvecnos, Rvecnos]);martixnos = torch.tensor(martixnos,device=device)
        ynos = net(martixnos);ynos = ynos.detach().cpu().numpy()
        dictsorthalf = {};
        lensorthalf = len(ynos)
        for i in range(lensorthalf):
            dictsorthalf[i] = ynos[i]
        sortresult = sorted(dictsorthalf.items(), key=lambda x: x[1],reverse=True)
        lensorthalf1 = int(lensorthalf * 0.85);
        tempsorthalf = 0
        x=0;y=0;
        for name in sortresult:
            if tempsorthalf < lensorthalf1:
                x+=1
                if noisy[name[0]] in noisy1:
                    triples1.append(noisy[name[0]])
                    noisy1.remove(noisy[name[0]])
                else:
                    triples2.append(noisy[name[0]])
                    noisy2.remove(noisy[name[0]])
                if noisy[name[0]] in triples_pos:
                    y+=1
            tempsorthalf += 1
        print(x,y);


        triples = triples1 + triples2;noisy = noisy1 + noisy2;
        updatenoisy();

        result = '{:.2%}'.format(y/x)
        print('这一轮产生了:'+str(x)+"个三元组。。。")
        print('这一轮产生的三元组中，正确的个数为:' + str(y) + "个，因此正确的比例为："+str(result))
        print("鉴别结束。。。")
        adj_matrix, r_index, r_val, rel_matrix, ent_matrix,adj_features,rel_features = update('data/en_fr_15k_V1/', triples1, triples2)
        triple_size = len(adj_matrix)                       # 考虑三元组及其他的逆三元组合在一起的规模
        del model;gc.collect()
        model, get_emb = get_trgat(dropout_rate=0.30, node_size=node_size, rel_size=rel_size, n_attn_heads=1, depth=1,
                                   gamma=3,
                                   node_hidden=100, rel_hidden=100, triple_size=triple_size)
        print("当前三元组的个数：" + str(len(triples)))

rel_type,r_num= rfunc(triples, node_size)
kg=getkgdict(triples,r_num)
Lset=dev_pair[:, 0];Rset=dev_pair[:, 1]


del martix;del martix_ne;del martix_ne2;del martix_ne3;del martixnos
del triples_sample;del triples_sample2;del triples_sample3
del triples_samplet;del triples_samplet2;del triples_samplet3;
del y;del y1;
del martix_net;del martix_net2;del martix_net3;
del triples;del triples1;del triples2;del triples_pos;del triples_pos1;del triples_pos2;del triples_1;del triples_2;
del noisy;del noisy1;del noisy2;del noisy_set_1;del noisy_set_2;del noisy_set_rel;del dict;del dictail;del dictsorthalf;del sortresult
del Lvecnos;del Rvecnos;del Lvecpos;del Rvecpos;del SRvec1;del SRvec2;del SRvec3;del SRvect1;del SRvect2;del SRvect3;
gc.collect()
torch.cuda.empty_cache()

# epoch = 300
# for turn in range(5):
#     print("iteration %d start." % turn)
#     for i in trange(epoch):
#         train_set = get_train_set()
#         inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
#         inputs = [np.expand_dims(item, axis=0) for item in inputs]
#         model.train_on_batch(inputs, np.zeros((1, 1)))
#         if i % 300 == 299:
#             CSLS_test()
#
#     new_pair = []
#     vec = get_embedding()
#     Lvec = np.array([vec[e] for e in rest_set_1])
#     Rvec = np.array([vec[e] for e in rest_set_2])
#     Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
#     Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
#     A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
#     B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
#     A = sorted(list(A));
#     B = sorted(list(B))
#     for a, b in A:
#         if B[b][1] == a:
#             new_pair.append([rest_set_1[a], rest_set_2[b]])
#     print("generate new semi-pairs: %d." % len(new_pair))
#
#     train_pair = np.concatenate([train_pair, np.array(new_pair)], axis=0)
#     for e1, e2 in new_pair:
#         if e1 in rest_set_1:
#             rest_set_1.remove(e1)
#
#     for e1, e2 in new_pair:
#         if e2 in rest_set_2:
#             rest_set_2.remove(e2)
#
# del model;gc.collect()
# model, get_emb = get_trgat(dropout_rate=0.30, node_size=node_size, rel_size=rel_size, n_attn_heads=1, depth=2,gamma=3,
#                            node_hidden=100, rel_hidden=100, triple_size=triple_size)

print("模型处理噪音完成，加入监督ot")
threashoud=0.6
threashoud1=0.5
rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)
train_pair=1*train_pair_initial
epoch = 300
for turn in range(7):
    print("iteration %d start." % turn)
    for i in trange(epoch):
        train_set = get_train_set()
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        model.train_on_batch(inputs, np.zeros((1, 1)))
        if i % 300 == 299:
            Lvec, Rvec = get_embedding1(dev_pair[:, 0], dev_pair[:, 1])
            evaluater.test(Lvec, Rvec)
            CSLS_test()

    Lvec, Rvec = get_embedding1(dev_pair[:, 0], dev_pair[:, 1])
    sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    # sim = getdistance(sim, Lset, Rset, rel_type, train_pair, kg)
    dev_pair = torch.tensor(dev_pair);
    sim = torch.tensor(sim)
    tempair = get_hits_sinkhorn(test_pair=dev_pair, S=sim)
    dev_pair = dev_pair.detach().numpy()
    del sim;gc.collect()

    new_pairot = []
    new_paircsls = []
    new_all = []

    temp = []
    for i in tempair:
        temp.append(list(i))
    vec = get_embedding()
    for i in temp:
        L = vec[i[0]];
        R = vec[i[1]]
        sim = 1 - spatial.distance.cosine(L, R)
        if sim > threashoud:
            new_pairot.append(i)
    threashoud -= 0.1

    vec = get_embedding()
    Lvec = np.array([vec[e] for e in rest_set_1])
    Rvec = np.array([vec[e] for e in rest_set_2])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
    B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
    A = sorted(list(A));
    B = sorted(list(B))
    for a, b in A:
        Le = vec[rest_set_1[b]];Re = vec[rest_set_2[a]]
        sim1 = 1 - spatial.distance.cosine(Le, Re)
        if B[b][1] == a:
            new_paircsls.append([rest_set_1[a], rest_set_2[b]])
    threashoud1 -= 0.03

    new_all = [x for x in new_pairot if x in new_paircsls]

    print("generate new semi-pairs: %d." % len(new_all))

    templen = 0
    for i in new_pairot:
        if (dev_pair == np.array(i)).all(1).any():
            templen += 1
    print(templen, len(new_pairot))

    templen = 0
    for i in new_paircsls:
        if (dev_pair == np.array(i)).all(1).any():
            templen += 1
    print(templen, len(new_paircsls))

    templen = 0
    for i in new_all:
        if (dev_pair == np.array(i)).all(1).any():
            templen += 1
    print(templen, len(new_all))

    train_pair = np.concatenate([train_pair, np.array(new_all)], axis=0)
    for e1, e2 in new_all:
        if e1 in rest_set_1:
            rest_set_1.remove(e1)

    for e1, e2 in new_all:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)

templen=0
for i in train_pair:
    if (dev_pair == i).all(1).any():
        templen+=1
print(templen,len(train_pair))
vec=get_embedding()
testlightea(dev_pair, vec, 500)
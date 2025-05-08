import gc
import warnings

import torch
from scipy import spatial
from torch import nn, optim

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
from layer import NR_GraphAttention, Classfier
from keras.losses import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lang = 'zh'
train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,alignment_pair,train_pair1,train_pair2,entityid1,entityid2,triples= load_dataNoisy('data/zh_en/',train_ratio=0.30)
entityidall=entityid1+entityid2
entityidall=list(set(entityidall))
entityidall=np.array(entityidall).reshape((len(entityid1)+len(entityid2),1))

list_l=[e1 for e1,e2 in train_pair1]
list_r=[e2 for e1,e2 in train_pair1]
list1_l=[e1 for e1,e2 in train_pair2]
list1_r=[e2 for e1,e2 in train_pair2]

adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
batch_size = node_size


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

def get_embedding1():
    inputs= [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    return get_emb1.predict_on_batch(inputs)


def test(wrank=None):
    vec = get_embedding()
    return get_hits(vec, dev_pair, wrank=wrank)


def CSLS_test(flag=True,thread_number=16, csls=10, accurate=True):
    vec = get_embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    if flag:
        vec = get_embedding1()
        Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
        Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    return None


def get_train_setTeacher(train_batch,batch_size=batch_size):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_batch, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set);
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set

def get_train_setStudent(new_pair,batch_size=batch_size):
    negative_ratio = batch_size // len(new_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(new_pair, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set);
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set


def get_trgatTeacher(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3,
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

    alignment_input1 = Input(shape=(None, 4))
    find1 = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
        [out_feature, alignment_input1])

    embedding=Input(shape=(None,600))
    embedding1 = Input(shape=(None, 600))

    def align_loss(inputs):
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

        tensor = inputs[0]
        tensor1 = inputs[1]
        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))

        l, r, fl, fr = [tensor1[:, 0, :], tensor1[:, 1, :], tensor1[:, 2, :], tensor1[:, 3, :]]
        loss1 = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))

        tensor2=inputs[2]
        tensor3=inputs[3]
        loss_dis_1 = mean_squared_error(tensor2, tensor3)
        total_loss_dis_1 = tf.reduce_sum(loss_dis_1)

        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)+tf.reduce_sum(loss1, keep_dims=True) / (batch_size)+total_loss_dis_1

    loss = Lambda(align_loss)([find,find1,embedding,embedding1])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input,alignment_input1,embedding,embedding1], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model

def get_trgatStudent(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3,
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

model,get_emb = get_trgatTeacher(dropout_rate=0.30,node_size=node_size,rel_size=rel_size,n_attn_heads = 1,depth=2,gamma =3,node_hidden=100,rel_hidden = 100,triple_size = triple_size)
model.summary();

model1,get_emb1 = get_trgatTeacher(dropout_rate=0.30,node_size=node_size,rel_size=rel_size,n_attn_heads = 1,depth=2,gamma =3,node_hidden=100,rel_hidden = 100,triple_size = triple_size)
model1.summary();

rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

epoch = 1500

noisy_1=[e1 for e1,e2 in train_pair]
noisy_2=[e2 for e1,e2 in train_pair]

true=[]
for i in trange(epoch):
    train_set = get_train_setTeacher(train_pair1)
    train_set1 = get_train_setTeacher(train_pair2)

    if i>299:
        train_new=get_train_setTeacher(new_pair1)
        train_new1=get_train_setTeacher(new_pair)
    else:
        train_new=train_set
        train_new1=train_set1
    vec=get_embedding();vec1=get_embedding1();
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set,train_new,vec,vec1]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    inputs1 = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set1,train_new1,vec1,vec]
    inputs1 = [np.expand_dims(item, axis=0) for item in inputs1]

    model.train_on_batch(inputs, np.zeros((1, 1)))
    model1.train_on_batch(inputs1, np.zeros((1, 1)))
    if i % 300 == 299:
        CSLS_test()

        new_pair = []
        new_pair1=[]
        correct=[]
        correct1 = []

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
            if B[b][1] == a:
                new_pair.append([rest_set_1[a], rest_set_2[b]])
        print("generate new semi-pairs: %d." % len(new_pair))

        Lvec = np.array([vec[e] for e in noisy_1])
        Rvec = np.array([vec[e] for e in noisy_2])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A));
        B = sorted(list(B))
        for a, b in A:
            if B[b][1] == a:
                correct.append((noisy_1[a], noisy_2[b]))
        print("generate new semi-pairs: %d." % len(correct))


        vec = get_embedding1()
        Lvec = np.array([vec[e] for e in rest_set_1])
        Rvec = np.array([vec[e] for e in rest_set_2])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A));
        B = sorted(list(B))
        for a, b in A:
            if B[b][1] == a:
                new_pair1.append([rest_set_1[a], rest_set_2[b]])
        print("generate new semi-pairs: %d." % len(new_pair1))

        Lvec = np.array([vec[e] for e in noisy_1])
        Rvec = np.array([vec[e] for e in noisy_2])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A));
        B = sorted(list(B))
        for a, b in A:
            if B[b][1] == a:
                correct1.append((noisy_1[a], noisy_2[b]))
        print("generate new semi-pairs: %d." % len(correct1))

        correctall=[x for x in correct if x in correct1]
        y=0
        for pair in correctall:
            true.append(pair)
            if pair in alignment_pair:
                y+=1
        print(y,len(correctall))



        for e1, e2 in correctall:
            if e1 in noisy_1:
                noisy_1.remove(e1)

        for e1, e2 in correctall:
            if e2 in noisy_2:
                noisy_2.remove(e2)

        new_pair = np.array(new_pair)
        new_pair1=np.array(new_pair1)




print('最终检验')
y=0
for i in true:
    if i in alignment_pair:
        y+=1

print(y,len(true))


epoch=1200
print('最终训练')
del model;del get_emb;del model1;del get_emb1
gc.collect()
model,get_emb = get_trgatStudent(dropout_rate=0.30,node_size=node_size,rel_size=rel_size,n_attn_heads = 1,depth=2,gamma =3,node_hidden=100,rel_hidden = 100,triple_size = triple_size)
model.summary();

for turn in range(5):
    print("iteration %d start." % turn)
    for i in trange(epoch):
        train_set = get_train_setStudent(true)
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        model.train_on_batch(inputs, np.zeros((1, 1)))
        if i % 300 == 299:
            CSLS_test(flag=False)

    new_pair = []
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
        if B[b][1] == a:
            new_pair.append([rest_set_1[a], rest_set_2[b]])
    print("generate new semi-pairs: %d." % len(new_pair))

    true = np.concatenate([true, np.array(new_pair)], axis=0)
    for e1, e2 in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1)

    for e1, e2 in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
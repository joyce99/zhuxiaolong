import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing

import torch
from sinkhorn_loss_wasserstein import *

def get_target(triples,file_paths):
    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_paths + "/ent_ids_" + str(i) for i in range(1,3)])
    
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return r_hs, r_ts, ids

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        # head,tail,r = [int(item) for item in line.split()]
        head, r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel):
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        print(ent_size,rel_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        for h,r,t in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
            adj_features[h,t] = 1; adj_features[t,h] = 1;
            radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
            rel_out[h][r] += 1; rel_in[t][r] += 1
            
        count = -1
        s = set()
        d = {}
        r_index,r_val = [],[]
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix,r_index,r_val,adj_features,rel_features      

def load_entityid(file_name):
    entity = []
    for line in open(file_name, 'r',encoding='utf-8'):
        e1, e2 = line.split()
        entity.append(int(e1))
    return entity

def load_data(lang,train_ratio = 0.3):
    entity1,rel1,triples1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2')
    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.shuffle(alignment_pair)
    if "en" in lang:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        np.random.shuffle(alignment_pair)
        train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair) * train_ratio)], alignment_pair[int(len(
            alignment_pair) * train_ratio):]
    else:
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features

def load_datainitial(lang,train_ratio = 0.3):
    entity1,rel1,triples1 = load_triples(lang + 'triples_1postive')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2postive')
    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.shuffle(alignment_pair)
    train_pair, dev_pair = load_alignment_pair(lang + 'new_train.txt'), load_alignment_pair(lang + 'test.txt')
    np.random.shuffle(train_pair)
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features


def load_dataNoisy(lang,train_ratio = 0.3):
    entity1,rel1,triples1 = load_triples(lang + 'triples_1postive')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2postive')
    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.shuffle(alignment_pair)
    train_pair, dev_pair = load_alignment_pair(lang+'new_train.txt'),load_alignment_pair(lang+'test.txt')
    np.random.shuffle(train_pair)
    train_pair1=train_pair[0:int(len(train_pair)/2)]
    train_pair2=[i for i in train_pair if i not in train_pair1]

    entityid1 = load_entityid(lang + 'ent_ids_1')
    entityid2 = load_entityid(lang + 'ent_ids_2')
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features,alignment_pair,np.array(train_pair1),np.array(train_pair2),entityid1,entityid2,triples1+triples2

def load_dataOT(lang,train_ratio = 0.3,flag=True):
    entity1,rel1,triples1 = load_triples(lang + 'triples_1postive')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2postive')
    if flag:
        train_pair, dev_pair = load_alignment_pair(lang+'new_train.txt'),load_alignment_pair(lang+'test.txt')
        np.random.shuffle(train_pair)
    else:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        alignment_pair=list(reversed(alignment_pair))
        np.random.shuffle(alignment_pair)
        train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair) * train_ratio)], alignment_pair[int(len(
            alignment_pair) * train_ratio):]
        # train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        # dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features,triples1+triples2


def load_dataLR(lang,train_ratio = 0.3,flag=True):
    entity1,rel1,triples1 = load_triples(lang + 'triples_zh.txt')
    entity2,rel2,triples2 = load_triples(lang + 'triples_vi.txt')
    if flag:
        train_pair, dev_pair = load_alignment_pair(lang+'new_train.txt'),load_alignment_pair(lang+'test.txt')
        np.random.shuffle(train_pair)
    else:
        alignment_pair = load_alignment_pair(lang + 'entity_seeds.txt')
        alignment_pair=list(reversed(alignment_pair))
        np.random.shuffle(alignment_pair)
        train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair) * train_ratio)], alignment_pair[int(len(
            alignment_pair) * train_ratio):]
        # train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        # dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features,triples1+triples2


def get_hits(vec, test_pair, wrank = None, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i,sim[i,j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank,-1),np.expand_dims(wrank,-1)],-1),axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))  
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))

def get_hits_sinkhorn(test_pair, S, top_k=(1, 10),flag=True):
    dev_s = test_pair[:, 0].to('cpu').numpy().tolist()
    dev_t = test_pair[:, 1].to('cpu').numpy().tolist()

    mu, nu = torch.ones(len(dev_s)), torch.ones(len(dev_t))
    sim = sinkhorn(mu, nu, S.to('cpu'), 0.05, stopThr=1e-3)
    if sim is None:
        S = torch.sqrt(S)
        sim = sinkhorn(mu, nu, S.to('cpu'), 0.05, stopThr=1e-3)
    if flag==False:
        return sim
    sim = -sim

    credible_pairs_s2t, credible_pairs_t2s = set(), set()

    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(len(dev_s)):
        rank = sim[i, :].argsort()
        credible_pairs_s2t.add((dev_s[i], dev_t[rank[0].numpy().tolist()]))
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    MRR_rl = 0
    for i in range(len(dev_t)):
        rank = sim[:, i].argsort()
        credible_pairs_t2s.add((dev_s[rank[0].numpy().tolist()], dev_t[i]))
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('Left:', end=" ")
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100), end='\t')
    print('MRR: %.3f' % (MRR_lr / len(dev_s)))
    print('Right:', end=" ")
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100), end='\t')
    print('MRR: %.3f' % (MRR_rl / len(dev_t)))
    # intersection
    final_credible_pairs = credible_pairs_s2t.intersection(credible_pairs_t2s)
    return final_credible_pairs

def rfunc(KG, e):
    head = {}
    cnt = {}
    rel_type = {}
    cnt_r = {}
    for tri in KG:
        r_e = str(tri[1]) + ' ' + str(tri[2])
        if r_e not in cnt:
            cnt[r_e] = 1
            head[r_e] = set([tri[0]])
        else:
            cnt[r_e] += 1
            head[r_e].add(tri[0])

        if tri[1] not in cnt_r:
            cnt_r[tri[1]] = 1

    r_num = len(cnt_r)

    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value

    del cnt
    del head
    del cnt_r
    cnt = {}
    head = {}

    for tri in KG:
        r_e_new = str(tri[1] + r_num) + ' ' + str(tri[0])
        if r_e_new not in cnt:
            cnt[r_e_new] = 1
            head[r_e_new] = set([tri[2]])
        else:
            cnt[r_e_new] += 1
            head[r_e_new].add(tri[2])

    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value

    return  rel_type,r_num

def getkgdict(M0,r_num):
    kg = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[0] not in kg:
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()

        kg[tri[0]].add((tri[1], tri[2]))
        kg[tri[2]].add((tri[1] + r_num, tri[0]))

    return kg

def getdistance(sim_e,L,R,rel_type,ref,kg):
    alignL=ref[:, 0];alignR=ref[:, 1]
    for i in range(len(L)):
        rank = sim_e[i, :].argsort()[:100]
        if L[i] in kg:
            templ=kg[L[i]]
            templ = list(templ)
            second_valuesl = np.array([t[1] for t in templ])
            intersection = np.intersect1d(second_valuesl, alignL)
        if intersection.size>0:
            for j in rank:
                if R[j] in kg and L[i] in kg:
                    match_num = 0
                    for n_1 in kg[L[i]]:
                        if n_1[1] in alignL:
                            temp = kg[R[j]]
                            temp = list(temp)
                            second_values = [t[1] for t in temp]
                            index = np.where(alignL == n_1[1])[0]
                            align=int(alignR[index])
                            if align in second_values:
                                second_values=np.array(second_values)
                                index1 = np.where(second_values == align)[0]
                                for k in index1:
                                    k=int(k)
                                    w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(temp[k][0]) + ' ' + str(temp[k][1])]
                                    match_num += w

                    sim_e[i, j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]))
                    # sim_e[i, j] = -10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]))
    return sim_e
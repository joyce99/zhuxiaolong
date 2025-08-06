import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

# import ot

big = 1e20
huge = 1e30
small = 1e-7


def kl_div(x, y):
    """
    KL divergence of two tensor
    :param x:
    :param y:
    :return:
    """
    # y = y.reshape(x.shape)
    div = torch.div(x, y + small)  # avoid singularity
    kl = torch.mul(y, div * torch.log(div + small) - div + 1)
    return kl


def myclamp(x):
    return torch.clamp(x, 0, huge)


def forward_relax_sinkhorn_iteration(C, mu, nu, lambdda, epsilon, numIterMax=100, tol=1e-6, debug=False):
    """
    Generalized Sinkhorn iteration for balanced Optimal Transport
    Parameters:
        C: [*, I, J] cost matrix, up to 3 dimension, the first for batch size, used in grouping or barycenter
        mu: [*, I, 1] source margin distribution
        nu: [*, 1, J] target margin distribution
        lambdda: the lambda coefficient for two KL relax term
        epsilon: entropy regulizer
        numIterMax: number of iteration
    Return:
        transport: the primal distance
        margin1: the KL divergence of the first marginal and source distribution
        margin2: the KL divergence of the second marginal and target distribution
        K: the transport plan
    """
    *_, I, J = C.shape
    _, I1, _ = mu.shape
    *_, J1 = nu.shape

    if debug:
        assert I == I1
        assert J == J1
        assert len(C.shape) == len(mu.shape)
        assert len(C.shape) == len(nu.shape)

    def K_calc(_u, _v):
        _K = myclamp(torch.exp((_u + _v - C) / epsilon))
        if debug:
            assert not torch.isnan(_K).any()
        return _K

    pow_coef = lambdda / (lambdda + epsilon)

    u = torch.zeros_like(mu)  # Kantorovich potential for source distribution

    b = torch.ones_like(nu)  # partial update of Kantorovich potential
    v = torch.zeros_like(nu)  # Kantorovich potential for target distribution
    K = K_calc(u, v)
    transport = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
    for ii in range(numIterMax):
        s = torch.sum(torch.mul(K, b), -1, keepdim=True)
        a = myclamp(torch.div(mu, s) ** pow_coef)

        s = torch.sum(torch.mul(K, a), -2, keepdim=True)
        b = myclamp(torch.div(nu, s) ** pow_coef)

        if ii % 10 == 0 or torch.max(a) > big or torch.max(b) > big or ii == numIterMax - 1:
            u += epsilon * torch.log(a)
            v += epsilon * torch.log(b)
            K = K_calc(u, v)
            b = torch.ones_like(nu)

            transport_new = torch.sum(torch.sum(torch.mul(K, C), -1), -1).squeeze()
            if abs(transport_new - transport) / abs(transport) < tol:
                # print('meet tol, jump out of sinkhorn iteration')
                break
            else:
                transport = transport_new

    kl1 = kl_div(torch.sum(K, -1, keepdim=True), mu)
    margin1 = torch.sum(kl1, -2).squeeze()
    kl2 = kl_div(torch.sum(K, -2, keepdim=True), nu)
    margin2 = torch.sum(kl2, -1).squeeze()

    return transport, margin1, margin2, K


def sinkhorn(a, b, M, reg, numItermax = 1000, stopThr = 1e-9, verbose=False, sqrt=False):
    """
    Solve the entropic regularization balanced optimal transport problem 

    Parameters:
    param: a(tensor (I, )) sample weights for source measure
    param: b(tensor (J, )) sample weights for target measure
    param: M(tensor (I, J)) distance matrix between source and target measure
    param: reg(float64) regularization factor > 0
    param: numItermax(int) max number of iterations
    param: stopThr(float64) stop threshol
    param: verbose(bool) print information along iterations

    Return:
    P(tensor (I, J)) the final transport plan
    loss(float) the wasserstein distance between source and target measure
    """
    import time
    assert a.device == b.device and b.device == M.device, "a, b, M must be on the same device"

    device = a.device
    a, b, M = a.type(torch.DoubleTensor).to(device), b.type(torch.DoubleTensor).to(device), M.type(torch.DoubleTensor).to(device)

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.DoubleTensor) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.DoubleTensor) / M.shape[1]
    
    I, J = len(a), len(b)
    assert I == M.shape[0] and J == M.shape[1], "the dimension of weights and distance matrix don't match"

    # init 
    u = torch.ones((I, 1), device=device, dtype=a.dtype) / I
    v = torch.ones((J, 1), device=device, dtype=b.dtype) / J
    # K = torch.exp(-M / reg).to(device)
    K = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.div(M, -reg, out=K)
    torch.exp(K, out=K)

    tmp2 = torch.empty(b.shape, dtype=b.dtype, device=device)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt, err = 0, 1 
    pos = time.time()
    while (err > stopThr and cpt < numItermax):
        uprev, vprev = u, v

        KtranposeU = torch.mm(K.t(), u)
        v = b.reshape(-1, 1) / KtranposeU
        u = 1. / Kp.mm(v)

        if (torch.any(KtranposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            print("Warning: numerical errors at iteration ", cpt)
            u, v = uprev, vprev
            if cpt < numItermax * 0.8 and sqrt is False:
                return None
            else:
                break

        if cpt % 10 == 0:
            tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
            err = torch.norm(tmp2 - b)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:5s}'.format('It.','Err') + '\n' + '-' * 19)
                print("{:5s}|{:5s}".format(cpt, err))
        
        cpt += 1
    print("ours cpt: {}, err: {}".format(cpt, err))
    print("ours time: {}".format(time.time() - pos))
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P


def Prior_sinkhorn(a, b, M, T, reg1, reg2, numItermax = 1000, stopThr = 1e-9, verbose = False):
    """
    Solve the entropic regularization balanced optimal transport problem 

    Parameters:
    param: a(tensor (I, )) sample weights for source measure
    param: b(tensor (J, )) sample weights for target measure
    param: M(tensor (I, J)) distance matrix between source and target measure
    param: T(tensor (I, J)) the prior transport plan of the problem
    param: reg1(float64) regularization factor > 0 for the enrtropic term
    param: reg2(float64) regularization factor > 0 for the KL divergence term between P and T
    param: numItermax(int) max number of iterations
    param: stopThr(float64) stop threshol
    param: verbose(bool) print information along iterations

    Return:
    P(tensor (I, J)) the final transport plan
    loss(float) the wasserstein distance between source and target measure
    """
    import time
    assert a.device == b.device and b.device == M.device, "a, b, M must be on the same device"

    device = a.device
    a, b, M = a.type(torch.DoubleTensor).to(device), b.type(torch.DoubleTensor).to(device), M.type(torch.DoubleTensor).to(device)

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.DoubleTensor) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.DoubleTensor) / M.shape[1]
    
    I, J = len(a), len(b)
    assert I == M.shape[0] and J == M.shape[1], "the dimension of weights and distance matrix don't match"

    # init 
    u = torch.ones((I, 1), device = device, dtype=a.dtype) / I
    v = torch.ones((J, 1), device = device, dtype=b.dtype) / J

    # compute K 
    K1 = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.div(M, -(reg1 + reg2), out=K1)
    K2 = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.log(T, out= K2)
    K2 = K2 * reg2
    torch.div(K2, (reg1 + reg2), out=K2)
    K = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.exp(K1 + K2, out=K)

    tmp2 = torch.empty(b.shape, dtype=b.dtype, device=device)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt, err = 0, 1 
    pos = time.time()
    while (err > stopThr and cpt < numItermax):
        uprev, vprev = u, v

        KtranposeU = torch.mm(K.t(), u)
        v = b.reshape(-1, 1) / KtranposeU
        u = 1. / Kp.mm(v)

        if (torch.any(KtranposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            print("Warning: numerical errors at iteration ", cpt)
            u, v = uprev, vprev
            break
        
        if cpt % 10 == 0:
            tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
            err = torch.norm(tmp2 - b)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:5s}'.format('It.','Err') + '\n' + '-' * 19)
                print("{:5s}|{:5s}".format(cpt, err))
        
        cpt += 1
    print("ours cpt: {}, err: {}".format(cpt, err))
    print("ours time: {}".format(time.time() - pos))
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, torch.sum(P * M)


def VGW(mu, nu, X, Y, reg, numItermax = 1000, stopThr = 1e-9, verbose = False):
    assert X.shape[0] == Y.shape[0]
    C1_square = (X * X).sum(axis=1, keepdim=True) - 2 * X.mm(X.t()) + (X * X).sum(axis=1, keepdim=True).reshape(1, -1)
    C2_square = (Y * Y).sum(axis=1, keepdim=True) - 2 * Y.mm(Y.t()) + (Y * Y).sum(axis=1, keepdim=True).reshape(1, -1)
    Xmean = (mu.reshape(1, -1).mm(X)).reshape(1, -1)
    Ymean = (nu.reshape(1, -1).mm(Y)).reshape(-1, 1)
    E = (C1_square * mu.reshape(-1, 1)).sum(axis = 0).reshape(-1, 1) + (C2_square * nu.reshape(-1, 1)).sum(axis = 0).reshape(1, -1) + 2 * X.mm(Ymean) + 2 * Xmean.mm(Y.t())
    Mt = E - 4 * torch.eye(E.shape[0], device=X.device)
    P, _ = sinkhorn(mu, nu, Mt, reg, numItermax, stopThr, verbose)
    return P, _ 


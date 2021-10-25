import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial, bernoulli
import time
from itertools import permutations
import igraph



#######################################
##### functions for visualization #####
#######################################



def visual_network(adj,label = None,color = None):
    # get the row, col indices of the non-zero elements in the adjacency matrix
    conn_indices = np.where(adj)

    # get the edge weights corresponding to these indices
    weights = adj[conn_indices]

    # a sequence of (i, j) tuples, each corresponding to an edge from i -> j
    edges = zip(*conn_indices)

    # initialize the graph from the edge sequence
    G = igraph.Graph(edges=edges, directed=False)

    # assign node names and weights to be attributes of the vertices and edges respectively
    if label is None:
        G.vs['label'] = np.arange(adj.shape[0])
    else:
        G.vs['label'] = label
    col_list = ['red','green','blue']
    G.vs['color'] = [col_list[i] for i in label]
    G.es['weight'] = weights

    # Assign the weights to the 'width' attribute of the edges. This
    # means that igraph.plot will set the line thicknesses according to the edge
    # weights
    G.es['width'] = weights

    # plot the graph
    visual_style = {}
    visual_style["layout"] = G.layout_kamada_kawai()
    visual_style["bbox"] = (300, 300)
    visual_style["margin"] = 10
    visual_style["labels"] = True

    f = igraph.plot(G, **visual_style)
    # igraph.plot(G, layout="rt", labels=True, margin=10)
    return f



def visual_history(history,names = ['loglikelihood','l1 difference']):
    '''
    show the given 2 types of learning history
    '''
    fig = plt.figure(2,figsize = (9,5))
    h1, h2 = history
    T = len(h1)
    
    ax = fig.add_subplot(1,2,1)
    plt.plot(np.arange(T),h1)
    plt.title(names[0])

    ax = fig.add_subplot(1,2,2)
    plt.plot(np.arange(T),h2)
    plt.title(names[1])

    plt.show()


    
def visual_history_3(history,names = ['loglikelihood','lower bound loglike','l1 difference']):
    '''
    show the given 3 types of learning history
    '''
    fig = plt.figure(3,figsize = (15,5))
    h1, h2, h3 = history
    T = len(h1)
    
    ax = fig.add_subplot(1,3,1)
    plt.plot(np.arange(T),h1)
    plt.title(names[0])
    
    ax = fig.add_subplot(1,3,2)
    plt.plot(np.arange(T),h2)
    plt.title(names[1])
    
    ax = fig.add_subplot(1,3,3)
    plt.plot(np.arange(T),h3)
    plt.title(names[2])
    
    plt.show()



def visual_membership(Pi,label = None,node_id = False):
    '''
    Visualize the membership of items
    --------------------------------------------------
    Pi (n x K array): membership matrix to be visualized, now only works for K=3
    label (n x . array): labels of classes, if None then use 1,...,K as labels. None by default
    node_id (n x . array): id of nodes, if None then use 1,...,n as id. None by default
    '''
    n, K = Pi.shape
    if label is None:
        label = Pi.argmax(axis = 1)
    vertex = np.array([[-1/2,0],[1/2,0],[0,np.sqrt(3)/2]])

    coordinate = Pi @ vertex

    f = plt.figure()
    col_list = ['red','green','blue']
    for i in range(K):
        index = np.where(label == i)[0]
        plt.plot(coordinate[index,0],coordinate[index,1],color = col_list[i],marker = 'o',linewidth = 0)
    # draw the triangle
    plt.plot([-1/2,1/2],[0,0],color = 'black')
    plt.plot([-1/2,0],[0,np.sqrt(3)/2],color = 'black')
    plt.plot([1/2,0],[0,np.sqrt(3)/2],color = 'black')

    # add id of nodes
    if node_id:
        for i in range(n):
            plt.text(coordinate[i,0], coordinate[i,1], str(i), fontsize=15)

    plt.gca().set_aspect('equal', adjustable='box')
    return f



def visual_membership_T(Pi,label = None,nrow = 1,node_id = False, figsize_factor = 10, fontsize = 15):
    '''
    Visualize the membership of items accross T snapshots
    -----------------------------------------
    Pi (T x n x K array): membership matrix to be visualized, now only works for K=3
    '''
    T, n, K = Pi.shape
    cluster = Pi.argmax(axis = 2)
    if label is None:
        label = Pi.argmax(axis = 2)
    vertex = np.array([[-1/2,0],[1/2,0],[0,np.sqrt(3)/2]])

    coordinate = Pi @ vertex

    ncol = np.floor(T/nrow) + 1 * (T % nrow != 0)
    f = plt.figure(figsize=(nrow * 10, ncol * 10))

    col_list = ['red','green','blue']
    
    for t in range(T):
        a = plt.subplot(nrow, ncol, t + 1)
        for i in range(K):
            index = np.where(cluster[t,:] == i)[0]
            plt.plot(coordinate[t,index,0],coordinate[t,index,1],color = col_list[i],marker = 'o',linewidth = 0)
               
        # draw the triangle
        plt.plot([-1/2,1/2],[0,0],color = 'black')
        plt.plot([-1/2,0],[0,np.sqrt(3)/2],color = 'black')
        plt.plot([1/2,0],[0,np.sqrt(3)/2],color = 'black')
        plt.gca().set_aspect('equal', adjustable='box')
        
#         plt.show()
        # 
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
            # add id of nodes
        if node_id:
            for i in range(n):
                plt.text(coordinate[t,i,0], coordinate[t,i,1], str(i), fontsize=fontsize)

    plt.tight_layout()
    return f



def visual_gamma(gamma,label = None):
    '''
    gamma: [n, K] gamma matrix to be visualized, now only works for K=3
    '''
    n, K = gamma.shape
    C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
    C = np.tile(C,(1,K))
    Pi = np.exp(gamma - C)
    if label is None:
        label = Pi.argmax(axis = 1)
    vertex = np.array([[-1/2,0],[1/2,0],[0,np.sqrt(3)/2]])

    coordinate = Pi @ vertex

    f = plt.figure()
    col_list = ['red','green','blue']
    for i in range(K):
        index = np.where(label == i)[0]
        plt.plot(coordinate[index,0],coordinate[index,1],color = col_list[i],marker = 'o',linewidth = 0)
    # draw the triangle
    plt.plot([-1/2,1/2],[0,0],color = 'black')
    plt.plot([-1/2,0],[0,np.sqrt(3)/2],color = 'black')
    plt.plot([1/2,0],[0,np.sqrt(3)/2],color = 'black')
    plt.gca().set_aspect('equal', adjustable='box')
    return f


def cluster_error(Pi,Pi_hat):
    n, K = Pi.shape
    cluster = Pi.argmax(axis = 1)
    permut = list(permutations(range(K)))
    cluster_hat = Pi_hat.argmax(axis = 1)
    best_permut = (permut[0])
    best_error = sum(cluster != cluster_hat)
    for i in range(1,len(permut)):
        this_permut = permut[i]
        cluster_hat_copy = cluster_hat.copy()
        for j in range(K):
            cluster_hat_copy[cluster_hat == j] = this_permut[j]
        error = sum(cluster != cluster_hat_copy)
        if error < best_error:
            best_error = error
            best_permut = this_permut
    Pi_hat_copy = Pi_hat.copy()
    for i in range(K):
        Pi_hat[:,i] = Pi_hat_copy[:,best_permut[i]]
    cluster_hat = Pi_hat.argmax(axis = 1)
    return best_error, best_permut, cluster, cluster_hat



def compare_membership(Pi,gamma,label = None,size = [10,10],use_gamma = True):
    N, K = gamma.shape
    if use_gamma:
        C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
        C = np.tile(C,(1,K))
        Pi_hat = np.exp(gamma - C)
    else:
        Pi_hat = gamma
    if label is None:
        label = Pi.argmax(axis = 1)
        
    cluster = Pi.argmax(axis = 1)
    permut = list(permutations(range(K)))
    cluster_hat = Pi_hat.argmax(axis = 1)
    best_permut = (permut[0])
    best_error = sum(cluster != cluster_hat)
    for i in range(1,len(permut)):
        this_permut = permut[i]
        cluster_hat_copy = cluster_hat.copy()
        for j in range(K):
            cluster_hat_copy[cluster_hat == j] = this_permut[j]
        error = sum(cluster != cluster_hat_copy)
        if error < best_error:
            best_error = error
            best_permut = this_permut
    Pi_hat_copy = Pi_hat.copy()
    for i in range(K):
        Pi_hat[:,best_permut[i]] = Pi_hat_copy[:,i]
    cluster_hat = Pi_hat.argmax(axis = 1)
    
    vertex = np.array([[-1/2,0],[1/2,0],[0,np.sqrt(3)/2]])

    coordinate = Pi @ vertex
    coordinate_hat = Pi_hat @ vertex

    f = plt.figure()
    col_list = ['red','green','blue']
    for i in range(K):
        index = np.where(cluster == i)[0]
        plt.plot(coordinate[index,0],coordinate[index,1],color = col_list[i],marker = 'o',linewidth = 0)
        index = np.where(cluster_hat == i)[0]
        plt.plot(coordinate_hat[index,0],coordinate_hat[index,1],color = col_list[i],marker = 'x',linewidth = 0)
    for i in range(N):
        plt.plot([coordinate[i,0],coordinate_hat[i,0]],[coordinate[i,1],coordinate_hat[i,1]],
                 color = 'black',linewidth = 0.3)
        
    # draw the triangle
    plt.plot([-1/2,1/2],[0,0],color = 'black')
    plt.plot([-1/2,0],[0,np.sqrt(3)/2],color = 'black')
    plt.plot([1/2,0],[0,np.sqrt(3)/2],color = 'black')
    plt.gca().set_aspect('equal', adjustable='box')
    print("mismatched cluster labels: ", best_error)
    return f



def compare_membership_T(Pi,gamma,label = None,nrow = 1,use_gamma = True):
    T, N, K = gamma.shape
    Pi_hat = Pi.copy()
    if use_gamma:
        for t in range(T):
            gamma_this = gamma[t]
            C = np.log(np.exp(gamma_this).sum(axis = 1,keepdims = True))
            C = np.tile(C,(1,K))
            Pi_hat[t] = np.exp(gamma_this - C)
    else:
        Pi_hat = gamma
    if label is None:
        label = Pi.argmax(axis = 2)
        
    cluster = Pi.argmax(axis = 2)
    permut = list(permutations(range(K)))
    cluster_hat = Pi_hat.argmax(axis = 2)
    best_permut = (permut[0])
    best_error = np.sum(cluster != cluster_hat)
    for i in range(1,len(permut)):
        this_permut = permut[i]
        cluster_hat_copy = cluster_hat.copy()
        for t in range(T):
            for j in range(K):
                cluster_hat_copy[cluster_hat == j] = this_permut[j]
        error = np.sum(cluster != cluster_hat_copy)
        if error < best_error:
            best_error = error
            best_permut = this_permut
    Pi_hat_copy = Pi_hat.copy()
    for t in range(T):
        for i in range(K):
            Pi_hat[t,:,best_permut[i]] = Pi_hat_copy[t,:,i]
    cluster_hat = Pi_hat.argmax(axis = 2)
    
    vertex = np.array([[-1/2,0],[1/2,0],[0,np.sqrt(3)/2]])

    coordinate = Pi @ vertex
    coordinate_hat = Pi_hat @ vertex
    
    ncol = np.floor(T/nrow) + 1 * (T % nrow != 0)
    f = plt.figure(figsize=(nrow * 15, ncol * 15))

    col_list = ['red','green','blue']
    
    for t in range(T):
        a = plt.subplot(nrow, ncol, t + 1)
        for i in range(K):
            index = np.where(cluster[t,:] == i)[0]
            plt.plot(coordinate[t,index,0],coordinate[t,index,1],color = col_list[i],marker = 'o',linewidth = 0)
            index = np.where(cluster_hat[t,:] == i)[0]
            plt.plot(coordinate_hat[t,index,0],coordinate_hat[t,index,1],color = col_list[i],marker = 'x',linewidth = 0)
        for i in range(N):
            plt.plot([coordinate[t,i,0],coordinate_hat[t,i,0]],[coordinate[t,i,1],coordinate_hat[t,i,1]],
                     color = 'black',linewidth = 0.3)
        
        # draw the triangle
        plt.plot([-1/2,1/2],[0,0],color = 'black')
        plt.plot([-1/2,0],[0,np.sqrt(3)/2],color = 'black')
        plt.plot([1/2,0],[0,np.sqrt(3)/2],color = 'black')
        plt.gca().set_aspect('equal', adjustable='box')
        
#         plt.show()
        # 
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    print("mismatched cluster labels: ", best_error)
    return f



#######################################
#### functions for data generation ####
#######################################



def generate_LNMMSB(N,K,mu,Sigma,B):
  '''
  Arguments:
      N,K size
      mu (K x . array)
      Sigma, B (K x K array)
  
  Returns:
      E, 
      label, 
      Pi, 
      Gamma, 
      Z_indicator
  '''
  gamma = np.random.multivariate_normal(mu,Sigma,size = N) #[N,K]
  C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
  C = np.tile(C,(1,K))
  Pi = np.exp(gamma - C)
  label = Pi.argmax(axis = 1)
  
  Z_indicator = np.zeros((N,N,2))
  for i in range(N):
      z_tmp = np.where(np.random.multinomial(1,Pi[i,:],2 * N))[1]
      Z_indicator[i,:,0] = z_tmp[0:N]
      Z_indicator[:,i,1] = z_tmp[N:(2 * N)]
  Z_indicator = Z_indicator.reshape((N * N,2)).astype(int)
  Q = B[Z_indicator[:,0],Z_indicator[:,1]].reshape(N,N)

  # Q = np.zeros((N,N))
  # for i in range(N):
  #   for j in range(N):
  #     z1 = np.where(np.random.multinomial(1,Pi[i,:],1))[1]
  #     z2 = np.where(np.random.multinomial(1,Pi[j,:],1))[1]
  # # E_t is the observed adjacency matrix at time t
  #     Q[i,j] = B[z1,z2]

  E = np.random.binomial(1,Q)

  return E, label, Pi, gamma, Z_indicator



def generate_LNMMSB_symmetry(N,K,mu,Sigma,B):
  '''
  Arguments:
      N,K size
      mu (K x . array)
      Sigma, B (K x K array)
  
  Returns:
      E, 
      label, 
      Pi, 
      Gamma, 
      Z_indicator
  '''
  gamma = np.random.multivariate_normal(mu,Sigma,size = N) #[N,K]
  C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
  C = np.tile(C,(1,K))
  Pi = np.exp(gamma - C)
  label = Pi.argmax(axis = 1)
  
  Z_indicator = np.zeros((N,N,2))
  for i in range(N):
      z_tmp = np.where(np.random.multinomial(1,Pi[i,:],2 * N))[1]
      Z_indicator[i,:,0] = z_tmp[0:N]
      Z_indicator[:,i,1] = z_tmp[N:(2 * N)]
  Z_indicator = Z_indicator.reshape((N * N,2)).astype(int)
  
  Q = B[Z_indicator[:,0],Z_indicator[:,1]].reshape(N,N)

  # Q = np.zeros((N,N))
  # for i in range(N):
  #   for j in range(N):
  #     z1 = np.where(np.random.multinomial(1,Pi[i,:],1))[1]
  #     z2 = np.where(np.random.multinomial(1,Pi[j,:],1))[1]
  # # E_t is the observed adjacency matrix at time t
  #     Q[i,j] = B[z1,z2]
  E = np.random.binomial(1,Q)
  E = np.triu(E,1) + np.triu(E,1).T
    
  return E, label, Pi, gamma, Z_indicator



def generate_dMMSB_original(N,K,T,nv,Phi,A,Sigma,l,psi,b,B = None):
    '''
    generate T networks from dMMSB on page 543 of Xing et al 2010.
    -------------------
    Arguments:
        Sigma (K x K x T array)
    
    Returns
        E (T x N x N array): E[:,:,t] is the observed adjacency matrix at time t
        Mu (T x K array): Mu[:,t] is
        Label (T x N array)
        Pi (T x N x K array)
        Gamma (T x N x K array)
        Z_indicator (N x N x T x 2 array): Z_indicator[i,j,t,0] is the index of 1 in z_{i->j}^{(t)}, 
                              Z_indicator[i,j,t,1] is the index of 1 in z_{j->i}^{(t)}
        B (T x K x K array):
    '''
    Mu = np.zeros((K,T))
    Pi = np.zeros((N,K,T))
    Gamma = np.zeros((N,K,T))
    E = np.zeros((N,N,T)) # observed adjacency matrix of the network
    Z_indicator = np.zeros((N,N,2,T))
    Q = np.zeros((N,N,T))
    Label = np.zeros((N,T))

    Mu[:,0] = np.random.multivariate_normal(nv,Phi)
    for t in range(1,T):
        m = A @ Mu[:,t-1]
        Mu[:,t] = np.random.multivariate_normal(m,Phi)
    if B is None:
        B = np.zeros((K,K,T))
        eta = l + np.sqrt(psi) * np.random.randn(K**2).reshape((K,K))
        B[:,:,0] = np.exp(eta)/(1 + np.exp(eta))
        for t in range(1,T):
            eta = b * eta + np.sqrt(psi) * np.random.randn(K**2).reshape((K,K))
            B[:,:,t] = np.exp(eta)/(1 + np.exp(eta))

    for t in range(T):
        gamma = np.random.multivariate_normal(Mu[:,t],Sigma[:,:,t],size = N) #[N,K]
        C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
        C = np.tile(C,(1,K))
        pi = np.exp(gamma - C)
        label = pi.argmax(axis = 1)
        Pi[:,:,t] = pi
        Gamma[:,:,t] = gamma
        Label[:,t] = label
        
        #### generate z
        Z_indicator_t = np.zeros((N,N,2))
        for i in range(N):
            z_tmp = np.where(np.random.multinomial(1,Pi[i,:,t],2 * N))[1]
            Z_indicator_t[i,:,0] = z_tmp[0:N]
            Z_indicator_t[:,i,1] = z_tmp[N:(2 * N)]
        Z_indicator[:,:,:,t] = Z_indicator_t.astype(int)
        Z_indicator_t = Z_indicator_t.reshape((N * N,2)).astype(int)
        B_t = B[:,:,t]
        Q_t = B_t[Z_indicator_t[:,0],Z_indicator_t[:,1]].reshape(N,N)
        # E_t is the observed adjacency matrix at time t
        E_t = np.random.binomial(1,Q_t)
#         if symmetry:
#             E_t = np.triu(E_t,1) + np.triu(E_t,1).T
        E[:,:,t] = E_t
        Q[:,:,t] = Q_t
    Label = Label.astype(int)
    return E.transpose((2,0,1)), Mu.T, Label.T, Pi.transpose((2,0,1)), Gamma.transpose((2,0,1)), Z_indicator, B.transpose((2,0,1))



def generate_dMMSB(N,K,T,nv,Phi,Sigma_e,Sigma,l,psi,b,B = None):
    '''
    generate T networks from a modified dMMSB
    -------------------
    Arguments:
        Sigma (K x K x T array)
    
    Returns
        E (T x N x N array): E[:,:,t] is the observed adjacency matrix at time t
        Mu (T x K array): Mu[:,t] is
        Label (T x N array)
        Pi (T x N x K array)
        Gamma (T x N x K array)
        Z_indicator (N x N x T x 2 array): Z_indicator[i,j,t,0] is the index of 1 in z_{i->j}^{(t)}, 
                              Z_indicator[i,j,t,1] is the index of 1 in z_{j->i}^{(t)}
        B (T x K x K array):
    '''
    Mu = np.zeros((T,K))
    Pi = np.zeros((T,N,K))
    Gamma = np.zeros((T,N,K))
    E = np.zeros((T,N,N)) # observed adjacency matrix of the network
    Z_indicator = np.zeros((T,N,N,2))
    Q = np.zeros((T,N,N))
    Label = np.zeros((T,N))

    Mu[0,:] = np.random.multivariate_normal(nv,Phi)
    gamma = np.random.multivariate_normal(Mu[0,:],Sigma[0,:,:],size = N) #[N,K]
    C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
    C = np.tile(C,(1,K))
    pi = np.exp(gamma - C)
    label = pi.argmax(axis = 1)
    Pi[0,:,:] = pi
    Gamma[0,:,:] = gamma
    Label[0,:] = label
    
    for t in range(1,T):
        epsilon = np.random.multivariate_normal(np.zeros(K),Sigma_e,size = N) #[N,K]
        gamma_pre = gamma.copy() 
        gamma = gamma_pre + epsilon
        C = np.log(np.exp(gamma).sum(axis = 1,keepdims = True))
        C = np.tile(C,(1,K))
        pi = np.exp(gamma - C)
        label = pi.argmax(axis = 1)
        Pi[t,:,:] = pi
        Gamma[t,:,:] = gamma
        Label[t,:] = label
        
    if B is None:
        B = np.zeros((T,K,K))
        eta = l + np.sqrt(psi) * np.random.randn(K**2).reshape((K,K))
        B[0,:,:] = np.exp(eta)/(1 + np.exp(eta))
        for t in range(1,T):
            eta = b * eta + np.sqrt(psi) * np.random.randn(K**2).reshape((K,K))
            B[t,:,:] = np.exp(eta)/(1 + np.exp(eta))

    for t in range(T):
        #### generate z
        Z_indicator_t = np.zeros((N,N,2))
        for i in range(N):
            z_tmp = np.where(np.random.multinomial(1,Pi[t,i,:],2 * N))[1]
            Z_indicator_t[i,:,0] = z_tmp[0:N]
            Z_indicator_t[:,i,1] = z_tmp[N:(2 * N)]
        Z_indicator[t,:,:,:] = Z_indicator_t.astype(int)
        Z_indicator_t = Z_indicator_t.reshape((N * N,2)).astype(int)
        B_t = B[t,:,:]
        Q_t = B_t[Z_indicator_t[:,0],Z_indicator_t[:,1]].reshape(N,N)
        # E_t is the observed adjacency matrix at time t
        E_t = np.random.binomial(1,Q_t)
#         if symmetry:
#             E_t = np.triu(E_t,1) + np.triu(E_t,1).T
        E[t,:,:] = E_t
        Q[t,:,:] = Q_t
    Label = Label.astype(int)
    return E, Mu.T, Label.T, Pi, Gamma, Z_indicator, B
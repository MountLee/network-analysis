import numpy as np
import scipy
import cvxopt
import itertools
from sklearn.cluster import KMeans



def lsq_const(A,b,E = None,f = None,G = None,h = None,obj_only = True):
    '''
    solve a least square problem with constraints:

    min ||Ax-b||^2
    s.t. Ex = f
         Gx >= h
    -----------------
    Returns:
        x: solution
        obj: objective function at x
        only return obj by default
    '''
    P =  np.matmul(A.T,A) # make sure P is symmetric
    q = -np.matmul(A.T,b)
    
    f = np.array(f * 1.).reshape((-1,1))
    f_d = f.shape[0]
    E = np.array(E).reshape((f_d,-1))
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([-cvxopt.matrix(G), -cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(E), cvxopt.matrix(f)])
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
#     return solution
    x = np.array(sol['x']).reshape((P.shape[1],))
#     obj = np.array(sol['y'])[0,0]
    obj = np.sum((A @ x - b)**2)
    if obj_only:
        result = obj
    else:
        result = [x,obj]
    return result



def get_max_dist(points, vertex_ind):
    '''
    calculate the maximum distance from given points to the convex hull formed by the given vertices
    --------------------
    Arguments:
        points (L x (K - 1) array): coordinates of L points in R^{K - 1}
        vertex_ind (K x . array): indices of K vertices for the convex hull
    Return:
        max_dist: maximum among (L - K) values of distance
    '''

    L, K = points.shape
    K += 1
    
    nonvertex = np.setdiff1d(np.arange(L),vertex_ind)
    nonvertex = points[nonvertex,:]
    vertex = points[vertex_ind,:]
    
    dist = np.zeros(L-K)
    for i in range(L-K):
        node = nonvertex[i,:]
        dist[i] = lsq_const(A = vertex.T,b = node,E = np.ones(K),f = 1,G = np.eye(K),h = np.zeros(K))
    max_dist = max(dist)
    return max_dist



def get_membership(R, vertices, eig_value, eig_vector):
    '''
    Calculate memberships of nodes given centers of K communities
    -------------------------------
    Arguments:
        R (n x (K - 1) array): embedded points of n nodes
        vertices (K x (K - 1) array): centers of K communities
        eig_value (m x . array, m >= K): leading eigenvalues of the adjacency matrix
        eig_vector (n x m array, m >= 1): leading eigenvectors of the adjacency matrix
    Returns:
        memberships (n x K array): memberships of n nodes
        degrees (n x . array): degrees of n nodes
    '''
    n, K = R.shape
    K += 1
    memberships = np.zeros((n,K))
    for i in range(n):
        out = lsq_const(A = vertices.T,b = R[i,:],E = np.ones(K),f = 1,G = np.eye(K),h = np.zeros(K),obj_only = False)
        memberships[i,:] = out[0]
    
    if np.isnan(np.sum(memberships)):
        print("nan in membership")
    
    memberships[memberships > 1] = 1
    memberships[memberships < 0] = 0
    for i in range(n):
        memberships[i,:] = memberships[i,:]/np.sum(memberships[i,:])
        
    tildeV = np.concatenate((np.ones((K,1)),vertices),axis = 1)
    b1_inv = np.sqrt(np.diag(tildeV @ np.diag(eig_value[0:K]) @ tildeV.T))
    
    #### get degrees
    degrees = abs(eig_vector[:,0] * np.sum(memberships @ np.diag(b1_inv), axis = 1))
    
    #### get tilded memberships
    memberships = memberships @ np.diag(b1_inv)
    
    #### normalize
    for i in range(n):
        memberships[i,:] = memberships[i,:]/np.sum(memberships[i,:])
    return memberships, degrees



def vertex_search(points,K):
    '''
    find a subset of K points from given points such that the maximal distances to the convex hull of the subset is minimized
    ---------------------------
    Arguments:
        points (L x (K - 1) array): L points in R^{K-1}
        K: number of vertices to find
    Return:
        ind (K x . array): indices of selected vertices
        min_dist: distance corresponding to selected vertices
        vertices (K x (K - 1) array): selected vertices
    '''
    L = points.shape[0]
    
    index_matrix = np.array(list(itertools.combinations(np.arange(L),K))).T # K x (L choose K) array
    n_c = index_matrix.shape[1]
    dist = np.zeros(n_c)
    for i in range(n_c):
        dist[i] = get_max_dist(points = points,vertex_ind = index_matrix[:,i])
    
    ind = index_matrix[:,np.argmin(dist)]
    vertices = points[ind,:]
    min_dist = min(dist)
    return ind, min_dist, vertices



def vertex_hunting(R, K, verbose = False):
    '''
    Find a subset from given points such that the maximum distance to the convex hull of the subset is minimized
    reference: the vertec hunting algorithm in Jin et al 2016.
    ----------------------------------------------
    Arguments:
        R (n x (K - 1) array): set of n points in R^{K - 1}
        K: number of points in the subset
    Returns:
        L_select: number of centers for k-means 
        vertices (K x (K - 1) array): K vertices
        centers (L x (K - 1) array): L_select centers of k-means
    '''
    L_candidate = np.arange(2 * K + 1,4 * K + 1)
    n_L = len(L_candidate)
    
    #### sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
    #### precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')[source]
    out_list = []
    for L in L_candidate:
        if verbose:
            print('L = ', L, '\n')
        kmeans = KMeans(n_clusters = L,max_iter = 300, n_init = 200).fit(R)
        centers = kmeans.cluster_centers_
        out = vertex_search(points = centers, K = K)
        if verbose:
            print('dist = ', out[1], '\n')
        out_list.append(out + (centers,))
        
    delta_L = np.zeros(n_L)
    for i in range(n_L):
        if i == 0:
            kmeans = KMeans(n_clusters = K, max_iter = 300, n_init = 200).fit(R)
            v_L_1 = kmeans.cluster_centers_
        else:
            v_L_1 = out_list[i - 1][2]
        v_L = out_list[i][2]
        
        perm = list(itertools.permutations(np.arange(K)))
        delta = np.zeros(len(perm))
        for p in range(len(perm)):
            diff = max(np.sum((v_L[perm[p],:] - v_L_1)**2, axis = 1))
            delta[p] = diff
        delta = np.max(delta)
        
        delta_L[i] = delta / (1 + out_list[i][1])
        
    L_ind = np.argmin(delta_L)
    L_select = L_candidate[L_ind]

    if verbose:
        print('Select L = ', L_select, '\n')

    vertices, centers = out_list[L_ind][2:]
    return L_select, vertices, centers



def score(A, K, threshold = None):
    '''
    main function for mix-SCORE
    ---------------------------
    Arguments:
        A (n x n array): adjacency matrix of the network
        K number of communities
    Returns:
        R (n x (K - 1) array): embedded points of n nodes 
        eig_value (K x . array): leading K eigenvalues (orderd by magnitude)
        eig_vector (n x K array): leading K eigenvectors
    '''
    n = A.shape[0]
    if (A != A.T).any():
        A = A + A.T
        A[A != 0] = 1
    if (np.diag(A) != 1).any():
        A = A + np.eye(n)
        A[A != 0] = 1
    
    eig_value, eig_vector = scipy.sparse.linalg.eigsh(A * 1., min(K + 5,n))
    
    abs_eig = abs(eig_value)
    order_eig = np.argsort(-abs_eig)[:K]
    
    eig_value = eig_value[order_eig]
    eig_vector = eig_vector[:,order_eig]
    
#     R = (eig_vector[:,1:K].T / eig_vector[:,0]).T
    R = eig_vector[:,1:K] / np.tile(eig_vector[:,0].reshape((n,1)),(1,K-1))
    
    R = regularize(R, quantile = 0.05)
    
    if threshold is None:
        threshold = np.log(n)
    R[R > threshold] = threshold
    R[R < -threshold] = -threshold
    
    return R, eig_value, eig_vector



def mix_score(A,K,verbose = False):
    '''
    main function for mix-SCORE
    ---------------------------
    Arguments:
        A: (n x n array) the adjacency matrix of the network
        K: number of communities
    Returns:
        R (n x (K - 1) array): embedded points of n nodes
        vh_out[0], vh_out[1], vh_out[2]: output of vertex_hunting()
        memberships (n x K array): memberships of n nodes
        degrees (n x . array): degrees of n nodes
        purity (n x . array): purity of n nodes
        major_labels (n x . array): major community labels of n nodes
        B_hat (K x K array): estimated connectivity matrix
    '''
    score_out = score(A = A,K = K)
    
    if verbose:
        print('Get ratios \n')
    
    R = score_out[0]
    
    if verbose:
        print('Vertex hunting \n')
        
    vh_out = vertex_hunting(R = R, K = K, verbose = verbose)
    
    if verbose:
        print('Get the membership \n')
    
    member_out = get_membership(R = R, vertices = vh_out[1], eig_value = score_out[1], eig_vector = score_out[2])
    memberships, degrees = member_out
    
    if verbose:
        print('Get the purity scores and hard clustering results \n')
        
    purity = np.max(memberships,axis = 1)
    major_labels = np.argmax(memberships,axis = 1)

    B_hat = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            E_sub = A[np.ix_(np.where(major_labels == i)[0],np.where(major_labels == j)[0])]
            B_hat[i,j] = np.sum(E_sub) / (E_sub.shape[0] * E_sub.shape[1])
    
    return R, vh_out[0], vh_out[1], vh_out[2], memberships, degrees, purity, major_labels, B_hat



def regularize(R, quantile = 0.05):
    '''
    Regularize the given set of points by shrink some outliers to the centroid
    Arguments:
        R: (n x d array) given set of n points in R^{d}
        quatile: the proportion of the farest points to be identified as outliers, 0.05 by default.
    Returns:
        R: (n x d array) Regularized version of R
    '''
    d = R.shape[1]
    centroid = np.zeros(d)
    for i in range(d):
        centroid[i] = np.median(R[:,i])
    
    radius = np.sum((R - centroid)**2, axis = 1)
    threshold = np.quantile(radius,1 - quantile)
    outlier_id = np.where(radius > threshold)
    outliers = R[outlier_id,:]
    
    r = np.sum((outliers - centroid)**2, axis = 1)
    zoom = np.quantile(radius,0.5)
    for i in range(outliers.shape[0]):
        outliers[i,:] = np.sqrt(zoom/r) * (outliers[i,:] - centroid) + centroid
    R_reg = R.copy()
    R_reg[outlier_id,:] = outliers
    return R_reg


def score_dMMSB(K, E, verbose = 0):
    '''
    Arguments:
    K (int): number of clusters
    E (T x n x n array): observed network at all timesteps
    verbose (int): print count iterations when computing. 0: no print, 1: only outer iterations, 2: outer & inner iterations

    Returns:
    mu (T x K array):
    B (K x K array): role-compatibility matrix
    Sigma (T x K x K array): 
    '''
    T, n = E.shape[:2]
    Pi_list = []
    label_list = []
    
    for t in range(T):
        result = mix_score(A = E[t],K = K)
#         R = result[0]
#         vertices = result[2]
#         centers = result[3]
        Pi_hat = result[4]
        if t > 0:
            Pi_hat = match_two_pi(Pi_list[t-1],Pi_hat)
        Pi_list.append(Pi_hat)
        label = Pi_hat.argmax(axis = 1)
        label_list.append(label)
    Pi_list = np.array(Pi_list)
    
    B_list = []
    for t in range(T):
        B_hat = np.zeros((K,K))
        E_this = E[t]
        label_this = label_list[t]
        for i in range(K):
            for j in range(K):
                E_sub = E_this[np.ix_(np.where(label_this == i)[0],np.where(label_this == j)[0])]
                B_hat[i,j] = np.sum(E_sub) / (E_sub.shape[0] * E_sub.shape[1])
        B_list.append(B_hat)
    B_list = np.array(B_list)
        
    return Pi_list, B_list, label_list



def match_two_pi(Pi_1,Pi_2):
    '''
    reorder the K columns of Pi_2 to minimize the mismatched labels between Pi_1 and Pi_2
    -----------------------
    Arguments:
        Pi_1, Pi_2: (N x K array) each row sum up to 1
    Returns:
        Pi_2_reorder: (N x K array) Reordered version of Pi_2
    '''
    N, K = Pi_1.shape
   
    cluster_1 = Pi_1.argmax(axis = 1)
    permut = list(itertools.permutations(range(K)))
    cluster_2 = Pi_2.argmax(axis = 1)
    best_permut = (permut[0])
    best_error = np.sum(cluster_1 != cluster_2)
    for i in range(1,len(permut)):
        this_permut = permut[i]
        cluster_2_copy = cluster_2.copy()
        for j in range(K):
            cluster_2_copy[cluster_2 == j] = this_permut[j]
        error = np.sum(cluster_1 != cluster_2_copy)
        if error < best_error:
            best_error = error
            best_permut = this_permut
    Pi_2_reorder = Pi_2.copy()
    for i in range(K):
        Pi_2_reorder[:,best_permut[i]] = Pi_2[:,i]
    cluster_2 = Pi_2.argmax(axis = 1)
    
    return Pi_2_reorder
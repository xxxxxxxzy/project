#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 06:10:50 2021

@author: xuzhaoyang
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import scipy.interpolate
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import scipy
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.neighbors import NearestNeighbors
import igl  
from scipy.interpolate import interp1d


# X:2d coordinates
# Y:3d coordinates
# n:the number of data you want to use in train set
# i:seed, >0
def data(X,Y,n,i):
    """
    Divide the data set into training set and test set.
    """
    
    p, d = X.shape #p = n, d = 2
    
    # We want to registration from 2d to 3d, so we add 0 to 2d like (x,y,0)
    z = np.zeros(p)
    X = np.c_[X,z.T]
    
    P = (p-n)/p
    
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P, random_state=i, shuffle=True)
    
    return X_train, X_test, Y_train, Y_test
    

def eta(X,Y):
    """
    For 2d thin plate spline, r = cdist(X,Y)
    if r > 0, eta = (r^2)*log(r)
    if r = 0, eta = 0
    Parameters:
    -----------
    X:  np.array
    Y:  np.array
    Returns:
    --------
    eta:  np.array
    """
    
    r = cdist(X,Y)
    r[r < 0] = 0

    K = r*r*np.log(r)
    
    K[np.isnan(K)] = 0
    
    return K



def coefficient(X,Y,lam):
    """
    Given a set of control points and their corresponding points, compute the
    coefficients of the TPS interpolant deforming surface.
    Parameters:
    -----------
    X:  np.array
    obervations, 2d image coordinates
    Y:  np.array
    3d coordinates
    lam:
        smoothing parameter
    Returns:
    --------
    coe:  np.array
    It will output the parameter of the function:
        f(x)=beta_0+beta_1*x+sum(alpha_i*eta)
    """

    p, d = X.shape

    K = eta(X, X)
    P = np.hstack([np.ones((p, 1)), X])


    
    K = K + lam * np.identity(p)
    
    M2 = np.vstack([
        np.hstack([K, P]),
        np.hstack([P.T, np.zeros((d + 1, d + 1))])
    ])
    
    y = np.vstack([Y, np.zeros((d + 1, d))])
    
    #coe = np.linalg.solve(M, y)
    #coe, _, _, _ = np.linalg.lstsq(M2, y, None)
    coe = np.linalg.pinv(M2)@y
    
    return coe



def deform(X_test,X_train,Y_train):
    """
    Transform the source points from the original surface to the deformed surface.
    
    Parameters:
    -----------
    X_test: np.array
    2d image coordinates, source points which you want to deform, n*2 array
    X_train: np.array
    2d image coordinates, control points, n*2 array
    Y_train: np.array
    3d coordinates, corresponding target points on the deformed, n*3 array
    
    Returns:
    --------
    deformed_points : np.array
    n*3 array of the transformed point on the target surface
    """
    
    p , d = X_test.shape
    coe = coefficient(X_train,Y_train,0)
    K = eta(X_test, X_train)
    #K = eta(X_train, X_train)
    M = np.hstack([K, np.ones((p, 1)), X_test])
    Y = M@coe
    return Y
 


# X:2d coordinates
# Y:3d coordinates
# n: the size of train data
# t: the time of sampling
def distance(X,Y,n,t):
    """
    Calculate the mean and max distance between the deformed the surface and true surface.
    """

    # X(x1,x2) to X(x1,x2,0)
    a, d = X.shape #p = 60, d = 2
    z = np.zeros(a)
    X = np.c_[X,z.T]
    
    p = n/30
    
    epsilon = np.empty(t)
    dist = np.empty(n)
    for i in range(1,t+1):
        
        X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=p, random_state=i, shuffle=True)
        x_train,x_test, y_train, y_test = train_test_split(X_test,Y_test,test_size=0.5, random_state=10000, shuffle=True)

        dist = np.diagonal(cdist(deform(x_test,x_train,y_train),y_test))
        dist_max = dist.max()
        dist_mean = np.mean(dist)
        epsilon[i-1] = dist_mean/dist_max
        
    
    return epsilon
    



def surface_curvature(X,Y,Z):
    """
    Curvature of the surface. By first and seconde fundamental form to calculate the
    Principle Curvatures, Mean Curvatures and Guassian Curvautures
    """
    
    a, d=X.shape
    
    #First Derivatives
    Xv,Xu=np.gradient(X)
    Yv,Yu=np.gradient(Y)
    Zv,Zu=np.gradient(Z)
    
    #Second Derivatives
    Xuv,Xuu=np.gradient(Xu)
    Yuv,Yuu=np.gradient(Yu)
    Zuv,Zuu=np.gradient(Zu)   

    Xvv,Xuv=np.gradient(Xv)
    Yvv,Yuv=np.gradient(Yv)
    Zvv,Zuv=np.gradient(Zv) 
    
    #2D to 1D conversion 
    #Reshape to 1D vectors
    Xu=np.reshape(Xu,a*d)
    Yu=np.reshape(Yu,a*d)
    Zu=np.reshape(Zu,a*d)
    Xv=np.reshape(Xv,a*d)
    Yv=np.reshape(Yv,a*d)
    Zv=np.reshape(Zv,a*d)
    Xuu=np.reshape(Xuu,a*d)
    Yuu=np.reshape(Yuu,a*d)
    Zuu=np.reshape(Zuu,a*d)
    Xuv=np.reshape(Xuv,a*d)
    Yuv=np.reshape(Yuv,a*d)
    Zuv=np.reshape(Zuv,a*d)
    Xvv=np.reshape(Xvv,a*d)
    Yvv=np.reshape(Yvv,a*d)
    Zvv=np.reshape(Zvv,a*d)
    
    Xu=np.c_[Xu, Yu, Zu]
    Xv=np.c_[Xv, Yv, Zv]
    Xuu=np.c_[Xuu, Yuu, Zuu]
    Xuv=np.c_[Xuv, Yuv, Zuv]
    Xvv=np.c_[Xvv, Yvv, Zvv]
    
    
    #% First fundamental Coeffecients of the surface (E,F,G)
	
    E=np.einsum('ij,ij->i', Xu, Xu) 
    F=np.einsum('ij,ij->i', Xu, Xv) 
    G=np.einsum('ij,ij->i', Xv, Xv) 
    
    m=np.cross(Xu,Xv,axisa=1, axisb=1) 
    p=np.sqrt(np.einsum('ij,ij->i', m, m)) 
    n=m/np.c_[p,p,p]
    
    # n is the normal
    #% Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
    L= np.einsum('ij,ij->i', Xuu, n) #e
    M= np.einsum('ij,ij->i', Xuv, n) #f
    N= np.einsum('ij,ij->i', Xvv, n) #g
    
    # Alternative formula for gaussian curvature in wiki 
    # K = det(second fundamental) / det(first fundamental)
    #% Gaussian Curvature
    K=(L*N-M**2)/(E*G-F**2)
    K=np.reshape(K,a*d)
	#print(K.size)
    
    #wiki trace of (second fundamental)(first fundamental inverse)
    #% Mean Curvature
    H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
    #print(H.shape)
    H = np.reshape(H,a*d)
    
    #% Principle Curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)
    
    #[Pmax, Pmin]
    Principle = [Pmax,Pmin]
    
    return H




def surface_region_curvature(xtest,xtrain,ytrain):
    
    """
    Curvature of the convex hull on surface. By first and seconde fundamental form to calculate the
    Principle Curvatures, Mean Curvatures and Guassian Curvautures
    
    Parameters:
    -----------
    xtest: np.array
    2d image coordinates, source points which you want to deform, n*2 array
    xtrain: np.array
    2d image coordinates, control points, n*2 array
    ytrain: np.array
    3d coordinates, corresponding target points on the deformed, n*3 array
    
    Returns:
    --------
    Mean curvature : float
    the mean curvature of all the points on convex hull
    """
    
# use convex hull to find the good region on 2d image
    hull = ConvexHull(xtest[:,0:2])
    ind = hull.vertices
        
# move on the tps surface 
# first find the correspond region points
# second plot the tps surface gird
# thrid find the points in the region
# finally compute the mean curvature

    de = deform(xtest,xtrain,ytrain)
    region_point = de[ind]
    #ch = np.vstack([de[a],de[a[0]]])
    #x, y, z = zip(*ch)


    x_grid = np.linspace(de[:,0].min(), de[:,0].max(), len(de[:,0]))
    y_grid = np.linspace(de[:,1].min(), de[:,1].max(), len(de[:,1]))
    
    spline = sp.interpolate.Rbf(de[:,0],de[:,1],de[:,2],function='cubic',smooth=0)
    
    B1, B2 = np.meshgrid(x_grid, y_grid)
        
    X = B1.reshape(-1)
    Y = B2.reshape(-1)
    
    Z = spline(B1,B2)
    Z1 = Z.reshape(-1)

    lab = np.vstack([X, Y, Z1])
       
    region_tps = Path(region_point[:,0:2]) # make a polygon
    grid = region_tps.contains_points(lab[0:2,:].T)
    np.sum(grid!=0)
    i = [i for i in range(len(grid)) if grid[i] == True]
    
    cur = surface_curvature(B1,B2,Z)
    M = np.sum(np.abs(cur[i]))/len(i)
    
    return M


# X:2d coordinates
# Y:3d coordinates
# n:the number of data you want to use in train set
# i:seed, >0
# m:the time you want to random the the train set    
def mean_surface_curvature(X,Y,n,m):
    
    all_curvature = 0
    for i in range (1,m+1):
        X_train, X_test, Y_train, Y_test = data(X,Y,n,i)
        curvature = surface_region_curvature(X_test,X_train,Y_train)
        all_curvature = all_curvature + curvature
        
    M = all_curvature/m
    
    return M
    
      
    
    
# Output:
#    v[indd]:region points on 3d traingle, the coordinates of 3d traingle
#    de:2d tps landmarks coordinates
#    region_points:points coordinates of convex hull region on tps surface
#    X_test[:,0:2]:2d points
#    Y_test:3d points
#    simplices:for plot 2d image with convell hull
def nearstraingle(X,Y,n,i,v):
    # use convex hull to find the good region on 2d image
    
    X_train, X_test, Y_train, Y_test = data(X,Y,n,i)
    
    hull = ConvexHull(X_test[:,0:2])
    ind = hull.vertices
    simplices = hull.simplices
        

    de = deform(X_test,X_train,Y_train)
    region_point = de[ind]
    
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(v)
    indd = knn.kneighbors(region_point, 3, return_distance=False)
    
    return v[indd],de,region_point,X_test[:,0:2],Y_test,simplices






def knearstraingle(X,Y,n,i,v):
    '''
    Find the corresponding vertex of 2d convex hull region points.
    Parameters
    ----------
    X : np.array
        2d image coordinates
    Y : np.array
        3d image coordinates of corresponding X
    n : int
        the size to train the tps deform
    i : int
        seed
    v : np.array
        the vertex

    Returns
    -------
    vertex: np.array
        the corresponding vertex of 2d convex hull region points
    simplices : np.array
        the order of convex hull points
    ind : TYPE
        the index of convex hull points from X

    '''
    # use convex hull to find the good region on 2d image
    
    X_train, X_test, Y_train, Y_test = data(X,Y,n,i)
    
    hull = ConvexHull(X)
    ind = hull.vertices
    simplices = hull.simplices
    
    p, d = X.shape #p = 60, d = 2
    z = np.zeros(p)
    Xnew = np.c_[X,z.T]


    de = deform(Xnew,X_train,Y_train)
    region_point = de[ind]
    
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(v)
    indd = knn.kneighbors(region_point, 3, return_distance=False)
    
    return v[indd],de,region_point,X_test[:,0:2],Y_test,simplices,ind

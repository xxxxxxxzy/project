#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 06:12:42 2021

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
import TPS



def Processing_3dmesh(v,f):
    """
    Data processing
    
    Parameters:
    -----------
    v: np.array
    vertex of 3d mesh
    f: np.array
    faces of 3d mesh
    Returns:
    --------
    v:np.array
    vertex of 3d mesh
    faces of 3d mesh
    Reference:
    ----------
    https://github.com/libigl/libigl-python-bindings/issues/54
    """
    #components = igl.face_components(f)
    sv,_,_,sf = igl.remove_duplicate_vertices(v,f,1e-7)
    sc = igl.face_components(sf)
    unique, counts = np.unique(sc, return_counts=True)
    dict(zip(unique, counts))
    f_lagest_component = sf[np.where(sc==0)]
    vn,fn,_,_ = igl.remove_unreferenced(sv,f_lagest_component)
    return vn,fn





# LSCM Flatten
def Flatten(v,f):
    """
    LSCM Flatten
    
    Parameters:
    -----------
    v: np.array
    vertex of 3d mesh
    f: np.array
    faces of 3d mesh
    Returns:
    --------
    uv: np.array
    uv coordinates
    """
    b = np.array([2, 1]) # Fix two points on the boundary
    bnd = igl.boundary_loop(f) # Compute ordered boundary loops for a manifold mesh and return the longest loop in terms of vertices.
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    ret,uv = igl.lscm(v, f, b, bc) # LSCM parametrization
    return uv


# uv_all: all the uv coordinates
# uv-landmarks: the uv points that you want to compute the 3d coordinates
# vn: all the 3d coordinates
# the code was written by the algorithm from:
#    https://computergraphics.stackexchange.com/questions/8470/how-to-get-the-3d-position-for-the-point-with-0-0-uv-coordinates
def Flatten_inverse(uv_all, uv_landmark, vn):
    """
    Flatten inverse. Get the vertex from the uv.
    
    Parameters:
    -----------
    uv_all: np.array
    the uv of the whole patch
    uv_landmarks: np.array
    the uv which you want to know its corresponds vertex
    vn: np.array
    the vertex of the whole patch
    Returns:
    --------
    vertex: np.array
    the corresponding vertex of the uv_landmarks
    Reference:
    ----------
    https://computergraphics.stackexchange.com/questions/8470/how-to-get-the-3d-position-for-the-point-with-0-0-uv-coordinates
    
    """
    
    p,d = uv_landmark.shape # p the number of landmarks; d = 2
    
    knn = NearestNeighbors(n_neighbors=2) # Use knn to find the nearest 3 points
    knn.fit(uv_all)
    ind = knn.kneighbors(uv_landmark, 3, return_distance=False) # dim:(n,3)
    
    
    uv_traingle = uv_all[ind] # dim: (100,3,2) the nearest traingle on uv
    vn_traingle = vn[ind] # dim: (100,3,3) the nearest traingle on 3d mesh
    
    
    
    m1 = uv_traingle[:,1] - uv_traingle[:,0] # (u_2,v_2)-(u_1,v_1) the length of one side; dim: (100,2) 
    m2 = uv_traingle[:,2] - uv_traingle[:,0] # (u_3,v_3)-(u_1,v_1) the length of one side; dim: (100,2) 
    
    a = np.hstack([m1,m2])


    M = a.reshape(p,2,2)
    M = M.transpose((0,2,1)) # traspose each matrix; dim(100,2,2); M = (m1.T m2.T)
    
    inv = np.linalg.inv(M)
    
    # find the affine coordinates corresponding to the point
    # (lambda_1 lambda_2).T = M^(-1) * (u-u_1 v-v_1).T
    # (u v) is the uv coordinate of the landmarks
    # (u_1 v_1) is the uv coordinate of a points of the uv traingle which that landmark inside
    lam = inv@((uv_landmark - uv_traingle[:,0]).reshape(p,2,1)) 
    
    # vertex = (1-lambda_1-lambda_2) * v1 + lambda_1 * v2 + lambda_2 * v3
    # v1,v2,v3 are the coordinates of the respective triangle vertices
    vertex = (1-lam[:,0]-lam[:,1])*vn_traingle[:,0]+lam[:,0]*vn_traingle[:,1]+lam[:,1]*vn_traingle[:,2]

    return vertex


'''
def spline_image_to_uv(v,uv,point2d,point3d,image_points):
    
    point3d = np.round(point3d,6)
    a = point3d[:,0]
    b = v[:,0]
    
    ind = np.in1d(b,a).nonzero()[0]
    ind2 = np.in1d(a,v[ind,0]).nonzero()[0]

    y = uv[ind]
    x = point2d[ind2]

    xu = x[:,0]
    xv = x[:,1]
    yu = y[:,0]
    yv = y[:,1]

    xu = xu.reshape(xu.shape[0],1)
    yu = yu.reshape(yu.shape[0],1)

    xv = xv.reshape(xv.shape[0],1)
    yv = yv.reshape(yv.shape[0],1)
    
    u = np.hstack([xu,yu])
    u = u[np.lexsort([u.T[0]])]
    
    v = np.hstack([xv,yv])
    v = v[np.lexsort([v.T[0]])]

    cubic_u = interp1d(u[:,0], u[:,1], kind='cubic')
    cubic_v = interp1d(v[:,0], v[:,1], kind='cubic')
    
    u = cubic_u(image_points[:,0])
    v = cubic_v(image_points[:,1])
    
    u = u.reshape(u.shape[0],1)
    v = v.reshape(v.shape[0],1)
    
    uv = np.hstack([u,v])
    
    return uv
'''

def spline_image_to_uv(v,uv,point2d,point3d,image_points):
    
    """
    Get the uv from the 2d image coordinates by spline(thin plate spline)
    
    Parameters:
    -----------
    v: np.array
    the vertex of the whole patch
    uv: np.array
    the uv of the whole patch
    point2d: np.array
    the 2d image coordinates
    point3d: np.array
    the 3d coordinates of corresponding the point2d
    image_points: np.array
    the target points which you want to find the corresponding uv
    Returns:
    --------
    uv: np.array
    the corresponding uv of the image_points  
    """
    
    # We use blender to output the 3d landmarks coordinates and blender round to the 8 decimal place
    # We use libigl to output the whole vertex coordinates and it round to the 6 decimal place
    
    point3d = np.round(point3d,6) # round the landmarks from blender to 6 decimal place
    a = point3d[:,0]
    b = v[:,0]
    
    ind = np.in1d(b,a).nonzero()[0] # find the index of landmarks in the whole patch 

    y = uv[ind] # find the uv coordinates corresponding to the landmarks
    x = point2d
    
    a = TPS.deform(image_points, x, y)
    
    return a


def spline_uv_to_image(uv,point2d,uv_points):
    
    """
    Get the 2d image coordinatesuv from the 2d image coordinates by spline(cubic)(Need to Modify!!!) 
    """
        
    
    x = uv_points[:,0]
    y = uv_points[:,1]

    t = np.logical_and(np.logical_and(x>=uv[:,0].min(), x<=uv[:,0].max()),np.logical_and(y>=uv[:,1].min(), y<=uv[:,1].max()))
    a = uv_points[[i for i, x in enumerate(t) if x]]
    
    uv_points = a
    

    y = uv
    x = point2d
    
    xu = x[:,0]
    xv = x[:,1]
    yu = y[:,0]
    yv = y[:,1]

    xu = xu.reshape(xu.shape[0],1)
    yu = yu.reshape(yu.shape[0],1)

    xv = xv.reshape(xv.shape[0],1)
    yv = yv.reshape(yv.shape[0],1)
    
    u = np.hstack([xu,yu])
    u = np.unique(u,axis=0)
    u = u[np.lexsort([u.T[1]])]
    
    v = np.hstack([xv,yv])
    v = np.unique(v,axis=0)
    v = v[np.lexsort([v.T[1]])]

    cubic_u = interp1d(u[:,1], u[:,0], kind='cubic')
    cubic_v = interp1d(v[:,1], v[:,0], kind='cubic')

    x = cubic_u(uv_points[:,0])
    y = cubic_v(uv_points[:,1])
    
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    
    xy = np.hstack([x,y])

    return xy
    
    

#for 3d mesh cutting
def region_3d(v_region):
    '''
    For mesh cut
    '''
    
    x3d=v_region[:,:,0]
    y3d=v_region[:,:,1]
    z3d=v_region[:,:,2]
    
    return x3d.min(),x3d.max(),y3d.min(),y3d.max(),z3d.min(),z3d.max()


def plot_3d(Y,v,region_point,deform3d,vn,fn):
    
    Y = np.round(Y,6)
    a = Y[:,0]
    b = v[:,0]
    
    b = b.tolist()
    a = a.tolist()
    ind = [b.index(i) for i in a]

    Y = v[ind]

    
    #vertex3d,deform3d,region_point,new2d,new3d,simplices = nearstraingle(point2d,point3d,30,1,v)
    
    
    #uv = spline_image_to_uv(v,uv,new2d,new3d,points)
    #xy = spline_uv_to_image(v,uv,new2d,new3d,uv_points)
    

    
    
    #plot convex hull region
    region_point = np.vstack([region_point,region_point[0,:]])
    xs = region_point[:,0] 
    ys = region_point[:,1] 
    zs = region_point[:,2] 
    
    
    #plot tps surface
    x = deform3d[:,0]
    y = deform3d[:,1]
    z = deform3d[:,2]
    
    x_grid = np.linspace(x.min(), x.max(), len(x))
    #print(x_grid)
    y_grid = np.linspace(y.min(), y.max(), len(y))
    
    
    B1, B2 = np.meshgrid(x_grid, y_grid)
    
    
    spline = sp.interpolate.Rbf(x,y,z,function='cubic',smooth=0)
    
    Z = spline(B1,B2)
    
    
    
    fig = plt.figure(figsize=(10,6))
    
    ax = axes3d.Axes3D(fig)
    
    ax.plot_wireframe(B1, B2, Z) #tps grid
    ax.plot_surface(B1, B2, Z,alpha=0.2) #tps grid
    ax.plot(xs, ys, zs) #convex hull
    ax.plot_trisurf(vn[:,0], vn[:,1], vn[:,2], triangles = fn, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False) #3d mesh
    
    #ax.scatter3D(x3d,y3d,z3d, c='r')
    
    return






def plot_2d(X,simplices,n):
    
    
    plt.plot(X[:,0], X[:,1], 'o')
    for simplex in simplices:
        plt.plot(X[simplex, 0], X[simplex, 1], 'k-')
    
    x_2dgrid = np.linspace(X[:,0].min(), X[:,0].max(), n)
    #print(x_grid)
    y_2dgrid = np.linspace(X[:,1].min(), X[:,1].max(), n)
    
    
    B1, B2 = np.meshgrid(x_2dgrid, y_2dgrid)
    

    x = B1.reshape(-1)
    y = B2.reshape(-1)
    grid = np.zeros(shape=(len(x),2))
    grid[:,0] = x
    grid[:,1] = y

    plt.scatter(B1, B2,c = 'r')
    plt.plot(B1,B2,c='grey')
    plt.plot(B1.T,B2.T,c='grey')
    #for i in range (0,len(x_2dgrid)):
    #    plt.plot(x_2dgrid[i],y_2dgrid)
    plt.show()
    
    return grid




from shapely.geometry import MultiPoint
from shapely.geometry import Polygon

def region_points(grid,polygon,landmarks):
    '''
    find the points in polygon

    Parameters
    ----------
    grid : np.array
        all the point coordinates which generated by n*n grid.
    polygon : TYPE
        the polygon
    landmarks : np.array
        the points on the polygon line

    Returns
    -------
    points: np.array
        all the points in the polygon include the points on the boundary line

    '''
    
    points1 = MultiPoint(grid)
    inde = []
    for i in range (0,len(points1)):
        if polygon.intersects(points1[i]) == True:
            inde.append(i)
            
    return np.vstack((grid[inde],landmarks))



def region_points2(grid,polygon):
    '''
    find the points in polygon

    Parameters
    ----------
    grid : np.array
        all the point coordinates which generated by n*n grid.
    polygon : TYPE
        the polygon

    Returns
    -------
    points: np.array
        all the points in the polygon not include the points on the boundary line
    index: list
        the index of the points inside the polygon

    '''
    
    points1 = MultiPoint(grid)
    inde = []
    for i in range (0,len(points1)):
        if polygon.intersects(points1[i]) == True:
            inde.append(i)
            
    return grid[inde],inde


#find the over region: we have two patch, patch i and patch j. 
#And this function will find the overlapping region on patch i.   
    #back_2d:the 2d coordinates of patch i
    #vf:the vertex of 3d patch i
    #ff:the face of 3d patch i
    #uvf:the uv of patch i
    #back_3d:the 3d coordinates of patch i
    #vs:the vertex of 3d patch j
    #fs:the face of 3d patch j
def over_region(back_2d,simplices,n,polygon,vf,ff,uvf,back_3d,vs,fs):
    '''
    Find the over region: we have two patch, patch i and patch j. 
    And this function will find the overlapping region on patch i. 

    Parameters
    ----------
    back_2d : np.array
        the 2d coordinates of patch i
    simplices : np.array
        the order of points on convex hull
    n : int
        the number of the points you want to generate on the image
    polygon : TYPE
        polygon
    vf : np.array
        the vertex of 3d patch i
    ff : np.array
        the face of 3d patch i
    uvf : np.array
        the uv of patch i
    back_3d : np.array
        the 3d coordinates of patch i
    vs : np.array
        the vertex of 3d patch j
    fs : np.array
        the face of 3d patch j

    Returns
    -------
    points : np.array
        the coordinates of the points both on the patch i and patch j

    '''
    
    # find the grid points in convell hull of patch1
    grid11 = plot_2d(back_2d,simplices,n)
    all2d_back1 = region_points(grid11,polygon,back_2d)


    # 2d image coordinates to uv coordinates
    a = spline_image_to_uv(vf,uvf,back_2d,back_3d,all2d_back1)
    
    # uv coordinates to 3d coordinates
    b = Flatten_inverse(uvf, a, vf)


    fig = plt.figure(figsize=(10,6))   
    ax = axes3d.Axes3D(fig)
    ax.set_zlim(3, 5)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 4)
    
    ax.scatter(b[:,0], b[:,1], b[:,2])
    ax.plot_trisurf(vf[:,0], vf[:,1], vf[:,2], triangles = ff, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False) #3d mesh
    #ax.plot_trisurf(v22[:,0], v22[:,1], v22[:,2], triangles = f22, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False) #3d mesh
    
    c = vs[:,0:2]
    hull = ConvexHull(c)

    ind1to2 = hull.vertices
    c[ind1to2]
    
    polygon3 = Polygon(c[ind1to2])

    d = b[:,0:2]
    
    _,inddd = region_points2(d, polygon3)
    
    
    b1 = b[inddd]
 
    fig = plt.figure(figsize=(10,6))
    
    ax = axes3d.Axes3D(fig)
    
    
    ax.scatter(b1[:,0], b1[:,1], b1[:,2])
    #ax.plot_trisurf(v11[:,0], v11[:,1], v11[:,2], triangles = f11, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False) #3d mesh
    ax.plot_trisurf(vs[:,0], vs[:,1], vs[:,2], triangles = fs, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False) #3d mesh
    
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(vs)
    dis,indddd = knn.kneighbors(b1, 3, return_distance=True)
    
    e = (dis[:,0]+dis[:,1]+dis[:,2])/3
    
    #eind = np.where(e<0.5)
    #eind = list(np.where(e>0.5))
    #print(type(eind))
    aaaa = np.array(inddd)[np.where(e<0.5)]
    bbbb = aaaa.tolist()
    cccc = all2d_back1[bbbb]

    
    return cccc

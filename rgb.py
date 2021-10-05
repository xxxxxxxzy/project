import numpy as np
import igl
from scipy.sparse.linalg import spsolve


back = '/Users/xuzhaoyang/Desktop/back.off'
back1 = '/Users/xuzhaoyang/Desktop/back1.off'
back2 = '/Users/xuzhaoyang/Desktop/back2.off'


def read_off(path):
    '''

    Parameters
    ----------
    path : path
        file.off

    Returns
    -------
    U : array
        rgb array of whole patch
    U_tilde : array
        random change 5% rgb of whole patch

    '''

    with open(path, 'r') as f:
        a = f.read().split()
        b = a[4:] # remove 'COFF' and other lines
        
        # list to array
        file_float = []
        for num in b:
            file_float.append(float(num))
        c = np.array(file_float)
        d = c.reshape(-1,7)
        
        # only keep rgb lines, remove the index of vertex lines
        ind = list(d[:,0]).index(3)
        vertex = d[0:ind,0:3]
        rgba = d[0:ind,3:7] # include alpha, RGBA
        U = d[0:ind,3:6]
        
        p,d = U.shape # p = n, d = 3
        
        # randomly choose 5% from the whole patch, and change them
        ran_index = np.random.random_integers(0,p,size = int(p*0.05))
        U_tilde = U
        for i in ran_index:
            U[i] = np.random.random_integers(1,255,size=3)
            
    return vertex,U,U_tilde

V1,U1,U1_tilde = read_off(back1)
V2,U2,U2_tilde = read_off(back2)
V,U,_ = read_off(back)

p1,_ = V1.shape
p2,_ = V2.shape
p,_ = V.shape

v1_ind = []
for i in range(0,p1):
    for j in range(0,p):
        if np.all(V1[i] == V[j]):
            v1_ind.append(j)
            print(j)
            break
print(v1_ind)

v2_ind = []
for i in range(0,p2):
    for j in range(0,p):
        if np.all(V2[i] == V[j]):
            v2_ind.append(j)
            print(j)
            break
print(v2_ind)

inter = list(set(v1_ind)&set(v2_ind))
            
U1_delta = U1-U1_tilde
U2_delta = U2-U2_tilde
    

term1 = np.zeros((p,3))
for i in range (0,len(v1_ind)):
    term1[v1_ind[i]] = U1_delta[i]

term2 = np.zeros((p,3))
for i in range (0,len(v2_ind)):
    term2[v2_ind[i]] = U2_delta[i]


term = term1**2 + term2**2

for i in inter:
    term[i] = term[i]/2

R = term[:,0]
G = term[:,1]
B = term[:,2]

v,f = igl.read_triangle_mesh(back)

l = igl.cotmatrix(v, f)

n = igl.per_vertex_normals(v, f)*0.5+0.5
c = np.linalg.norm(n, axis=1)


vs = [v]
cs = [c]
for i in range(10):
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    s = (m - 0.001 * l)
    b = m.dot(v)
    v = spsolve(s, m.dot(v))
    n = igl.per_vertex_normals(v, f)*0.5+0.5
    c = np.linalg.norm(n, axis=1)
    vs.append(v)
    cs.append(c)
    
print(len(c))
print(len(vs))




import numpy as np
import igl
from scipy.sparse.linalg import spsolve
import scipy as sp


insect = '/Users/xuzhaoyang/Desktop/graphosoma/Graphosoma.obj'
v,f = igl.read_triangle_mesh(insect)

b = np.array([4331, 5957])
bc = np.array([1., -1.])
B = np.zeros((v.shape[0], 1))

## Construct Laplacian and mass matrix
L = igl.cotmatrix(v, f)
M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
Minv = sp.sparse.diags(1 / M.diagonal())

## Bi-Laplacian
Q = L @ (Minv @ L)

## Solve with only equality constraints
Aeq = sp.sparse.csc_matrix((0, 0))
Beq = np.array([])
_, z1 = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, True)

## Solve with equality and linear constraints
Aeq = sp.sparse.csc_matrix((1, v.shape[0]))
Aeq[0,6074] = 1
Aeq[0, 6523] = -1
Beq = np.array([0.])
_, z2 = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, True)

## Normalize colors to same range
min_z = min(np.min(z1), np.min(z2))
max_z = max(np.max(z1), np.max(z2))
z = [(z1 - min_z) / (max_z - min_z), (z2 - min_z) / (max_z - min_z)]
z = z[0]*255

## Plot the functions
p = subplot(v, f, z[0], shading={"wireframe":False}, s=[1, 2, 0])
subplot(v, f, z[1], shading={"wireframe":False}, s=[1, 2, 1], data=p)
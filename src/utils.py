import glob as glob
import numpy as np
from paths import *
import json
import math
from typing import *
from scipy.optimize import minimize
from scipy.io import loadmat

def GetSequences(type: str = '3d_cam') -> np.ndarray:
    S = []
    for path in glob.glob(SEQUENCES_PATH):
        with open(path, 'r') as f:
            data = json.load(f)
        for frame in data['frames']:
            S.append(frame[type])
    S = np.array(S)
    return S

def fillGap(A: np.ndarray, i: int, j: int, val: int) -> np.ndarray:
    if np.isnan(val):
        mask = ~np.any(np.isnan(A), axis=2)
    else:
        mask = ~np.any(A == val, axis=2)
        
    x, y = np.where(mask)
    
    dist = (x - i)**2 + (y - j)**2
    
    ind = np.argmin(dist)
    
    return A[x[ind], y[ind], :]

def getNormal(x1: np.ndarray, a: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, bool]:
    nth = 1e-4
    
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:
        n = np.cross(x, x1)
        flg = True
    else:
        n = np.cross(a, x)
        flg = False
    
    n = n / np.linalg.norm(n)
    
    return n, flg

def getPointsOnPlane(X: np.ndarray, v: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = v / np.linalg.norm(v[0:3])
    
    e1 = v[0:2]
    T = gramschmidt(np.column_stack([e1, e2, np.cross(e1, e2)]))
    x = (T[:,1:3].T) @ X
    
    bb = minBoundingBox(x)
    points = T @ np.vstack((-v[3] * np.ones((1, 4)), bb))

    tri = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int)
    
    return points, tri

def getPointsOnPlane1(X: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = v / np.linalg.norm(v[0:3])
    
    U, _, _ = np.linalg.svd(v[0:3] @ np.ones((1,3)), full_matrices=True)
    B = U[:, 1:3]
    T = np.column_stack([B, v[0:3]])
    
    x = B.T @ X
    
    bb = minBoundingBox(x)
    points = T @ np.vstack((bb, -v[3] * np.ones((1,4))))
    
    tri = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int)
    
    return points, tri

def getProjectedBounds(S, v) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = v / np.linalg.norm(v[0:3])
    b1 = v[0:3]
    
    n = S.shape[1]
    sm = np.mean(S, axis=1, keepdims=True)
    Sd = S - (sm @ np.ones((1,n)))
    
    D, U = np.linalg.eig(Sd @ Sd.T)
    ind = int(np.argmax(D))
    b2 = U[:, ind]
    b3 = np.cross(b1, b2)
    T = gramschmidt(np.column_stack([b1, b2, b3]))
    
    prjS = T[:, 1:3].T @ S
    
    bnds = np.array([
        np.min(prjS[0, :]),
        np.max(prjS[0, :]),
        np.min(prjS[1, :]),
        np.max(prjS[1, :])
    ])
    
    return bnds, b2, prjS
    
def costFnc(x: np.ndarray) -> float:
    xd = x[0:3]
    xx2 = xd.T @ xd
    f = x[3]**2 / xx2
    
    return f

def gradient(x: np.ndarray) -> float:
    xd = x[0:3]
    xx2 = xd.T @ xd
    
    g = (2. / xx2**2) * (np.append([0, 0, 0], [x[3]]) * xx2 - np.append(xd, [0]) * x[3]**2)
        
    return g

def getSeperatingPlane(X: np.ndarray, x0: np.ndarray = None) -> Tuple[np.ndarray, float]:
    N = X.shape[1]
    A = np.column_stack([X.T, np.ones(N,1)])
    b = np.zeros(N)
    
    if x0 is None or np.linalg.norm(x0[0:3]) == 0:
        x0 = np.ones(4)
        
    options = {
        'disp': False,
        'maxiter': 3000,
        'ftol': 1e-20,
    }
        
    constraints = {
        'type': 'ineq',
        'fun': lambda x: -(A @ x),
        'jac': lambda x: -A
    }
    
    result = minimize(
        costFnc,
        x0,
        method='SLSQP',
        jac=gradient,
        constraints=constraints,
        options=options
    )
    
    x = result.x
    fval = result.fun
    
    sepPlane = x / np.linalg.norm(x[0:3])
    
    return sepPlane, fval
    
def getSmoothAngleSpread(p: int, t_c: int, p_c: int, winSize: int = 3) -> np.ndarray:
    global angleSprd, thEdge, phEdge
    
    nth = 0.15
    a = np.array([0.9975, 0.0023, 0.0709])
    
    N = winSize // 2
    Nt = len(thEdge)
    Np = len(phEdge)
    
    tmin = t_c - N
    tmax = t_c + N
    tI = np.arange(tmin, tmax + 1)
    tI[tI < 0] += Nt
    tI[tI >= Nt] -= Nt

    pmin = p_c - N
    pmax = p_c + N
    pI = np.arange(pmin, pmax + 1)
    pI[pI < 0] += Np
    pI[pI >= Np] -= Np
    
    AS = np.squeeze(angleSprd[p][t_c, p_c, :, :]).copy()
    
    for i in range(len(tI)):
        for j in range(len(pI)):
            t_i = int(tI[i])
            p_j = int(pI[j])
            AS = AS | np.squeeze(angleSprd[p][t_i, p_j, :, :])

    return AS

mat = loadmat(STATICPOSE_PATH)  # Update path as needed
a = mat['a']
di = mat['di']

bone_names = [
    "pelvis",       # 0
    "thigh_r",      # 1
    "calf_r",       # 2
    "foot_r",       # 3
    "thigh_l",      # 4
    "calf_l",       # 5
    "foot_l",       # 6
    "spine_04",     # 7
    "neck_01",      # 8
    "head",         # 9
    "upperarm_l",   # 10  
    "lowerarm_l",   # 11
    "hand_l",       # 12
    "upperarm_r",   # 13 
    "lowerarm_r",   # 14 
    "hand_r",       # 15
]

edges: List[Tuple[int,int]] = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9),        # spine/head
    (8,10), (10,11), (11,12),      # left arm
    (8,13), (13,14), (14,15),      # right arm
]



name_to_idx = {n:i for i,n in enumerate(bone_names)}
mat = loadmat(STATICPOSE_PATH)
a = mat['a']
di = mat['di']

def parent_of(child_idx: int) -> int | None:
    """
    Return parent joint index for the given child joint index
    """
    for p, c in edges:
        if c == child_idx:
            return p
    return None

def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

def getNormal(x1: np.ndarray, a: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, bool]:
    nth = 1e-4
    
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:
        n = np.cross(x, x1)
        flg = True
    else:
        n = np.cross(a, x)
        flg = False
    n = n / np.linalg.norm(n)
    
    return n, flg

# in the child the upper arm refers to elbow
# we only get the possible rotation of the childs array so none of shoulders or hips etc
# 1: belly, 
# 2: R-shldr, 
# 3: R-elbow, 
# 4: R-wrist, 
# 5: L-shldr, 
# 6: L-elbow, 
# 7: L-wrist, 
# 8: Neck, 
# 9: face, 
# 10: R-hip, 
# 11: R-knee, 
# 12: R-ankle,   
# 13: R-foot, 
# 14: L-hip, 
# 15: L-knee, 
# 16: L-ankle, 
# 17: L-foot 

# they have 17 bones we have 16 not all of them are exact matches but work well enough
bones = {
1: 0,       # belly     --> pelvis
2: 13,      # R-shldr   --> upperarm_r
3: 14,      # R-elbow   --> lowerarm_r
4: 15,      # R-wrist   --> hand_r
5: 10,      # L-shldr   --> upperarm_l
6: 11,      # L-elbow   --> lowerarm_l
7: 12,      # L-wrist   --> hand_l
8: 8,       # Neck      --> neck_01
9: 9,       # face      --> head
10: 1,      # R-hip     --> thigh_r
11: 2,      # R-knee    --> calf_r
12: 3,      # R-ankle   --> foot_r
13: np.nan, # R-foot    --> None
14: 4,      # L-hip     --> thigh_l
15: 5,      # L-knee    --> calf_l
16: 6,      # L-ankle   --> foot_l
17: np.nan, # L-foot    --> None

99: 7       # None      --> spine_04
}
# parents
# 2: R-shldr, 3: R-elbow, 5: L-shldr, 6: L-elbow, 2: R-shldr, 9: face, 10: R-hip, 11: R-knee, 13: R-foot, 14: L-hip, 15: L-knee

# childs
# 3: R-elbow, 4: R-wrist, 6: L-elbow, 7: L-wrist, 8: Neck, 10: R-hip, 11: R-knee, 12: R-ankle, 14: L-hip, 15: L-knee, 16: L-ankle, 
# idx corresponds with both
prnts = np.array([2, 3, 5, 6, 2, 9, 10, 11, 13, 14, 15])
chlds = np.array([3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16])
nprts = len(chlds);  

def global2local(dS: np.ndarray, typ: int = 1) -> np.ndarray:
    global prnts, chlds, chldsT, di, a
    
    nprts = len(chlds)
    
    back = dS[name_to_idx['spine_04']]
    sh_l = dS[name_to_idx['upperarm_l']]
    sh_r = dS[name_to_idx['upperarm_r']]
    shldr = sh_l - sh_r
    hip_l = dS[name_to_idx['thigh_l']]
    hip_r = dS[name_to_idx['thigh_r']]
    hip = hip_l - hip_r
    
    upper = {
        name_to_idx['upperarm_l'],
        name_to_idx['upperarm_r'],
        name_to_idx['head']
    }
    lower = {
        name_to_idx['thigh_l'],
        name_to_idx['thigh_r']
    }
    
    dSl = dS.copy()
    
    for i in chlds:
        if i in upper:
            u = _unit(shldr)
            v = _unit(back)
        elif i in lower:
            u = _unit(hip)
            v = _unit(back)
        else:
            u = _unit(parent_of(i))
            # di looks like the 
            # [v] = getNormal(R*di(:, prnts(i)), R*a, u);
            v = getNormal(R @ di[:, parent_of[i]], R @ a, u)
        
        w = _unit(np.cross(u, v))
        
        if typ == 1:
            R = gramschmidt(np.column_stack([u, v, w]))
        elif typ == 2:
            R = gramschmidt(np.column_stack([w, u, v]))
        elif typ == 3:
            R = gramschmidt(np.column_stack([v, w, u]))
        elif typ == 4:
            R = gramschmidt(np.column_stack([u, -w, v]))
        elif typ == 5:
            R = gramschmidt(np.column_stack([-w, v, u]))
        else:
            R = gramschmidt(np.column_stack([v, u, -w]))
            
        dSl[:, chlds[i]] = R.T @ dS[:, chlds[i]]
    
    return dSl


def gramschmidt(A: np.ndarray) -> np.ndarray | None:
    n = A.shape[0]
    if A.shape[1] != n:
        return None
    
    B = np.zeros((n,n))
    B[:, 0] = (1 / np.linalg.norm(A[:,0])) * A[:,0]
    
    for i in range(1,n):
        v = A[:,i].copy()
        U = B[:,0:i]
        pc = U.T @ v
        p = U @ pc
        v = v - p
        if np.linalg.norm(v) < np.finfo(float).eps:
            return None
        v = _unit(v)
        B[:, i] = v
        
    return B
    
def jointAngles2Cart3(dS, angles, typ, p = np.nan):
    pass

def minBoundingBox(X):
    pass

def pointFromPlane2Sphere(Xh, v, ri):
    pass
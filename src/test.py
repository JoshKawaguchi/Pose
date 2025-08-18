import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.linalg import svd
import cv2
import json
import os
from glob import glob
from typing import List, Tuple, Dict, Optional, Any
from utils import *
from scipy.io import loadmat

def main():
    print(1 * np.random.rand(1,3))
    mat = loadmat(STATICPOSE_PATH)
    a = mat['a']
    di = mat['di']
    print(di.shape)
    print(di)
    print('-----------------')
    print(mat)
    
    mat2 = loadmat('..\metadata\skeleton_17Pts.mat')
    print('-----------------')
    print(mat2)

if __name__ == "__main__":
    main()
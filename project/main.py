#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 06:16:39 2021

@author: xuzhaoyang
"""
import TPS
import method2 as m2
import numpy as np
import igl
from shapely.geometry import Polygon

back1_2d = np.loadtxt('coodinates_2d_back1.txt')
back1_3d = np.loadtxt('coodinates_3d_back1.txt')

back2_2d = np.loadtxt('coodinates_2d_back2.txt')
back2_3d = np.loadtxt('coodinates_3d_back2.txt')

v, f = igl.read_triangle_mesh("Graphosoma.obj")
v1, f1 = igl.read_triangle_mesh("back1.obj")
v2, f2 = igl.read_triangle_mesh("back2.obj")


_,_,_,_,_,simplices1,ind1=TPS.knearstraingle(back1_2d,back1_3d,30,1,v)
_,_,_,_,_,simplices2,ind2=TPS.knearstraingle(back2_2d,back2_3d,30,1,v)


v11,f11 = m2.Processing_3dmesh(v1,f1)
uv1 = m2.Flatten(v11,f11)

v22,f22 = m2.Processing_3dmesh(v2,f2)
uv2 = m2.Flatten(v22,f22)


grid1 = m2.plot_2d(back1_2d,simplices1,10)
grid2 = m2.plot_2d(back2_2d,simplices2,10)


region2d_back1 = back1_2d[ind1]
region2d_back2 = back2_2d[ind2]

polygon1 = Polygon(region2d_back1)
polygon2 = Polygon(region2d_back2)

m2.over_region(back1_2d,simplices1,30,polygon1,v11,f11,uv1,back1_3d,v2,f2)
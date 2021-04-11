#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:23:34 2021

@author: ryan
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import pandas as pd
from collections import defaultdict
from numpy.linalg import norm, inv
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import string


def top_k_closest(k, name, v_mat,movie_index,index_movie):
  result = []
  result_distances = []
  result_idx = []
  dists = []
  
  idx = movie_index[name]
  vidx = v_mat[idx]

  for i in range(len(v_mat)):
    if i != idx:
      euc_distance = euclidean(vidx, v_mat[i])
      dists.append((euc_distance, i))
    
  dists = sorted(dists, key=lambda x: x[0])

  for i in range(k):
    dist_idx, idx = dists[i][0], dists[i][1]
    name = index_movie[idx]
    result.append(name)
    result_distances.append(dist_idx)
    result_idx.append(idx)
  
  return result, result_distances, result_idx

def recommend2user(index_user,index_movie, u_mat,v_mat):
    pred_ratings = np.matmul(u_mat,v_mat.T)
    top5movies = np.argsort(pred_ratings)[:,-5:]
    user_recommendation = []
    for i,movies in enumerate(top5movies):
        user_name = index_user[i]
        for movie in movies:
            user_recommendation.append([user_name,index_movie[movie]])
    user_recommendation = pd.DataFrame(user_recommendation,columns = ["user_id","movie_id"])
    return user_recommendation
    
    
    
    

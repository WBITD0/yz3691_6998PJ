#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:42:12 2021

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
def initialise_umat(num_users):
  u_mat=[]
  for i in range(num_users):
    u_mat.append(np.random.multivariate_normal(mean=np.zeros(10), cov=np.identity(10), size=1)[0])
  u_mat= np.asarray(u_mat)
  return u_mat

def initialise_vmat(num_movies):
  v_mat=[]
  for i in range(num_movies):
    v_mat.append(np.random.multivariate_normal(mean=np.zeros(10), cov=np.identity(10), size=1)[0])
  v_mat= np.asarray(v_mat)
  return v_mat

def f_UV(user_index,movie_index,u_mat, v_mat, ratings_array, lambda_u=0.25, lambda_v=0.25, bias=False, bu=None, bv=None, mu=None):
  term_1 = 0
  num_users = len(user_index)
  num_movies = len(movie_index)
  for i in range(0, len(ratings_array)):
    idx_u = int(ratings_array[i][0]) 
    idx_m = int(ratings_array[i][1]) 
    temp_term = ratings_array[i][2] - u_mat[user_index[idx_u]].dot(v_mat[movie_index[idx_m]])
    if bias:
      temp_term -= bu[user_index[idx_u]] + bv[movie_index[idx_m]] + mu
    term_1 += temp_term**2

  term_2 = 0
  for u in range(num_users):
    term_2 += norm(u_mat[u])**2
    if bias:
      term_2 += bu[u] ** 2

  term_3 = 0
  for m in range(num_movies):
    term_3 += norm(v_mat[m])**2
    if bias:
      term_3 += bv[m]**2
 
  return term_1 + lambda_u*term_2 + lambda_v*term_3

def update_user_location(u_id, v, movie_index,dicts, lambda_u=0.25, bias=False, bv=None, mu=None):
  user_dict = dicts[0]
  train_ratings_dict = dicts[2]
  L = v.shape[1]
  rt_term = np.zeros(L)
  movie_set = user_dict[u_id]
  R = np.array([train_ratings_dict[(u_id, v_id)] for v_id in movie_set])
  R = R[:,np.newaxis]
  movie_index_set = [int(movie_index[m]) for m in movie_set]

  m = lambda_u * np.identity(L)
  m = m + v[movie_index_set].T.dot(v[movie_index_set])
  if bias:
    d = v[movie_index_set].T.dot(R - mu - bv[movie_index_set,np.newaxis])
  else:
    d = v[movie_index_set].T.dot(R)
  result = inv(m).dot(d)
  return result[:,0]

def update_object_location(v_id, u, user_index,dicts,lambda_v=0.25, bias=False, bu=None, mu=None):
  movie_dict = dicts[1]
  train_ratings_dict = dicts[2]
  L = u.shape[1]
  rt_term = np.zeros(L)
  user_set = movie_dict[v_id]
  R = np.array([train_ratings_dict[(u_id, v_id)] for u_id in user_set])
  R = R[:,np.newaxis]
  user_index_set = [int(user_index[u]) for u in user_set]

  m = lambda_v * np.identity(L)
  m = m + u[user_index_set].T.dot(u[user_index_set])

  if bias:
    d = u[user_index_set].T.dot(R - mu - bu[user_index_set,np.newaxis])
  else:
    d = u[user_index_set].T.dot(R)
  result = inv(m).dot(d)
  return result[:,0]

#goes from i=1 to N1 (number of user locations), updating each row
def update_u(user_index,movie_index,dicts,u_mat, v_mat, lambda_u=0.25, bias=False, bv=None, mu=None):
  for id in user_index.keys():
    index = user_index[id]
    u_mat[index] = update_user_location(id, v_mat, movie_index,dicts,lambda_u, bias, bv, mu)
  return u_mat

def update_v(user_index,movie_index,dicts,u_mat, v_mat, lambda_v=0.25, bias=False, bu=None, mu=None):
  for id in movie_index.keys():
    index = movie_index[id]
    v_mat[index] = update_object_location(id, u_mat,user_index,dicts, lambda_v, bias, bu, mu)
  return v_mat

def run_MAP_coor_ascent(train_data,lambda_u,lambda_v,num_of_iterations=20, bias=False):
  userId = np.unique(np.asarray(train_data['userId']))
  user_index = defaultdict(int)
  num_users = 0
  for u in userId:
    user_index[u] = num_users
    num_users += 1

# movie_index: dict() movie id -> movie index
  movieId = np.unique(np.asarray(train_data['movieId']))
  movie_index = defaultdict(int)
  num_movies = 0
  train_array = np.asarray(train_data)
  for m in movieId:
    movie_index[m] = num_movies
    num_movies += 1

# user_dict : dict() user id -> movie id which user gave rating
  user_dict = defaultdict(list)
    # movie_dict : dict() movie id -> user id which user gave rating
  movie_dict = defaultdict(list)
    # rating_dict : dict() (user id, movie id) -> rating
  train_ratings_dict = defaultdict(float)
    
  for r in train_array:
    if r[0] in user_dict:
        user_dict[int(r[0])].append(int(r[1]))
    else:
        user_dict[int(r[0])] = [int(r[1])]
    if r[1] in movie_dict:
        movie_dict[int(r[1])].append(int(r[0]))
    else:
        movie_dict[int(r[1])] = [int(r[0])]
    
    train_ratings_dict[(int(r[0]), int(r[1]))]= r[2]
  dicts = [user_dict,movie_dict,train_ratings_dict]
  u = initialise_umat(num_users)
  v = initialise_vmat(num_movies)
  F_values=[]
  Train_rmse = []
  bu = None
  bv = None
  mu = None
  Converge = False
  last_train_rmse = 100
  if bias:
    bu = np.zeros(num_users)
    bv = np.zeros(num_movies)
    mu = np.mean(train_data['rating'])

  for i in range(num_of_iterations):
    if i % (num_of_iterations//min(10, num_of_iterations)) == 0:
      print('.',end='')
    
    if bias:
      v = np.c_[np.ones(num_movies),v]
      u = np.c_[bu,u]
    u = update_u(user_index,movie_index,dicts,u,v,lambda_u,bias,bv,mu)
    if bias:
      bu = u[:,0]
      u = u[:,1:]
      v = v[:,1:]
    if bias:
      v = np.c_[bv,v]
      u = np.c_[np.ones(num_users) ,u]
    v = update_v(user_index,movie_index,dicts,u,v,lambda_v,bias,bu,mu)
    if bias:
      bv = v[:,0]
      v = v[:,1:]
      u = u[:,1:]

    f = f_UV(user_index,movie_index,u,v,train_array,lambda_u,lambda_v,bias,bu,bv,mu)
    F_values.append(f)
    # print(f)

    train_predict = predict_ratings(train_array, user_index,movie_index,u, v, bias, bu, bv, mu)
    train_predict[train_predict>5] = 5
    train_predict[train_predict<0] = 0
    train_rmse = math.sqrt(mean_squared_error(np.asarray(train_data['rating']), train_predict))
    Train_rmse.append(train_rmse)
    # print(train_rmse)
    # print(test_rmse)

    if abs(last_train_rmse - train_rmse) <= 0.001:
      Converge = True
      break
    last_train_rmse = train_rmse
  if Converge:
    print('Algorithm convergence on training data for %d steps!' % (i+1))
  else:
    print('Maximum number of iterations reached!')
  return u, v, F_values, Train_rmse

def predict_ratings(data, user_index,movie_index,u_mat, v_mat, bias=False, bu=None, bv=None, mu=None):
  preds=[]
  for i in range(len(data)):
    # print(data)
    u_idx = user_index[int(data[i][0])]
    v_idx = movie_index[int(data[i][1])]
    u_vect = u_mat[u_idx]
    v_vect = v_mat[v_idx]
    
    if bias:
      bu_num = bu[u_idx]
      bv_num = bv[v_idx]
      m = u_vect.dot(v_vect) + mu + bu_num + bv_num
    else:
      m = u_vect.dot(v_vect)
    rating = np.random.normal(m, 0.25, 1)[0]

    preds.append(rating)                         
  return np.array(preds)

def run(train_data,lambda_u,lambda_v,bias=False):
  #Looping over 10 runs
  results_dict = {}
  u_mat, v_mat, F_values, Train_rmse = run_MAP_coor_ascent(train_data,lambda_u,lambda_v,50, bias)
    
  results_dict["Train_rmse"]= Train_rmse
  results_dict["F_values"]= F_values
  results_dict["u"]= u_mat
  results_dict["v"]= v_mat


  return results_dict


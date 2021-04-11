#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:26:51 2021

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
import boto3
from ALS import *
import recommendation
import tqdm
credentials = boto3.Session().get_credentials()
access_key = credentials.access_key
secret_key = credentials.secret_key
s3 = boto3.client("s3")

obj = s3.get_object(Bucket= 'yz3691projectstorage', Key= 'ratings.csv') 
ratings_data = pd.read_csv(obj['Body'])
train_data =ratings_data
train_array = np.asarray(train_data)

lambda_u = 10
lambda_v = 10
result_dict = run(train_data,lambda_u,lambda_v, bias=False)
u_mat = result_dict['u']
v_mat = result_dict['v']
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

index_movie = dict((v,k) for k, v in movie_index.items())
index_user = dict((v,k) for k, v in user_index.items())
movie2movie = []
for name in tqdm.tqdm(movie_index.keys()):
    result = recommendation.top_k_closest(5,name,v_mat,movie_index,index_movie)
    recom_movies = result[0]
    for recom_movie in recom_movies:
        movie2movie.append([name,recom_movie])
movie2movie = pd.DataFrame(movie2movie,columns = ['movie','recommended_movie'])
    
movie2movie.to_csv("similar_movies.csv")
with open("similar_movies.csv", "rb") as f:
    s3.upload_fileobj(f, "yz3691projectstorage", "similar_movies.csv")

recommended_movies = recommendation.recommend2user(index_user,index_movie,u_mat,v_mat)
recommended_movies.to_csv("recommended_movies.csv")
with open("recommended_movies.csv", "rb") as f:
    s3.upload_fileobj(f, "yz3691projectstorage", "recommended_movies.csv")

u_mat_pd = pd.DataFrame(u_mat,columns = ["k"+str(i) for i in range(10)])
v_mat_pd = pd.DataFrame(v_mat,columns = ["k"+str(i) for i in range(10)])
u_mat_pd.to_csv("u_mat.csv")
with open("u_mat.csv", "rb") as f:
    s3.upload_fileobj(f, "yz3691projectstorage", "u_mat.csv")

v_mat_pd.to_csv("v_mat.csv")
with open("v_mat.csv", "rb") as f:
    s3.upload_fileobj(f, "yz3691projectstorage", "v_mat.csv")



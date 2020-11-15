# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:06:14 2020

@author: mansi
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = ["London Paris London", "Paris London Paris"]

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)
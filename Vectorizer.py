#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy as np
import math
from collections import Counter

class Vectorizer:
    def __init__(self, min_word_length=3, max_df=1.0, min_df=0.0):
        self.min_word_length = min_word_length
        self.max_df=max_df
        self.min_df=min_df
        self.term_df_dict = {}

    def fit(self, raw_documents):
        self.document_count = len(raw_documents)
        for raw_document in raw_documents:
            words = np.unique(raw_document.split(' '))
            for word in words:
                if len(word) < self.min_word_length:
                    continue
                try:
                    self.term_df_dict[word] += 1
                except KeyError:
                    self.term_df_dict[word] = 1

        for word in self.term_df_dict.keys():
            self.term_df_dict[word] = float(self.term_df_dict[word]) / float(self.document_count)
            if self.min_df <= self.term_df_dict[word] and self.term_df_dict[word] <= self.max_df:
                continue
            else:
                del self.term_df_dict[word]
        self.vocabulary = self.term_df_dict.keys()

    def _transform(self, raw_document, method):
 
        words = raw_document.split(" ")
        if method == "existance":
            counts = Counter(words)
            return np.vectorize(lambda x: 1 if x in counts else 0)(self.term_df_dict.keys())
        elif method == "count":
            counts = Counter(words)
            return np.vectorize(lambda x: counts[x] if x in counts else 0)(self.term_df_dict.keys())
        elif method == "tf-idf":
            counts = Counter(words)
            tf = np.vectorize(lambda x: counts[x] if x in counts else 0)(self.term_df_dict.keys())
            
            idf = np.vectorize(lambda lbl: math.log( (1+self.document_count) / (1+self.term_df_dict[lbl]*self.document_count if lbl in self.term_df_dict else 0) ) + 1)(self.term_df_dict.keys()) 
            result = np.array(tf*idf)
            denom = np.sqrt(np.sum(np.square(result)))
            if denom == 0:
                return result
            return result / denom
            
    def transform(self, raw_documents, method="tf-idf"):

        feature_vectors = []
        for raw_document in raw_documents:
            feature_vectors.append(self._transform(raw_document, method))
        return np.array(feature_vectors)


    def fit_transform(self, raw_documents, method="tf-idf"):
        self.fit(raw_documents)
        return self.transform(raw_documents, method)

    def get_feature_names(self):
        try:
            self.vocabulary
        except AttributeError:
            print "Please first fit the model."
            return []
        return self.vocabulary

    def get_term_dfs(self):
        return sorted(self.term_df_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)



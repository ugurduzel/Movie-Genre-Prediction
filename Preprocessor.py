#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import os
from nltk.corpus import stopwords
import codecs
import errno
import string

class Preprocessor:
    def __init__(self, dataset_directory="Dataset", processed_dataset_directory= "ProcessedDataset"):
        self.dataset_directory = dataset_directory
        self.processed_dataset_directory=processed_dataset_directory
        nltk.download("stopwords")
        nltk.download("punkt")
        self.stop_words = set(stopwords.words('english'))

    def _remove_puncs_numbers_stop_words(self, tokens):
        new_tokens = list()
        puncs_and_nums = string.punctuation
        for token in tokens:
            array = []
            for char in token:
                if char in "0123456789":
                    array = []
                    break
                if char not in puncs_and_nums:
                    array.append(char)
            temp = "".join(array)
            if temp != "":
                new_tokens.append(temp)

        remaining_tokens = []
        for token in new_tokens:
            if token not in self.stop_words:
                remaining_tokens.append(token)

        return remaining_tokens


    def _tokenize(self, sentence):
        return nltk.tokenize.word_tokenize(sentence.lower())

    def _stem(self, tokens):
        stems = []
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        for token in tokens:
            token = stemmer.stem(token)
            if token != "":
                stems.append(token)
        return stems

    def preprocess_document(self, document):
        tokens = self._tokenize(document)
        tokens = self._remove_puncs_numbers_stop_words(tokens)
        processed_string = str()
        for temp in self._stem(tokens):
            processed_string += temp
            processed_string += " "
        return processed_string

    def preprocess(self):
        for root, dirs, files in os.walk(self.dataset_directory):
            if os.path.basename(root) != self.dataset_directory:
                print "Processing", root, "directory."
                dest_dir = self.processed_dataset_directory+"/"+root.lstrip(self.dataset_directory+"/")
                if not os.path.exists(dest_dir):
                    try:
                        os.makedirs(dest_dir)
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                for file in files:
                    file_path = root + "/" + file
                    with codecs.open(file_path, "r", "ISO-8859-1") as f:
                        data = f.read().replace("\n", " ")
                    processed_data = self.preprocess_document(data)
                    output_file_path = dest_dir + "/" + file
                    with codecs.open(output_file_path, "w", "ISO-8859-1") as o:
                        o.write(processed_data)





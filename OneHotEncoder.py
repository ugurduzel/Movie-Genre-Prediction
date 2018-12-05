import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.tags=[]

    def fit(self,X):
        self.tags.extend(list(np.unique(X)))

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        results = list()
        for label in X:
            results.append(np.vectorize(lambda x: 1 if x == label else 0)(self.tags))
        return np.array(results)

    def get_feature_names(self):
        return self.tags

    def decode(self, one_hot_vector):
        return self.tags[np.argmax(one_hot_vector)] 


from OneHotEncoder import OneHotEncoder
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import functools
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

top3_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=3)

top3_acc.__name__ = 'top3_acc'


top2_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=2)

top2_acc.__name__ = 'top2_acc'

metrics = ['categorical_accuracy', top2_acc, top3_acc]

class ExperimentSuite:
    def __init__(self, dataset_directory = "ProcessedDataset", train_directory = "TrainDataset", test_directory = "TestDataset"):
        self.dataset_directory = dataset_directory
        self.train_directory = train_directory
        self.test_directory = test_directory

        self.train_contents = []
        self.train_labels = []
        self.test_contents = []
        self.test_labels = []
        self.read_data()

        self.enc = OneHotEncoder()
        self.enc.fit(self.train_labels)
        self.train_y = self.enc.transform(self.train_labels)
        self.test_y = self.enc.transform(self.test_labels)

    def train_model(self,layers, tbCallBack, train_x, train_y, test_x, test_y, loss, activation, epoch):
        self.number_of_features = train_x.shape[1]
        model = tf.keras.models.Sequential()

        first = True
        for layer in layers:
            if first == True:
                model.add(tf.keras.layers.Dense(layer, activation = activation, input_dim=self.number_of_features))
                first = False
            else:
                model.add(tf.keras.layers.Dense(layer, activation = activation))        
        model.add(tf.keras.layers.Dense(len(self.enc.tags), activation = 'softmax'))

        model.compile(optimizer=tf.train.AdamOptimizer(),loss=loss, metrics=metrics) 
        model.fit(train_x, train_y, epochs=epoch, verbose = 0)

        evaluation = model.evaluate(test_x, test_y, verbose=0)
        print "Activation function:", activation.__name__
        print "Loss function:", loss
        print "Layers:",layers
        print "Test set evaluation:",zip(model.metrics_names, evaluation)
        Y_pred = model.predict(test_x)
        y_pred = np.argmax(Y_pred, axis=1)
        print 'Confusion Matrix'
        print self.enc.get_feature_names()
        print(confusion_matrix(test_y.argmax(axis=1), y_pred))
        print "\n"
        return evaluation

    def read_data(self):
        for root, dirs, files in os.walk(self.dataset_directory + "/" + self.train_directory):
            for file in files:
                self.train_labels.append(os.path.basename(root))
                with open(root + "/" + file, "r") as i:
                    self.train_contents.append(i.read())

        for root, dirs, files in os.walk(self.dataset_directory + "/" + self.test_directory):
            for file in files:
                self.test_labels.append(os.path.basename(root))
                with open(root + "/" + file, "r") as i:
                    self.test_contents.append(i.read())

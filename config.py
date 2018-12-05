from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

__C.DATASET_PATH = "ProcessedDataset"
# It should be in the from -> ./PrcoessedDataset/TrainDataset and the same for the test_set
__C.TRAINING_SET = "TrainDataset"
__C.TEST_SET = "TestDataset"

__C.MIN_DF_VECTORIZER = 0.25
__C.MAX_DF_VECTORIZER = 0.97


__C.TRAIN = edict()

__C.TRAIN.EPOCH = 500
__C.TRAIN.LAYERS = (512,256)
__C.TRAIN.ACTIVATION = tf.nn.relu
__C.TRAIN.LOSS = "categorical_hinge"
# 'existence' / 'count' / 'tf-idf'
__C.TRAIN.VECTORIZATION_METHOD = 'count' 


__C.TEST = edict()
# 'existence' / 'count' / 'tf-idf'
__C.TEST.VECTORIZATION_METHOD = 'count'
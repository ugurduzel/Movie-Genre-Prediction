from ExperimentSuite import ExperimentSuite
from Vectorizer import Vectorizer
from Preprocessor import Preprocessor
import tensorflow as tf
import numpy as np
from config import cfg

if __name__ == "__main__":    
    es = ExperimentSuite(dataset_directory = cfg.DATASET_PATH, train_directory = cfg.TRAINING_SET, test_directory = cfg.TEST_SET)

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir="logs", 
    	histogram_freq=0, write_graph=True, write_images=True)

    vectorizer = Vectorizer(max_df = cfg.MAX_DF_VECTORIZER, min_df = cfg.MIN_DF_VECTORIZER)
    vectorizer.fit(es.train_contents) 

    train_x = vectorizer.transform(es.train_contents, cfg.TRAIN.VECTORIZATION_METHOD)
    test_x  = vectorizer.transform(es.test_contents, cfg.TRAIN.VECTORIZATION_METHOD)
    result = es.train_model((512,256,), tbCallBack, train_x, es.train_y, test_x, es.test_y,
            loss=cfg.TRAIN.LOSS, activation=cfg.TRAIN.ACTIVATION, epoch=cfg.TRAIN.EPOCH)
    print result



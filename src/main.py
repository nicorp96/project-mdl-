#!/usr/bin/env python3
import tensorflow as tf
from utils.data_generator import DataGenerator
from models.model import MyModel, MyTrainer
import sys
# CONSTANTS
IMG_WIDTH = 28
IMG_HEIGHT = 28
BATCH_SIZE = 128
DATA_PATH = "./data"

def main():
    print(sys.version)
    print("/n Human Detection Project MDL /n")
    try:
        dg = DataGenerator(DATA_PATH,IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE)
        train_ds, test_ds = dg.data_generator()
        dg.shape_of_mini_batches(train_ds,1)
        model = MyModel()
        config = {
            'n_epochs':     5, 
            'batch_size':   BATCH_SIZE, 
            'opt':          tf.optimizers.Adam(learning_rate=0.001),
            'train_ds':     train_ds,    
            'test_ds':      test_ds,
            'report_steps': 100,
            'test_steps':   250,
        }
        tr = MyTrainer(model,config)
        model.summary()
        hist = tr.train()
        pass
    except Exception as exp:
        print(exp)
        pass
    print("/n end /n")
    

if __name__ == "__main__":
    main()

import pymongo
import numpy as np



EXPERIMENT_NAME = 'EXP_4'
CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'

TRAINING_WINDOW      = 3
CONTEXT_DIMENSION    = 64
LEANING_RATE         = 1
DROPOUT              = 0.1
CONTEXT_DECAY        = 1/np.exp(1)
CONTRASTIVE_WEIGHT   = 0.01
NEGATIVE_SAMPLE_SIZE = 10


myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb       = myclient["mydatabase"]
collection = mydb.random_negative_sampling



'''
Experiment details:
randomly sample a few sibling connections rather than selecting all (to reduce run time per iteration)

To do :
comapar graph against train_1
'''

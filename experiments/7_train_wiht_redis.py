import numpy as np




EXPERIMENT_NAME = 'EXP_7'
CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'

TRAINING_WINDOW      = 8
CONTEXT_DIMENSION    = 64
LEANING_RATE         = 1
DROPOUT              = 0.1
CONTEXT_DECAY        = np.exp(-1)
CONTRASTIVE_WEIGHT   = 0.01
NEGATIVE_SAMPLE_SIZE = int(np.exp(1) * TRAINING_WINDOW)


DB = 'REDIS'


if DB == 'MONGO':
    import pymongo
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient["mydatabase"]
    collection = mydb.neighbour_aware_context_initilization_train_window_8

if DB == 'REDIS':
    import redis
    collection = redis.Redis(db=3)



'''
Experiment details:
used redis as db for graph, 
training took too long (more than 3 hrs)  

To do :
creating a set to store keys to speed up scan operation
'''


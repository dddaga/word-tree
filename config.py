import numpy as np




EXPERIMENT_NAME = 'EXP_14'
CORPUS_PATH = '/home/dddhiraj/Documents/stuff/data/wiki_en.txt'
TRAINING_WINDOW      = 4
CONTEXT_DIMENSION    = 32
LEANING_RATE         = 1
DROPOUT              = 1
CONTEXT_DECAY        = 1 - TRAINING_WINDOW ** -0.5
CONTRASTIVE_WEIGHT   = 1#0.1
NEGATIVE_SAMPLE_SIZE = TRAINING_WINDOW ** 2
CONEXT_INERTIA       = np.sqrt(CONTEXT_DIMENSION)


THREADS  = 6
CHUNK_SIZE = 256


DB = 'MONGO'


if DB == 'MONGO':
    import pymongo
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient["mydatabase"]
    collection = mydb.train_5#neighbour_aware_context_initilization_train_window_8
    collection.create_index('word')
if DB == 'REDIS':
    import redis
    collection = redis.Redis(db=5) #11
    key_collection= redis.Redis(db=6) #12
    #import redisai
    # collection = redisai.Client(db=14)
    # key_collection = redisai.Client(db=15)



'''
Experiment details:
Trained on wiki data with 51 million words.

Decoupled context and update vector
added context inertia

change in loss funciton(v3 policy)



number of connections created pre min :

1 thread  = 2252
3 thread =  2374
5 thread =  2362

1 process =  1802
3 process = 5519
5 process = 7336
7 process =  8647
9 process = 9008

'''


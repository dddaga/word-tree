import numpy as np




EXPERIMENT_NAME = 'EXP_12'
CORPUS_PATH = '/home/dddhiraj/Documents/stuff/data/wiki_en.txt'
TRAINING_WINDOW      = 5
CONTEXT_DIMENSION    = 64
LEANING_RATE         = 1
DROPOUT              = 0.05
CONTEXT_DECAY        =  1 - TRAINING_WINDOW ** -0.5
CONTRASTIVE_WEIGHT   = 1#0.1
NEGATIVE_SAMPLE_SIZE = TRAINING_WINDOW ** 2
CONEXT_INERTIA       =  np.sqrt(TRAINING_WINDOW)


THREADS  = 6
CHUNK_SIZE = 5000


DB = 'REDIS'


if DB == 'MONGO':
    import pymongo
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient["mydatabase"]
    collection = mydb.train_1#neighbour_aware_context_initilization_train_window_8

if DB == 'REDIS':
    import redis
    collection = redis.Redis(db=1) #11
    key_collection= redis.Redis(db=2) #12
    #import redisai
    # collection = redisai.Client(db=14)
    # key_collection = redisai.Client(db=15)



'''
Experiment details:
Trained on wiki data with 51 million words.
'''


import numpy as np




EXPERIMENT_NAME = 'EXP_11'
CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'

TRAINING_WINDOW      = 8
CONTEXT_DIMENSION    = 64
LEANING_RATE         = 1
DROPOUT              = 0.25
CONTEXT_DECAY        = 0.5
CONTRASTIVE_WEIGHT   = 0.1
NEGATIVE_SAMPLE_SIZE =  100#int(np.exp(1) * TRAINING_WINDOW)


DB = 'MONGO'


if DB == 'MONGO':
    import pymongo
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient["mydatabase"]
    collection = mydb.train_1#neighbour_aware_context_initilization_train_window_8

if DB == 'REDIS':
    import redis
    collection = redis.Redis(db=9)
    key_collection= redis.Redis(db=10)


'''
Experiment details:
Incread CONTRASTIVE_WEIGHT to 0.1 and drop out to 0.25

primary loss is tensor(3.8142, dtype=torch.float64, grad_fn=<SumBackward0>)
contrastive loss is  tensor(0.4982, dtype=torch.float64, grad_fn=<DivBackward0>)
training completed in 5282.486120939255


To do :
increase contrastive weight, increase drop out
'''


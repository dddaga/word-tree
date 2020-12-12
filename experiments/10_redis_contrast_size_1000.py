import numpy as np




EXPERIMENT_NAME = 'EXP_10'
CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'

TRAINING_WINDOW      = 8
CONTEXT_DIMENSION    = 64
LEANING_RATE         = 1
DROPOUT              = 0.1
CONTEXT_DECAY        = np.exp(-1)
CONTRASTIVE_WEIGHT   = 0.01
NEGATIVE_SAMPLE_SIZE =  1000#int(np.exp(1) * TRAINING_WINDOW)


DB = 'REDIS'


if DB == 'MONGO':
    import pymongo
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient["mydatabase"]
    collection = mydb.neighbour_aware_context_initilization_train_window_8

if DB == 'REDIS':
    import redis
    collection = redis.Redis(db=7)
    key_collection= redis.Redis(db=8)


'''
Experiment details:
increased windoe
contrastive loss is  0.0
primary loss is tensor(3.9504, dtype=torch.float64, grad_fn=<SumBackward0>)
contrastive loss is  tensor(0.2359, dtype=torch.float64, grad_fn=<DivBackward0>)
training completed in 6602.483899116516

Observation :
based on activation scorces duirng graph traveral it seems most of the condtex ventors are pointing in same direction


To do :
increase contrastive weight, increase drop out
'''


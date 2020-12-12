import numpy as np




EXPERIMENT_NAME = 'EXP_9'
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
    collection = redis.Redis(db=5)
    key_collection= redis.Redis(db=6)



'''
Experiment details:

primary loss is tensor(3.8691, dtype=torch.float64, grad_fn=<SumBackward0>)
contrastive loss is  tensor(0.0367, dtype=torch.float64, grad_fn=<DivBackward0>)
training completed in 4417.234423160553


To do :
comapar with graph against exp_4 and exp_5
'''


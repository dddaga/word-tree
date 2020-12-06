import pymongo

TRAINING_WINDOW    = 7
CONTEXT_DIMENSIN   = 256
CONTEXT_DECAY      = 0.5
CONTRASTIVE_WEIGHT = 0.1
LEANING_RATE       = 1
DROPOUT            = 0.1


myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb       = myclient["mydatabase"]
collection = mydb.train_1


CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'


EXPERIMENT_NAME = 'EXP_2'



'''
Experiment details:





'''


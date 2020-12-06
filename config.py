import pymongo



EXPERIMENT_NAME = 'EXP_2'
CORPUS_PATH = 'data/pride_and_prejudice_cleaned.txt'

TRAINING_WINDOW    = 3
CONTEXT_DIMENSION  = 64
CONTEXT_DECAY      = 0.5
CONTRASTIVE_WEIGHT = 0.001
LEANING_RATE       = 1
DROPOUT            = 0.1


myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb       = myclient["mydatabase"]
collection = mydb.parallel_trainging



'''
Experiment details:

contrastive_loss magnitude get up to 50-100 time the primary loss
Set CONTRASTIVE_WEIGHT to 0.001 in next experiment.

Rutime is directly proportional to TRAINING_WINDOW 
Set TRAINING_WINDOW = 3 (to experiment with )




'''


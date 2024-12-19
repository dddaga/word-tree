import numpy as np

# Subword Tokenization Configuration
SUBWORD_METHOD = 'BPE'  # Options: 'BPE', 'WordPiece', etc.
MIN_SUBWORD_FREQ = 5

# Graph Configuration
GRAPH_DB_URI = 'bolt://localhost:7687'
GRAPH_DB_USER = 'neo4j'
GRAPH_DB_PASSWORD = 'password'
MAX_CARDINALITY = 5

# Model Configuration
N = 8  # Degree of freedom for tensor elements
M = 32  # Length of the tensors (1D vectors)
LEARNING_RATE = 0.001
ACTIVATION_THRESHOLD = 0.1

# Training Configuration
BATCH_SIZE = 256
EPOCHS = 10
THREADS = 4
CHUNK_SIZE = 1024

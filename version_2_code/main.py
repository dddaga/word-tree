from preprocessing.subword_tokenization import SubwordTokenizer
from graph.graph_operations import GraphOperations
from model.context_propagation import ContextPropagation
from services.training.trainer import GraphTrainer
from services.training.orchestrator import TrainingOrchestrator
from config import *

# Initialize components

tokenizer = SubwordTokenizer(method=SUBWORD_METHOD, min_freq=MIN_SUBWORD_FREQ)
graph_ops = GraphOperations(GRAPH_DB_URI, GRAPH_DB_USER, GRAPH_DB_PASSWORD)
context_prop = ContextPropagation(N, M, ACTIVATION_THRESHOLD, MAX_ACTIVATION_STRENGTH)
trainer = GraphTrainer(graph_ops, context_prop)
orchestrator = TrainingOrchestrator(tokenizer, graph_ops, context_prop, trainer)

# Start training
orchestrator.train(CORPUS_PATH)

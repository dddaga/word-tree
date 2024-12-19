import torch
import torch.optim as optim
import time
import uuid
from config import *

class TrainingOrchestrator:
    def __init__(self, tokenizer, graph_ops, context_prop, trainer):
        """
        Initialize training orchestrator
        Args:
            tokenizer: Subword tokenizer instance
            graph_ops: Graph operations instance
            context_prop: Context propagation instance
            trainer: Graph trainer instance
        """
        self.tokenizer = tokenizer
        self.graph_ops = graph_ops
        self.context_prop = context_prop
        self.trainer = trainer
        self.lock_manager = Lock() if DB == 'MONGO' or DB == 'REDIS' else None

    def initialize_nodes(self, subwords):
        """Initialize nodes if they don't exist"""
        for subword in subwords:
            if not self.graph_ops.find_node('Subword', 'subword', subword):
                self.graph_ops.create_subword_node(subword)

    def train_sequence(self, sequence):
        """Train on a single sequence"""
        subwords = self.tokenizer.tokenize(sequence)
        self.initialize_nodes(subwords)
        
        # Initialize running context
        running_context = ComplexTensor(N=M, steps=THETA_STEPS, max_clip_value=MAX_CLIP_VALUE)
        optimizer = optim.Adam([running_context.tensor], lr=LEARNING_RATE)
        
        sequence_loss = 0
        for i in range(len(subwords) - 1):
            current_node = subwords[i]
            next_node = subwords[i + 1]
            
            # Acquire lock for subgraph
            if self.lock_manager:
                sub_graph = self.lock_manager([current_node, next_node])
                while sub_graph.locked:
                    time.sleep(0.1)
                    sub_graph.check_status()
                sub_graph.set_lock()

            try:
                # Train step
                running_context, step_loss = self.trainer.train_step(
                    current_node, next_node, running_context, optimizer)
                sequence_loss += step_loss
                
            except Exception as e:
                print(f"Error training sequence: {e}")
                
            finally:
                # Release lock
                if self.lock_manager:
                    sub_graph.release_lock()
                    del sub_graph
        
        return sequence_loss / (len(subwords) - 1)

    def train(self, corpus_path):
        """Train on entire corpus"""
        self.tokenizer.train(corpus_path)
        start_time = time.time()
        total_loss = 0
        n_sequences = 0
        
        for chunk in get_chunks(corpus_path, CHUNK_SIZE):
            chunk_loss = 0
            for sequence in chunk:
                sequence_loss = self.train_sequence(sequence)
                chunk_loss += sequence_loss
                n_sequences += 1
            
            avg_chunk_loss = chunk_loss / len(chunk)
            total_loss += chunk_loss
            print(f"Chunk completed. Average loss: {avg_chunk_loss:.4f}")
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        print(f"Final average loss: {total_loss/n_sequences:.4f}")
        
        self.graph_ops.close()

def get_chunks(corpus_path, chunk_size):
    with open(corpus_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

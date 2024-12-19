# Version 2 Codebase

## Overview

This project is focused on developing a scalable and efficient model for sequence prediction using a graph-based approach. The model leverages context-aware weight distribution across a Directed Acyclic Graph (DAG), where each node represents a subword, and each connection holds contextual weight.

## Directory Structure

- **config/**: Contains configuration files for model parameters and database connections.
- **graph/**: Handles graph operations and database interactions.
- **model/**: Includes implementations of complex tensor operations and context propagation.
- **preprocessing/**: Contains tools for subword tokenization.
- **services/**:
  - **training/**: Manages the training process, including orchestrating tasks and executing training steps.
  - **inference/**: Provides tools for sequence generation and setting up an inference server.
- **utilities/**: Contains utility functions, such as logging.
- **main.py**: The main entry point for starting the training process.

## Design and Implementation

### Graph-Based Model

The model uses a graph to represent sequences at the subword level, enforcing context aware weight distribution by design. This approach aims to improves scalability by only retrieving and processing relevant subgraphs during inference.

## Context and Related Work

### Version 1
[Release details](https://docs.google.com/document/d/1lg0FeP-rpvj5-Kgj6b_QxpQNCK_ZBVrRksyGeA_LB9k/edit?usp=sharing)
(This was written before LLMs and there usecases were lesser known, so the trageted usecase may seem outdated)

This version focused on enabling scalability of training via hyper distributed computing, establishing a foundation for graph-based sequence modeling.


### Literature Survey

Research such as "Knowledge Circuits in Pretrained Transformers" highlights the importance of understanding internal model mechanisms. This project draws inspiration from such studies, aiming to create a model that not only predicts sequences but also adapts dynamically to new data, similar to how knowledge circuits function in LLMs.

### Word Tree

The concept of a word tree is central to this project, where sequences are broken down into subwords, and their relationships are mapped in a graph. This structure allows for efficient context propagation and sequence prediction, addressing the challenges of scalability and real-time processing.

## Future Directions

The project is designed with modularity as a priority, allowing for future expansions such as porting the model to other platforms like mobile devices for real-time, federated learning.
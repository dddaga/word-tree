# Version 1 Codebase

## Introduction

The Version 1 codebase is an ambitious project aimed at developing a scalable, real-time AI model for sequence completion over a set of discrete classes. This project draws inspiration from the impressive information compression capabilities of modern Large Language Models (LLMs) and seeks to provide a more scale-efficient alternative to traditional transformer-based models.

## Motivation

### The Challenge

LLMs have transformed AI interactions but present significant challenges in terms of scalability. Their vast number of parameters makes them computationally expensive and difficult to scale to millions of daily users. The goal is to build a model that not only understands sequences but also predicts the next element with precision, all while remaining scalable and responsive in real-time.

### The Solution

The project explores using a Directed Acyclic Graph (DAG) to represent sequences at the subword level. This approach leverages context-aware weight distribution across the graph, where each node represents a subword, and each connection holds contextual weight. By selectively retrieving and serving only the pertinent subgraphs, the system reduces computational burden, making the model more scalable. This dynamic topology adapts as it processes more data, akin to just-in-time (JIT) compilation for the forward pass.

## Work Done

### Graph-Based Sequence Modeling

In Version 1, we laid the groundwork for graph-based sequence modeling using MongoDB and Redis for graph operations. These tools, while effective for initial development, were identified as suboptimal for complex graph manipulations, leading to improvements in subsequent versions.

### Tensor Operations

The system utilizes PyTorch for tensor operations, providing a flexible and performant framework for manipulating word connections and contexts. This setup allows for efficient context propagation and sequence prediction.

## System Capabilities

- **Scalability**: The model is designed to handle large-scale operations efficiently, adapting dynamically to the data it processes.
- **Real-Time Processing**: With a focus on real-time responsiveness, the system can manage sequence predictions swiftly and accurately.
- **Modular Design**: The architecture is modular, facilitating future expansions and adaptations, such as porting to mobile platforms for federated learning.

## Releases and Future Directions

[Release details](https://docs.google.com/document/d/1lg0FeP-rpvj5-Kgj6b_QxpQNCK_ZBVrRksyGeA_LB9k/edit?usp=sharing)
(This was written before LLMs and there usecases were lesser known, so this is a bit outdated)

This version focuses on enabling scalability of training via hyper distributed computing, establishing a foundation for graph-based sequence modeling. Future releases aim to enhance accuracy and adaptability, with plans to explore deployment on mobile platforms and experiments with federated learning.

## References and Learning Materials

For those interested in delving deeper into the concepts and methodologies used in this project, the following resources are recommended:

- **Knowledge Circuits in Pretrained Transformers**: This paper provides insights into the internal mechanisms of LLMs, offering valuable context for understanding the project's approach.
- **PyTorch Documentation**: A comprehensive resource for learning about tensor operations and deep learning frameworks.
- **Graph Databases**: Explore Neo4j and other graph databases to understand their advantages in handling complex relationships and queries.

## Resources

- **Graphs**: [Google Drive](https://drive.google.com/drive/folders/1P16RkA4j0zzuuEf3Fin2fO6m4qPoIQBm?usp=sharing)
- **LASER (Python Library)**: [Engati Blog](https://www.engati.com/blog/laser-for-nlp-tasks-part-ii)
- **Corpus Sources**: [KDnuggets](https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html)
- **NLP**: [McCormick ML](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/)
- **Reinforcement Learning**: [YouTube Video](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- **TF Graph Neural Network Samples**: [GitHub](https://github.com/microsoft/tf-gnn-samples)
- **Knowledge Graph-Based Text Generation**: [arXiv Paper](https://arxiv.org/abs/2012.10813)
- **Ablation Study**: [Stack Exchange](https://stats.stackexchange.com/questions/380040/what-is-an-ablation-study-and-is-there-a-systematic-way-to-perform-it)
- **TensorFlow (First Lecture)**: [YouTube Video](https://youtu.be/g-EvyKpZjmQ)
- **PyTorch**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/)
- **Unsupervised Learning (Auto Regression)** (Watch first 20 mins): [YouTube Video](https://youtu.be/rjZCjosEFpI)

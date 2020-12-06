import torch
import numpy as np
import pandas as pd
from src.utilities.graph_oprations_db import  get_ngram
from config import CONTEXT_DIMENSIN
from src.services.get_corpus import load_corpus, train_step_genrator
from src.utilities.policies.v_1 import train_graph
import time




def train_context():
    running_context = torch.zeros(CONTEXT_DIMENSIN)

    for step in train_step_genrator(load_corpus()):
        neighbours = get_ngram(step, 2)
        context_history = neighbours[:-1]
        target = neighbours[-1]
        running_context = train_graph(context_history,target,running_context)


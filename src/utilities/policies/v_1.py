import torch
import numpy as np
from src.utilities.graph_oprations.mongo_db import touch_connection_db, find_word, update_graph_context
from config import DROPOUT,CONTRASTIVE_WEIGHT,CONTEXT_DECAY,LEANING_RATE


drop = torch.nn.Dropout(p=DROPOUT)


def train_graph(context_history,target,running_context):

    # Building graph  ################################################################################################

    connection_tensors = []
    context_trajectory = []
    for neighbour in context_history:
        connection = touch_connection_db(neighbour[0], neighbour[1])
        a = torch.tensor(connection['context'] ,requires_grad=True)
        connection_tensors.append(a)
        context_trajectory.append(connection)
        context_step = a.add(drop(running_context))
        running_context = context_step / context_step.norm()

    target_connection = touch_connection_db(target[0] ,target[1])
    target_context    = torch.tensor(target_connection['context'],requires_grad=True)
    primary_error  =  running_context - target_context
    primary_loss   =  primary_error.square().sum()

    #secondary error
    connection_df = find_word(target[0])
    connection_count = len(connection_df)
    negative_df = connection_df.loc[connection_df['connection'] != target[1]]
    negative_tensors = []
    contrastive_loss = 0
    for index, negative in negative_df.iterrows():
        c = torch.tensor(negative['context'],requires_grad=True)
        negative_tensors.append(c)
        contrast_error = drop(running_context) - c
        contrastive_loss += contrast_error.square().sum()

    loss = primary_loss - CONTRASTIVE_WEIGHT * (contrastive_loss / connection_count)

    # Updating weights ########################################################################################
    loss.backward()

    print('primary loss is', primary_loss )
    print('contrastive loss is ',contrastive_loss / connection_count)

    # update context:
    connection_count = len(context_trajectory)
    for index, connection in enumerate(context_trajectory):
        update_count = connection['update_count']
        gradient = connection_tensors[index].grad.numpy()
        weight = LEANING_RATE * (CONTEXT_DECAY ** (connection_count - index)) * (1 / np.sqrt(update_count + 1))
        connection['updated_context'] = connection['context'] - weight * gradient
        update_graph_context(connection)

    target_gradient = target_context.grad.numpy()
    traget_update_count = target_connection['update_count'] +1
    target_connection['updated_context'] = target_connection['context'] - \
                             LEANING_RATE * (1/np.sqrt(traget_update_count))*target_gradient
    update_graph_context(target_connection,update_count=True)

    negative_df['tensors'] = negative_tensors
    negative_df['updated_context'] = negative_df['context'] - \
                                     LEANING_RATE * negative_df['tensors'].apply(lambda x :x.grad.numpy())
    negative_df.apply(update_graph_context,axis=1)

    return torch.tensor(context_trajectory[0]['updated_context'])
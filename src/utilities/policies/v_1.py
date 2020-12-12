import torch
import numpy as np
import random
from config import DB, DROPOUT,CONTRASTIVE_WEIGHT,CONTEXT_DECAY,LEANING_RATE, NEGATIVE_SAMPLE_SIZE

if DB == 'MONGO':
    from src.utilities.graph_operations.mongodb import touch_connection_db, find_word, update_graph_context
if DB == 'REDIS':
    from src.utilities.graph_operations.redis import touch_connection_db, find_word, update_graph_context

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
    connection_list = find_word(target[0])
    connection_count = len(connection_list)
    sampling_size    = min(NEGATIVE_SAMPLE_SIZE,connection_count)
    connection_list   = random.sample(connection_list,sampling_size)

    negative_tensors =  []
    negative_neighbours= []
    contrastive_loss = 0
    for negative in connection_list:
        if negative['connection'] != target[1]:
            negative_neighbours.append(negative)
            c = torch.tensor(negative['context'],requires_grad=True)
            negative_tensors.append(c)
            contrast_error = drop(running_context) - c
            contrastive_loss += contrast_error.square().sum()

    loss = primary_loss - CONTRASTIVE_WEIGHT * (contrastive_loss / sampling_size)

    # Updating weights ########################################################################################
    loss.backward()

    print('primary loss is', primary_loss )
    print('contrastive loss is ',contrastive_loss / connection_count)

    # update context:
    context_updates= []

    connection_count = len(context_trajectory)
    for index, connection in enumerate(context_trajectory):
        update_count = connection['update_count']
        gradient = connection_tensors[index].grad.numpy()
        weight = LEANING_RATE * (CONTEXT_DECAY ** (connection_count - index)) * (1 / np.sqrt(update_count + 1))
        connection['updated_context'] = connection['context'] - weight * gradient
        context_updates.append(connection)

    target_gradient = target_context.grad.numpy()
    traget_update_count = target_connection['update_count'] +1
    target_connection['updated_context'] = target_connection['context'] - \
                             LEANING_RATE * (1/np.sqrt(traget_update_count))*target_gradient
    update_graph_context([target_connection],update_count=True)


    for negative_index,negative_connction in enumerate(negative_neighbours):
        upate_count = negative_connction['update_count']
        negative_gradient = negative_tensors[negative_index].grad.numpy()
        negative_connction['updated_context'] = negative_connction['context']  - LEANING_RATE *(1 / np.sqrt(update_count + 1))* negative_gradient
        context_updates.append(negative_connction)

    update_graph_context(context_updates)

    return torch.tensor(context_trajectory[0]['updated_context'])

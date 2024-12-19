import numpy as np
from .neo4j_connector import Neo4jConnector
from config import MAX_CARDINALITY, N, M

class GraphOperations:
    def __init__(self, uri, user, password):
        self.db = Neo4jConnector(uri, user, password)

    def create_subword_node(self, subword):
        properties = {
            'subword': subword,
            'vector': list(np.random.uniform(-1, 1, M)),
            'cardinality': 0
        }
        return self.db.create_node('Subword', properties)

    def create_connection(self, parent_subword, child_subword, context_vector):
        properties = {
            'context': list(context_vector),
            'weight': list(np.random.uniform(-1, 1, M))
        }
        return self.db.create_relationship('Subword', 'Subword', 'CONNECTS_TO', properties)

    def update_node(self, subword, new_vector):
        query = "MATCH (n:Subword {subword: $subword}) SET n.vector = $vector RETURN n"
        return self.db.execute_query(query, {'subword': subword, 'vector': list(new_vector)})

    def get_connected_subwords(self, subword):
        query = ("MATCH (a:Subword {subword: $subword})-[r:CONNECTS_TO]->(b:Subword) "
                 "RETURN b.subword AS subword, r.context AS context, r.weight AS weight")
        return self.db.execute_query(query, {'subword': subword})

    def close(self):
        self.db.close()

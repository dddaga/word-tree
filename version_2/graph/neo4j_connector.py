from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    def create_node(self, label, properties):
        query = f"CREATE (n:{label} {{props}}) RETURN n"
        return self.execute_query(query, {'props': properties})

    def create_relationship(self, label1, label2, relationship, properties):
        query = (f"MATCH (a:{label1}), (b:{label2}) "
                 f"CREATE (a)-[r:{relationship} {{props}}]->(b) RETURN r")
        return self.execute_query(query, {'props': properties})

    def find_node(self, label, key, value):
        query = f"MATCH (n:{label} {{{key}: '{value}'}}) RETURN n"
        return self.execute_query(query)

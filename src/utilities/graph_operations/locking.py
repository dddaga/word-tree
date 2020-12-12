# from config import collection
#
# class Lock:
#
#     def __init__(self,sub_graph):
#         self.sub_graph   = sub_graph
#         self.locked = False
#
#     def check_status(self):
#         connection_index=0
#         self.locked = False
#         path_size = len(self.sub_graph)
#         while  (not self.locked) and (connection_index < path_size):
#             node = collection.find_one({'word':self.sub_graph[connection_index][0] ,'connection': self.sub_graph[connection_index][1] })
#             connection_index += 1
#             if node is not None:
#                 self.locked = node['lock'] or self.locked
#
#         return self.locked
#
#     def set_lock(self):
#         for connection in self.sub_graph:
#             collection.update_one({'word': connection[0] ,'connection' : connection[1]} ,{'$set' :{'lock':True} } )
#
#     def release_lock(self):
#         for connection in self.sub_graph:
#             collection.update_one({'word': connection[0], 'connection': connection[1]}, {'$set': {'lock': False}})
#
#
#
#
# def unlock_graph():
#     locked_nodes = list(collection.find({'lock':True}))
#     for node in locked_nodes :
#         collection.update_one({'word': node['word'], 'connection': node['connection']}, {'$set': {'lock': False}})

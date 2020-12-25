#import redis
#import pymongo
import numpy as np



def redis_to_mongo_v1(redis_db,mongo_collection):
    connection_keys = redis_db.keys()
    for key in connection_keys :
        connection = key.decode("utf-8").split(':')
        try :
            print(key)
            connection_properties    = redis_db.hgetall(key)
            update_count             = int(connection_properties[b'update_count'])
            context                  = np.frombuffer(connection_properties[b'context'])
            mongo_collection.insert_one({'word':connection[0], 'connection' : connection[1] ,'update_count':update_count,'context':list(context) })
        except Exception as e:
            print(' {} \n error in inserting {}'.format(e, connection))
            

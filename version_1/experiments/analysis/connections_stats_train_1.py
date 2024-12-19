import pymongo
import pandas as pd




myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb       = myclient["mydatabase"]
collection_read = mydb.train_1

collection_write = mydb.train_1_stats


def get_connection_count(word):
    word_df = pd.DataFrame(collection_read.find({'word':word}))
    connection_count = len(word_df)
    if connection_count > 0 :
        connections = word_df['connection']
    else :
        connections = []
        
    return connection_count, connections

def get_levelwise_count(word, depth):
    level_count = [0]*depth

    def count_children(word, index=0, level_count=level_count):
        while index < depth :   
            count ,connections = get_connection_count(word)
            level_count[index] += count
            if count > 0:
                for connection in connections:
                    level_count = count_children(connection,index+1,level_count)              
            index +=1
        #print(level_count)
        return level_count

    return count_children(word,0,level_count)




words = collection_read.distinct('word')

for word in words:
    print(word)
    
    counts = get_levelwise_count(word, 3)
    print(counts)
    
    collection_write.insert_one({'word':word, 'level_1':counts[0] , 'level_2':counts[1], 'level_3':counts[2]})
    
    
    



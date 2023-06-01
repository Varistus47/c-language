#a*
import heapq
class Node:
    def __init__(self,state,parent,cost,heuristic):
        self.state=state
        self.parent=parent 
        self.cost=cost
        self.heuristic=heuristic
    def __lt__(self,other):
        return (self.cost+self.heuristic<self.cost+other.heuristic)
def astar(start,goal,graph):
    heap=[]
    heapq.heappush(heap,(0,Node(start,None,0,0)))
    visited=set()
    while heap:
        (cost,current)=heapq.heappop(heap)
        if current.state==goal:
            path=[]
            while current is not None:
                path.append(current.state)
                current=current.parent
            return path[::-1]
        if current.state in visited:
            continue
        visited.add(current.state)
        for state,cost in graph[current.state].items():
            if state not in visited:
                heuristic=0
                heapq.heappush(heap,(cost,Node(state,current,current.cost+cost,heuristic)))
    return None

graph={
    'A':{'B':1,'D':3},
    'B':{'A':1,'C':2,'D':4},
    'C':{'B':2,'D':5,'E':2},
    'D':{'A':3,'B':4,'C':5,'E':3},
    'E':{'C':2,'D':3}
}
start='A'
goal='E'
result=astar(start,goal,graph)
print(result)
            


#simple nn node
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=4)
test_data=np.array([[0,0],[0,1],[1,0],[1,1]])
predict=model.predict(test_data)
print(predict)


#dfs
graph = {
'5' : ['3','7'],
'3' : ['2', '4'],
'7' : ['8'],
'2' : [],
'4' : ['8'],
'8' : []
}
visited = set() 
def dfs(visited, graph, node): 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
print("Following is the Depth-First Search")
dfs(visited, graph, '5')

#bfs
graph = {
'5' : ['3','7'],
'3' : ['2', '4'],
'7' : ['8'],
'2' : [],
'4' : ['8'],
'8' : []
}  
visited = []
queue = []
def bfs(visited,graph, node): 
    visited.append(node)
    queue.append(node)
    while queue: 
        m = queue.pop(0)
        print (m, end = " ")
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

print("Following is the Breadth-First Search")
bfs(visited,graph, '5') 

#last nn
import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense 
import keras
model = Sequential() 
model.add(Dense(units=64, activation='relu', input_dim=100)) 
model.add(Dense(units=10, activation='softmax')) 
model.compile(loss='categorical_crossentropy', 
 optimizer='sgd', 
 metrics=['accuracy']) 
data = np.random.random((1000, 100)) 
labels = np.random.randint(10, size=(1000, 1)) 
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10) 
# Train the model on the data 
model.fit(data, one_hot_labels, epochs=10, batch_size=32) 
# Evaluate the model on a test set 
test_data = np.random.random((100, 100)) 
test_labels = np.random.randint(10, size=(100, 1)) 
test_one_hot_labels = keras.utils.to_categorical(test_labels, num_classes=10) 
loss_and_metrics = model.evaluate(test_data, test_one_hot_labels, batch_size=32) 
print("Test loss:", loss_and_metrics[0]) 
print("Test accuracy:", loss_and_metrics[1]) 
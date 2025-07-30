import hnswlib
import numpy as np


dim = 256   #dimension of vectors
num_elements = 10000    #number of vectors
data = np.random.rand(num_elements,dim).astype(np.float32)  #example data


# index making

p = hnswlib.Index(space = 'l2', dim = dim) 


#index initialize
p.init_index(max_elements=num_elements, ef_construction=200, M = 16)
p.set_ef(200)

p.add_items(data)

query_vector = np.random.rand(dim).astype(np.float32)

labels, distances = p.knn_query(query_vector,k = 5)

print("Nearest Neighbors'   labels: ", labels)
print("Distances: ", distances)
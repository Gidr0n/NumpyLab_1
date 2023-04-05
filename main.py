#1
import numpy as np
arr = np.random.uniform(0,20,28).reshape(4,7)
print('Массив \n' ,arr)
def normalize(arr):
    norm = np.linalg.norm(arr)
    arr = arr/norm
    return arr
print('Нормированный массив \n',normalize(arr))

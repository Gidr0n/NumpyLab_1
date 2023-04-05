#1
import numpy as np
arr = np.random.uniform(0,20,28).reshape(4,7)
print('Массив \n' ,arr)
def normalize(arr):
    norm = np.linalg.norm(arr)
    arr = arr/norm
    return arr
print('Нормированный массив \n',normalize(arr))
#2
import numpy as np
k=0
arr = np.random.randint(0, 10, size = (8, 10))
print(arr)
stroka = min((r for r in arr) , key = sum)
print(stroka)
#3
import numpy as np
vector1= np.random.randint(0, 11, size=(1, 3))
vector2 = np.random.randint(0, 11, size=(1, 3))
ev_rastoyanie = np.linalg.norm(vector1-vector2)
print(vector1,vector2)
print(ev_rastoyanie)
#4
import numpy as np
A = [[-1, 2, 4], [-3, 1, 2], [-3, 0, 1]]
B=[[3, -1], [2, 1]]
C=[[7, 21], [11, 8], [8, 4]]
#A*X*B=-C
#X=A^(-1)*(-C)*B^(-1)
a=np.array(A)
b=np.array(B)
c=np.array(C)
c=-1*c
a=np.linalg.inv(a)
b=np.linalg.inv(b)
y=np.dot(a,c)
x=np.dot(y,b)
print(x)
#Лабораторная работа №1
import numpy as np
import pandas
data = pandas.read_csv('minutes_n_ingredients.csv')
print(data)
#1
data = np.loadtxt('minutes_n_ingredients.csv', delimiter=',', skiprows=1, dtype=np.int32)
print(data[:5])
#2
print(data[:, 1:].mean(axis=0), data[:, 1:].min(axis=0), data[:, 1:].max(axis=0), np.median(data[:, 1:], axis=0), sep='\n')
#3
q = np.quantile(data[:, 1], q=0.75)
data[:, 1] = data[:, 1].clip(max=q)
print(q)
#4
print(data[data[:, 1]==0].shape[0])
data[data[:, 1]==0, 1]=1
#5
len(np.unique(data[:, 0], axis=0))
#6
len(np.unique(data[:, 2]))
np.unique(data[:, 2])
#7
data_n_ingr_m5 = data[data[:, 2] <= 5].copy()
data_n_ingr_m5
#8
medium_ingredient_by_minutes = data[:, 2] / data[:, 1]
medium_ingredient_by_minutes.mean(), medium_ingredient_by_minutes.max()
#9
data[data[:, 1].argsort()][-100:][:, 2].mean()
#10
random_recept = np.random.randint(0, 10000, size=10)
data[random_recept]
#11
sred_ingredient = data[:, 2].mean()
len(data[data[:, 2] < sred_ingredient]) / len(data) * 100
#12
prostoy = (data[:, 1] <= 20) & (data[:, 2] <= 5)
prostoy = prostoy.astype(np.int32)
prostoy = prostoy[:, np.newaxis]
data_prostoy = np.hstack((data, prostoy))
data_prostoy
#13
len(data_prostoy[data_prostoy[:, 3] == 1]) / len(data_prostoy) * 100
#14
short = data[data[:, 1] < 10]
standart = data[(data[:, 1] >= 10) & (data[:, 1] < 20)]
long = data[data[:, 1] >= 20]
stop = min([len(arr) for arr in [short, standart, long]])
np.array([short[:stop], standart[:stop], long[:stop]])

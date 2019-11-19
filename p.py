# we will have data
import pandas as pd 
from pandas.plotting import scatter_matrix
import numpy as np 
import matplotlib.pyplot as plt 

# we will need to deal with matrices

# weights will be a row vector
w = np.array([[1.0, 2.0, 3.0]]) 

# data will be a col vector
x = np.array([[1.0, 2.0, 3.0]]).T
print(w)

print(x)

# for the perceptron, we will need the dot product

print(np.dot(w, x))

# in general, introduce x_0 = 1 and w_0 = -threshold

x[0] = 1
w[0, 0] = -0.7

# load the test dataset

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names = names)

print(data.shape) 
print(data.head(20)) # seeing the dataset
print(data.groupby('class').size()) # how many of each

data.hist() # each attribute
# plt.show()

# now let's look at multivariate plots

scatter_matrix(data)
# plt.show()

array = data.values
array[0:50] 
array[50:100]   # these two are linearly separable
array[100:150] # not this one tho

# we will use just the linearly separable data

linear_data = array[0:100]
linear_data
train_data1 = array[0:40]
train_data2 = array[50:90]
train_data = np.concatenate((train_data1, train_data2))
train_data.shape

train_data.shape

# we will let 1 = Iris-setosa and -1 = Iris-versicolor
xs = []
ys = []
for i in range(len(train_data)):
    ys.append(train_data[i][4])
    xs.append(np.concatenate((np.array([1]).T, train_data[i][:4].T)))

xs = np.array(xs)
ys = np.array(ys)
for i in range(len(ys)):
    if ys[i] == 'Iris-setosa':
        ys[i] = 1
    else:
        ys[i] = -1
print(xs)
print(ys)

# xs is our data
# ys is the desired result

ws = [0] * 5
ws = np.array(ws)

print(len(np.dot(ws, xs.T)))

# sign(1)

# while True:
#     i = 0
# while True:
misclassified = 10
while misclassified != 0:
    misclassified = 0
    for i in range(80):
        sign = np.sign(np.dot(ws, xs[i].T))
        sign
        if sign == 0:
            sign = 1


        if sign != ys[i]: # misclassified point
            misclassified += 1
            if ys[i] == 1:
                ws = ws + xs[i]
            elif (ys[i] == -1):
                ws = ws - xs[i]

        print(ws)
    




    

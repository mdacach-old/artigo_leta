# data analysis and manipulation
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# visualization
import seaborn as sns

# we will do our algorithm by hand

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())
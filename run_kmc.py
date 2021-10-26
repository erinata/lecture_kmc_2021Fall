import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot


data = pandas.read_csv("dataset.csv")

# print(data)

data = data.values

# print(data)

pyplot.scatter(data[:,0], data[:,1])
pyplot.savefig("scatterplot.png")


machine = KMeans(n_clusters=4)
machine.fit(data)
results = machine.predict(data)

print(results)
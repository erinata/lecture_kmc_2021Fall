import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot

data = pandas.read_csv("dataset.csv")

# print(data)

data = data.values

# print(data)

pyplot.scatter(data[:,0], data[:,1])
pyplot.savefig("scatterplot.png")


def run_kmeans(n, data):
	machine = KMeans(n_clusters=n)
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	print(ssd)
	pyplot.scatter(data[:,0], data[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_color_" + str(n) + ".png")
	pyplot.close()


run_kmeans(1, data)
run_kmeans(2, data)
run_kmeans(3, data)
run_kmeans(4, data)
run_kmeans(5, data)
run_kmeans(6, data)
run_kmeans(7, data)





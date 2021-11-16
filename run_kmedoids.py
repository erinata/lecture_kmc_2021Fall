import pandas
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids 

import matplotlib.pyplot as pyplot

from sklearn.metrics import silhouette_score

# data = pandas.read_csv("dataset.csv")
data = pandas.read_csv("dataset_small2.csv")

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
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, machine.labels_, metric='euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_kmeans_color_" + str(n) + ".png")
	pyplot.close()
	return ssd, silhouette


def run_kmedoids(n, data):
	machine = KMedoids(n_clusters=n)
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, machine.labels_, metric='euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=60)
	pyplot.savefig("scatterplot_kmedoids_color_" + str(n) + ".png")
	pyplot.close()
	return ssd, silhouette





result= [ run_kmeans(i+1, data) for i in range(7)]

ssd_result = [ i[0] for i in result] 
silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("kmeans_ssd.png")
pyplot.close()

ssd_result_diff = [ ssd_result[i-1] - x for i,x  in enumerate(ssd_result)][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmeans_silhouette.png")
pyplot.close()

print("KMeans")
print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", 
	silhouette_result.index(max(silhouette_result))+2)







result= [ run_kmedoids(i+1, data) for i in range(7)]

ssd_result = [ i[0] for i in result] 
silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("kmedoids_ssd.png")
pyplot.close()

ssd_result_diff = [ ssd_result[i-1] - x for i,x  in enumerate(ssd_result)][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmedoids_silhouette.png")
pyplot.close()

print("KMedoids")
print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", 
	silhouette_result.index(max(silhouette_result))+2)




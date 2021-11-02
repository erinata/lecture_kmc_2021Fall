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
	pyplot.scatter(data[:,0], data[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_color_" + str(n) + ".png")
	pyplot.close()
	return ssd

# result = []
# for i in range(7):
# 	ssd = run_kmeans(i+1, data)	
# 	result.append(ssd)

result = [ run_kmeans(i+1, data) for i in range(7)]
print(result)

pyplot.plot(range(7), result)
pyplot.savefig("ssd.png")
pyplot.close()

# result_diff = []
# for i,x  in enumerate(result):
# 	diff = result[i-1] - x
# 	result_diff.append(diff)

result_diff = [ result[i-1] - x for i,x  in enumerate(result)][1:]
print(result_diff)









# ssd1 = run_kmeans(1, data)
# ssd2 = run_kmeans(2, data)
# ssd3 = run_kmeans(3, data)
# ssd4 = run_kmeans(4, data)
# ssd5 = run_kmeans(5, data)
# ssd6 = run_kmeans(6, data)
# ssd7 = run_kmeans(7, data)





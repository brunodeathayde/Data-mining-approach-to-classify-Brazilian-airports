
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot


file_name = 'data_clean.csv'
data = pd.read_csv(file_name, sep=";",low_memory=False)
data = data.drop(columns=['ANO', 'MES']).groupby('AEROPORTO').mean().reset_index()
airports = data[['AEROPORTO']]
data = data.drop(columns=['AEROPORTO'])

# Calculate the variance of each feature
variances = data.var(axis=0)

# Normalize the variance values between 0 and 1
normalized_vector = (variances - np.min(variances)) / (np.max(variances) - np.min(variances))


data = data.drop(columns=['TOTAL_PASSAGEIROS_PAGOS'])
data = data.drop(columns=['TOTAL_PASSAGEIROS_GRATIS'])
data = data.drop(columns=['TOTAL_CARGA_PAGA_KG'])
data = data.drop(columns=['TOTAL_CARGA_GRATIS_KG'])
data = data.drop(columns=['TOTAL_CORREIO_KG'])
data = data.drop(columns=['TOTAL_COMBUSTIVEL_LITROS'])
data = data.drop(columns=['TOTAL_DISTANCIA_VOADA_KM'])
data = data.drop(columns=['TOTAL_DECOLAGENS'])
data = data.drop(columns=['TOTAL_ASSENTOS'])

# Standarize data
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)

# Aplying k-medoids algorithm
kmedoids = KMedoids(n_clusters=4, random_state=0)
kmedoids.fit(data_norm)
labels = kmedoids.labels_

# Calculate silhouette score
silhouette_avg = silhouette_score(data_norm, labels)
print(f"The average silhouette score is: {silhouette_avg}")

silhouette_vals = silhouette_samples(data_norm, labels)
y_lower, y_upper = 0, 0
yticks = []
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_vals = silhouette_vals[labels == cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
    yticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, np.unique(labels) + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()


# Initialize and train the SOM
som = MiniSom(x=10, y=10, input_len=5, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_norm)
som.train_random(data_norm, num_iteration=100)

# Get the shape of the SOM grid
weights_shape = som.get_weights().shape
x, y = weights_shape[0], weights_shape[1]

# Calculate the U-matrix
u_matrix = np.zeros((x, y))
for i in range(x):
    for j in range(y):
        neighbors = [som.get_weights()[ni, nj]
                     for ni in range(max(0, i-1), min(x, i+2))
                     for nj in range(max(0, j-1), min(y, j+2))]
        u_matrix[i, j] = np.mean([np.linalg.norm(som.get_weights()[i, j] - neighbor)
                                  for neighbor in neighbors])

# Plotting the U-matrix
plt.figure(figsize=(10, 10))
plt.pcolor(u_matrix.T, cmap='coolwarm')
plt.colorbar(label='Distance')
plt.title('U-Matrix')
plt.show()

airports['TOTAL_ASK'] = data[['TOTAL_ASK']]
airports['TOTAL_RPK'] = data[['TOTAL_RPK']]
airports['TOTAL_ATK'] = data[['TOTAL_ATK']]
airports['TOTAL_RTK'] = data[['TOTAL_RTK']]
airports['TOTAL_PAYLOAD'] = data[['TOTAL_PAYLOAD']]
airports['CLUSTERS'] = labels

airports.to_excel('results.xlsx', index=False)





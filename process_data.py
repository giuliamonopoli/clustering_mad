import config as cfg
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def load_patient_data(clinical_datapath):
    patientdata = pd.read_excel(clinical_datapath).drop([0,1])
    patientdata["Pat_no"] = patientdata["Pat_no"].astype(int)
    patientdata = patientdata.set_index("Pat_no")
    return patientdata

clinical_data = load_patient_data(cfg.CLINICAL_DATA_XL)
# print(clinical_data.head())

feature = ["Aborted_cardiac_arrest","Ventricular_tachycardia","nsVT",
#feature_mvp = 
"Mitral_regurg","Leaflet_thickness","MAD_presence","MAD_4CH_length",
#l = [
    "LVs_mass","EF","LA_Volume","CMR_LV_EDV","CMR_LV_ESV","CMR_ESV","CMR_EF","CMR_LGE_Myocardium_percent","CMR_LGE_ANT_Pap_muscle","CMR_LGE_POST_Pap_muscle","CMR_max_ED_thickness_ineferolat_wall",
     "CMR_max_ES_thickness_ineferolat_wall"]


features = clinical_data[feature].fillna(0).to_numpy()
print(features.shape)

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()
    
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(features)

# plt.scatter(features, c=kmeans.labels_)
# plt.show()
from sklearn.decomposition import PCA

# Apply PCA and reduce the features to 2 dimensions for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Fit KMeans with the desired number of clusters
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(features)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', marker='o')

# Marking the centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')

plt.title('Clusters of Patients')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt #import yang digunakan utk gambar 2d ataupun 3d
import numpy as np #utk operasi vector maupun matrix
from sklearn.metrics import silhouette_samples, silhouette_score #untuk menampilkan ketepatan
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

st.title(""" WELCOME TO OUR WEBSITE | CLUSTERING WITH K-MEANS \n""")

with st.sidebar:

    st.header("Machine Learning - 3 TI E")
    # st.caption("Dibuat Oleh") 
    name = '<p color:Black; font-size: 20px;>Nama Kelompok:</b><br/>1. Mistia Adinda Dwi Syahputri<br/>2. Putri Ridha Tasmara<br/>3. Faris Upangga </p>'
    st.markdown(name, unsafe_allow_html=True)

df=pd.read_csv('WineQT.csv')
fg = df.head()

st.subheader("Dataset WineQT.csv: ")
st.write(fg)

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# @st.cache
def data():
    uploaded_file = st.file_uploader("Upload Dataset Terbaru: ")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('WineQT.csv')
    
    dataset = df
    X = dataset.iloc[:,[1,12]].values
    return X
    
X = data()

klaster_slider = st.slider(
    min_value=1, max_value=4, value=3, label="Cluster"
)
kmeans = KMeans(n_clusters=klaster_slider, random_state=2023).fit(X)
labels = kmeans.labels_

seleksi2 = st.selectbox("Standar Devisiasi : ", [1,2,3])

warna = ["red", "seagreen", "orange", "blue", "yellow", "purple"]

jumlah_label = len(set(labels))

individu = st.selectbox("Subplot Tunggal: ", [False, True])
    
if individu:
    fig, ax = plt.subplots(ncols=jumlah_label)
else:
    fig, ax = plt.subplots()

for i, yi in enumerate(set(labels)):
    if not individu:
        a = ax
    else:
        a = ax[i]

    xi = X[labels == yi]
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    a.scatter(x_pts, y_pts, c=warna[yi])

plt.tight_layout()
st.write(fig)
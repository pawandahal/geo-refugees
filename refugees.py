import streamlit as st
import os
import zipfile
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Country Graph GCN with Streamlit")

# Paths (update as needed)
zip_path = 'C:/Users/acer/Downloads/API_SM.POP.REFG.OR_DS2_en_csv_v2_13553.zip'
extract_path = 'C:/Users/acer/Downloads/geospital_data'

# Try-except for zip extraction
try:
    if not os.path.exists(extract_path) or len(os.listdir(extract_path)) == 0:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        st.write("Extracted zip contents:")
        st.write(os.listdir(extract_path))
    else:
        st.write("Using extracted files:" )
        st.write(os.listdir(extract_path))
except FileNotFoundError:
    st.error(f"Zip file not found at: {zip_path}\nPlease place the zip file there before running.")
    st.stop()
except zipfile.BadZipFile:
    st.error(f"File at {zip_path} is not a valid zip file or is corrupted.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during extraction: {e}")
    st.stop()

# Load CSV with error handling
csv_filename = 'API_SM.POP.REFG.OR_DS2_en_csv_v2_13553.csv'
csv_path = os.path.join(extract_path, csv_filename)

try:
    df = pd.read_csv(csv_path, skiprows=4)
except FileNotFoundError:
    st.error(f"CSV file not found at: {csv_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while reading the CSV: {e}")
    st.stop()

# Preprocess
df = df[['Country Name', 'Country Code', '2010', '2011', '2012', '2013', '2014']]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

st.write("Data sample:")
st.dataframe(df.head())

# Normalize features
features = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float)
features = (features - features.mean(dim=0)) / features.std(dim=0)

# Build graph edges based on first letter
edge_index = []
country_names = df['Country Name'].tolist()
for i in range(len(country_names)):
    for j in range(i + 1, len(country_names)):
        if country_names[i][0] == country_names[j][0]:
            edge_index.append([i, j])
            edge_index.append([j, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create dummy labels
labels = torch.randint(0, 2, (features.shape[0],), dtype=torch.long)

# Create graph data object
data = Data(x=features, edge_index=edge_index, y=labels)

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Instantiate model, optimizer, loss
model = GCN(input_dim=features.shape[1], hidden_dim=16, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop with progress bar in Streamlit
epochs = 100
progress_bar = st.progress(0)
status_text = st.empty()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean()
        status_text.text(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
    progress_bar.progress((epoch + 1) / epochs)

# Build NetworkX graph for visualization
G = nx.Graph()
for i in range(data.x.size(0)):
    G.add_node(i, label=country_names[i])

edges = edge_index.t().tolist()
G.add_edges_from(edges)

pos = nx.spring_layout(G, seed=20, k=0.8)

plt.figure(figsize=(24, 14))
nx.draw(
    G, pos,
    with_labels=True,
    labels={i: country_names[i] for i in G.nodes()},
    node_color=data.y.tolist(),
    cmap=plt.cm.Set1,
    node_size=1200,
    font_size=10
)
plt.title("Graph of Countries Connected by First Letter", fontsize=16)

# Display plot in Streamlit
st.pyplot(plt.gcf())

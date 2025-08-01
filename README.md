Geo-Refugees: GCN Visualization of Refugee Data with Streamlit

The latest (version 0.3) This project: automatically gathers data to visualize the trend of population distribution based on refugees by means of Graph Convolutional Network(GCN), and visualizes this information via an interactive Streamlit UI. The countries are linked by their first letters and the GCN focuses on classification of these graph nodes using synthetic labels.

📦 Features

– Extract and clean curse data of World Bank on refugee population

Adjust multi-year data per country - Make sure the list is normalized by one.

If we build a graph in which the countries are vertices and there is an edge between two countries if they start with the same letter, then what we're looking for here?

Implement a simple 2-layer GCN model with pytorch geometric

Visualize the network in Streamlit with NetworkX & matplotlib

🧠 Libraries Used

streamlit

torch, torch_geometric

pandas
## 📁 Dataset

- World Bank dataset file:

`API_SM.POP.REFG.OR_DS2_en_csv_v2_13553.zip`

- 📍 Place the zip file at:

matplotlib, networkx

zipfile, os

import random as rand
import os
import math
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.link_prediction import preferential_attachment, adamic_adar_index
import community as community_louvain

# random generator
rand.seed(89)

# %% Učitavanje podataka
data = pd.read_csv('countries_fb_social_connectedness.csv')
print(data)

data = data.rename(columns={ "user_loc": 
                                "first_location",
                            "fr_loc" :
                                "second_location", 
                            "scaled_sci": 
                                "facebook_connections"})

print(data)

# Osiguranje numeričkih vrednosti
data['facebook_connections'] = pd.to_numeric(data['facebook_connections'], errors='coerce')

# Provera i zamena ne-numeričkih vrednosti sa 0
data['facebook_connections'].fillna(0, inplace=True)

# %% Kreiranje graf objekta
G = nx.Graph()

# Dodavanje čvorova i grana u graf
for index, row in data.iterrows():
    if not pd.isnull(row['facebook_connections']):
        G.add_edge(row['first_location'], row['second_location'], weight=row['facebook_connections'])

# Vizualizacija grafa
plt.figure(figsize=(15, 10))
plt.gca().set_facecolor('white')  # Promena pozadine u belu za bolji kontrast
pos = nx.spring_layout(G, k=0.15)
weights = [G[u][v]['weight'] / 100000 for u, v in G.edges()]  # Smanjite delitelja za tanje linije
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', width=weights, edge_color='black')
plt.title('Mreža društvenih veza između zemalja')
plt.show()

# Dodavanje nove kolone koja pokazuje da li je zemlja više povezana sa sobom
data['more_connected_to_self'] = data.apply(lambda row: row['facebook_connections'] > data[(data['first_location'] == row['first_location']) & (data['second_location'] != row['first_location'])]['facebook_connections'].max(), axis=1)

print(data)

# %% Dodatna analiza
# Identifikacija centralnih čvorova (Analiza centralnosti)
degree_centrality = nx.degree_centrality(G)
central_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
print("Top 10 centralnih zemalja:", central_nodes)

# Prosečna dužina kratke staze
avg_shortest_path_length = {}
for node in central_nodes:
    lengths = nx.shortest_path_length(G, source=node)
    avg_length = np.mean(list(lengths.values()))
    avg_shortest_path_length[node] = avg_length
    print(f"Prosečna dužina kratke staze za {node}: {avg_length}")

# Blizinska centralnost
closeness_centrality = nx.closeness_centrality(G)
central_nodes_closeness = {node: closeness_centrality[node] for node in central_nodes}
print("Blizinska centralnost centralnih zemalja:", central_nodes_closeness)

# Posrednička centralnost
betweenness_centrality = nx.betweenness_centrality(G)
central_nodes_betweenness = {node: betweenness_centrality[node] for node in central_nodes}
print("Posrednička centralnost centralnih zemalja:", central_nodes_betweenness)

# Identifikacija potencijalnih posrednika (zemlje sa visokim posredničkim centralnostima)
potential_intermediaries = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:10]
print("Top 10 zemalja posrednika:", potential_intermediaries)

# Vizualizacija mreže s posredničkim zemljama označenim zeleno
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', width=weights)
nx.draw_networkx_nodes(G, pos, nodelist=potential_intermediaries, node_color='green', node_size=700)
plt.title('Mreža društvenih veza između zemalja s označenim posredničkim zemljama')
plt.show()

# %% Detekcija hubova i analiza otpornosti mreže
# Uklanjanje centralnih čvorova i analiza uticaja
G_copy = G.copy()
for hub in central_nodes:
    G_copy.remove_node(hub)
    # Provera povezanosti mreže
    is_connected = nx.is_connected(G_copy)
    # Analiza prosečne dužine putanja u modifikovanoj mreži
    if is_connected:
        avg_path_length = nx.average_shortest_path_length(G_copy)
    else:
        avg_path_length = float('inf')
    print(f"Nakon uklanjanja {hub}: Povezanost mreže - {is_connected}, Prosečna dužina putanja - {avg_path_length}")

# %% Distribucija stepena
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure()
plt.hist(degree_sequence, bins=20)
plt.title("Distribucija stepena čvorova")
plt.xlabel("Stepen čvora")
plt.ylabel("Broj čvorova")
plt.show()

# %% Kratke staze
shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
for node in central_nodes[:5]:  # Prikazujemo samo za prvih 5 centralnih zemalja
    print(f"Kratke staze od zemlje {node}:")
    for target, length in shortest_paths[node].items():
        print(f" - do {target}: {length}")

# %% Vizualizacija s centralnim čvorovima označenim crveno
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', width=weights)
nx.draw_networkx_nodes(G, pos, nodelist=central_nodes, node_color='red', node_size=700)
plt.title('Mreža društvenih veza između zemalja s označenim centralnim čvorovima')
plt.show()

# %% Predikcija budućih veza
# Preferencijalno povezivanje
predicted_edges = list(preferential_attachment(G))
predicted_edges.sort(key=lambda x: x[2], reverse=True)  # Sortiranje po verovatnoći

# Prikaz prvih 10 predviđenih veza
print("Prvih 10 predviđenih veza (preferencijalno povezivanje):")
for u, v, p in predicted_edges[:10]:
    print(f"{u} - {v}: {p}")

# Adamic/Adar indeks
predicted_edges_adamic = list(adamic_adar_index(G))
predicted_edges_adamic.sort(key=lambda x: x[2], reverse=True)  # Sortiranje po verovatnoći

# Prikaz prvih 10 predviđenih veza
print("Prvih 10 predviđenih veza (Adamic/Adar indeks):")
for u, v, p in predicted_edges_adamic[:10]:
    print(f"{u} - {v}: {p}")


# %% Detekcija zajednica - Louvainov metod
# Louvainova detekcija zajednica
partition = community_louvain.best_partition(G, weight='weight')

# Prikaz broja zajednica
num_communities = len(set(partition.values()))
print(f"Broj detektovanih zajednica: {num_communities}")

# Vizualizacija zajednica
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.15)
cmap = plt.get_cmap('viridis', num_communities)
nx.draw(G, pos, node_size=500, node_color=[cmap(partition[node]) for node in G.nodes()], with_labels=True, font_weight='bold')
plt.title('Mreža društvenih veza između zemalja s detektovanim zajednicama')
plt.show()
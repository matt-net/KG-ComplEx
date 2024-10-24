import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('books_graph_facts.csv')


# Initialize a directed graph
G = nx.DiGraph()

# # Add edges with relations as labels
# for _, row in df.iterrows():
#     G.add_edge(row['head'], row['tail'], label=row['relation'])
#
# # Position the nodes using spring layout
# pos = nx.spring_layout(G)
#
# # Plot the graph
# plt.figure(figsize=(10, 8))
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
#
# # Add edge labels (relation names)
# edge_labels = {(row['head'], row['tail']): row['relation'] for _, row in df.iterrows()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
#
# plt.show()


unique_relations = df['relation'].nunique()

# Count unique entities (combining head and tail)
unique_entities = pd.concat([df['head'], df['tail']]).nunique()

# Count total number of triples
total_triples = len(df)

# Get unique relations
unique_relations = df['relation'].unique()
print(f"Number of unique relations: {len(unique_relations)}")
print(f"Relations: {unique_relations}")

# Get unique entities (combining head and tail)
unique_entities = pd.concat([df['head'], df['tail']]).unique()
print(f"Number of unique entities: {len(unique_entities)}")
print(f"Entities: {unique_entities}")

# Count total number of triples
total_triples = len(df)
print(f"Total number of triples: {total_triples}")
import pandas as pd
from complEx import ComplEx
import torch
from utils import *

if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv('books_graph_facts.csv')
    # df = df.head(20)
    # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()

    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    # Create mappings for entities and relations to indices
    entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}

    # Convert triples into index-based tensors
    triples = []
    for _, row in df.iterrows():
        subject_id = entity2id[row['head']]
        relation_id = relation2id[row['relation']]
        object_id = entity2id[row['tail']]
        triples.append([subject_id, relation_id, object_id])

    triples = torch.tensor(triples)
    labels = torch.ones(len(triples), dtype=torch.float32)  # Positive labels

    # Generate negative triples and append to training data
    negative_triples = negative_sampling(triples, num_entities)
    all_triples = torch.cat([triples, negative_triples])
    all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negative_triples))])

    # Initialize model
    embedding_dim = 50  # Dimensionality of complex vectors
    model = ComplEx(num_entities, num_relations, embedding_dim)
    # Load the model and test triples
    model.load_state_dict(torch.load('complex_model.pth', weights_only=True))

    # Prepare your test triples (should be a tensor of shape [num_tests, 3])
    test_triples = torch.tensor([[entity2id['author_175'], relation2id['wrote'], entity2id['book_3593']]])

    # Calculate Filtered MRR
    mrr = filtered_mrr(model, test_triples, num_entities)
    print("Filtered MRR:", mrr)
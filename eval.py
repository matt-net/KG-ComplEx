from complEx import ComplEx
import pandas as pd
import torch
from utils import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('books_graph_facts.csv')
    # # df = df.head(20)
    # # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()
    #
    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}


    train_triples = torch.load('tensor_train_triples.pt')
    train_labels = torch.load('tensor_train_labels.pt')
    entity2id = torch.load('tensor_entity2id.pt')
    val_triples = torch.load('tensor_val_triples.pt')
    val_labels = torch.load('tensor_val_labels.pt')


    embedding_dim = 512  # Dimensionality of complex vectors
    model = ComplEx(num_entities, num_relations, embedding_dim)

    model.load_state_dict(torch.load(f'complex_model_{embedding_dim}.pth'))

    batch_size = 512
    k = 10

    hits = model.hits_at_k(val_triples, val_labels, batch_size, k)

    test_triples = torch.tensor([[entity2id['reader_2654'], relation2id['read'], entity2id['book_8']]])
    predictions = model.predict(test_triples)
    print("Predictions for test triples:", predictions)

    relation_real_embeddings = model.relation_real.weight.detach().cpu().numpy()
    relation_imag_embeddings = model.relation_imag.weight.detach().cpu().numpy()

    # Combine the real and imaginary embeddings
    combined_relation_embeddings = np.concatenate([relation_real_embeddings, relation_imag_embeddings], axis=1)

    entity_real, entity_imag = get_entity_embeddings(model)

    reduced_real_embeddings = reduce_dimensions_tsne(entity_real, n_components=2)
    reduced_imag_embeddings = reduce_dimensions_tsne(entity_imag, n_components=2)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_relation_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])

    for i, relation in enumerate(unique_relations):
        plt.text(pca_result[i, 0], pca_result[i, 1], relation, fontsize=9)

    plt.title('PCA of Relation Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

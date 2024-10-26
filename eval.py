from complEx import ComplEx
import pandas as pd
import torch
from utils import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # # Load the CSV
    df = pd.read_csv('books_graph_facts.csv')
    # # df = df.head(20)
    # # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()
    #
    num_entities = len(unique_entities)
    num_relations = len(unique_relations)
    #
    # # Create mappings for entities and relations to indices
    entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}

    # # Convert triples into index-based tensors
    # triples = []
    # for _, row in df.iterrows():
    #     subject_id = entity2id[row['head']]
    #     relation_id = relation2id[row['relation']]
    #     object_id = entity2id[row['tail']]
    #     triples.append([subject_id, relation_id, object_id])
    #
    # triples = torch.tensor(triples)
    # labels = torch.ones(len(triples), dtype=torch.float32)  # Positive labels

    # Generate negative triples and append to training data
    # negative_triples = negative_sampling(triples, num_entities, 0.1)
    # all_triples = torch.cat([triples, negative_triples])
    # all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negative_triples))])
    # #
    # tensors = {
    #     "all_triples": all_triples,
    #     "all_labels": all_labels,
    #     "triples": triples,
    #     "entity2id": entity2id,
    #     "relation2id": relation2id
    # }
    #
    # # Save each tensor with a descriptive filename
    # for name, tensor in tensors.items():
    #     torch.save(tensor, f'tensor_{name}.pt')

    train_triples = torch.load('tensor_train_triples.pt')
    train_labels = torch.load('tensor_train_labels.pt')
    entity2id = torch.load('tensor_entity2id.pt')
    val_triples = torch.load('tensor_val_triples.pt')
    val_labels = torch.load('tensor_val_labels.pt')


    embedding_dim = 512  # Dimensionality of complex vectors
    model = ComplEx(num_entities, num_relations, embedding_dim)

    # Initialize the model (same architecture and hyperparameters as when you saved it)

    # Load the saved state dictionary into the model
    model.load_state_dict(torch.load(f'complex_model_{embedding_dim}.pth'))

    batch_size = 512
    k = 10

    hits = model.hits_at_k(val_triples, val_labels, batch_size, k)

    test_triples = torch.tensor([[entity2id['reader_2654'], relation2id['read'], entity2id['book_8']]])
    predictions = model.predict(test_triples)
    print("Predictions for test triples:", predictions)

    # Assume the model has already been trained and relation embeddings are available
    relation_real_embeddings = model.relation_real.weight.detach().cpu().numpy()  # Get real part of relation embeddings
    relation_imag_embeddings = model.relation_imag.weight.detach().cpu().numpy()  # Get imaginary part of relation embeddings

    # Combine the real and imaginary embeddings
    combined_relation_embeddings = np.concatenate([relation_real_embeddings, relation_imag_embeddings], axis=1)

    # Step 1: Extract embeddings from your trained model
    entity_real, entity_imag = get_entity_embeddings(model)

    # Step 2: Reduce dimensions using t-SNE (can use real or imaginary part)
    reduced_real_embeddings = reduce_dimensions_tsne(entity_real, n_components=2)
    reduced_imag_embeddings = reduce_dimensions_tsne(entity_imag, n_components=2)

    # Step 3: Generate some labels (for example, each point gets a different label or relation)
    # Example of using some dummy labels (replace this with your actual labels)
    # If you don't have labels, you can cluster embeddings first or use random ones for visualization
    # labels = [0, 1, 2, 3, 4]  # Replace with actual labels corresponding to your embeddings
    #
    # # Step 4: Visualize the embeddings with color-coded labels
    # plot_with_labels(reduced_real_embeddings, labels, title="t-SNE of Entity Embeddings (Real Part)")
    # plot_with_labels(reduced_imag_embeddings, labels, title="t-SNE of Entity Embeddings (Imag Part)")

    # # Apply PCA to reduce dimensions (e.g., to 2 components)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_relation_embeddings)
    #
    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])

    # Label the points with relation names for better visualization
    for i, relation in enumerate(unique_relations):
        plt.text(pca_result[i, 0], pca_result[i, 1], relation, fontsize=9)

    plt.title('PCA of Relation Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

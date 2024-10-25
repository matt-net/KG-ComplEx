import numpy as np
import torch

class Negative_sampler:
    def __init__(self):
        self.negative_samples = None

    def corrupt_tail(self, triple, num_entities):
        """
        Corrupt the tail in the given graph by replacing it with a random entity.
        """
        head, relation, tail = triple

        # Generate a random tail which is not equal to the true tail entity
        corrupted_tail = tail
        while corrupted_tail == tail:
            corrupted_tail = torch.randint(0, num_entities, (1,)).item()  # Ensure it's a scalar

        return head, relation, corrupted_tail

    def generate_negative_samples(self, triples, num_samples, num_entities):
        """
        Generate negative samples by corrupting the tail of true triples.
        """
        # Convert the list of true triples into a set for faster membership checking
        true_triples_set = set(tuple(triple) for triple in triples)

        negative_samples = []
        while len(negative_samples) < num_samples:
            # Randomly picking a true triple
            true_triple = triples[np.random.randint(0, len(triples))]
            # Corrupt the tail
            negative_triple = self.corrupt_tail(true_triple, num_entities)

            # Ensure that the generated negative sample does not exist in the set of true triples
            if tuple(negative_triple) not in true_triples_set:
                negative_samples.append(torch.tensor(negative_triple))  # Store as tensor

        return torch.stack(negative_samples)

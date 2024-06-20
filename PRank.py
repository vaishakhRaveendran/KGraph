import numpy as np
from collections import defaultdict

def pagerank(adjacency_matrix, damping_factor=0.85, max_iterations=100, tolerance=1e-8):
    """
    Calculates the PageRank scores for nodes in a graph represented by an adjacency matrix.
    """
    # ... (implementation same as the previous example) ...

def find_dense_subgraphs(sentence_graph, similarity_threshold=0.5, num_clusters=3):
    """
    Finds dense subgraphs (clusters) in the sentence graph using PageRank.

    Args:
        sentence_graph (dict): A dictionary representing the sentence graph, where keys are sentence IDs,
                               and values are lists of tuples (neighbor_id, similarity_score).
        similarity_threshold (float, optional): The minimum similarity score to consider for PageRank edges. Defaults to 0.5.
        num_clusters (int, optional): The desired number of dense subgraphs (clusters) to return. Defaults to 3.

    Returns:
        list: A list of lists, where each inner list contains the sentence IDs of a dense subgraph (cluster).
    """
    num_nodes = len(sentence_graph)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Create the adjacency matrix from the sentence graph
    node_indices = {node_id: idx for idx, node_id in enumerate(sentence_graph.keys())}
    for node_id, neighbors in sentence_graph.items():
        node_idx = node_indices[node_id]
        for neighbor_id, similarity in neighbors:
            if similarity >= similarity_threshold:
                neighbor_idx = node_indices[neighbor_id]
                adjacency_matrix[node_idx, neighbor_idx] = similarity
                adjacency_matrix[neighbor_idx, node_idx] = similarity  # Assuming an undirected graph

    # Run PageRank algorithm
    pagerank_scores = pagerank(adjacency_matrix)

    # Sort nodes by their PageRank scores
    sorted_nodes = sorted([(score, node_id) for node_id, score in enumerate(pagerank_scores)], reverse=True)

    # Group nodes into dense subgraphs (clusters)
    dense_subgraphs = []
    current_cluster = []
    for score, node_idx in sorted_nodes:
        node_id = list(node_indices.keys())[list(node_indices.values()).index(node_idx)]
        if len(current_cluster) >= num_nodes // num_clusters:
            dense_subgraphs.append(current_cluster)
            current_cluster = [node_id]
        else:
            current_cluster.append(node_id)

    if current_cluster:
        dense_subgraphs.append(current_cluster)

    return dense_subgraphs

# Example usage
sentence_graph = {
    'sent1': [('sent2', 0.8), ('sent3', 0.6), ('sent4', 0.2)],
    'sent2': [('sent1', 0.8), ('sent3', 0.7), ('sent5', 0.4)],
    'sent3': [('sent1', 0.6), ('sent2', 0.7), ('sent6', 0.9)],
    'sent4': [('sent1', 0.2), ('sent5', 0.3)],
    'sent5': [('sent2', 0.4), ('sent4', 0.3), ('sent6', 0.5)],
    'sent6': [('sent3', 0.9), ('sent5', 0.5)]
}

dense_subgraphs = find_dense_subgraphs(sentence_graph, similarity_threshold=0.6, num_clusters=3)
print(dense_subgraphs)
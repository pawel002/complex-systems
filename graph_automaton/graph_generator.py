import networkx as nx

class RegularGraphGenerator:
    @staticmethod
    def _nx_to_adj_list(G, n_nodes):
        adj_list = [[] for _ in range(n_nodes)]
        for u, neighbors in G.adjacency():
            adj_list[u] = list(neighbors.keys())
        return adj_list

    @staticmethod
    def generate_ring_regular(n, k):
        G = nx.watts_strogatz_graph(n, k, p=0.0)
        return RegularGraphGenerator._nx_to_adj_list(G, n)

    @staticmethod
    def generate_toroidal_grid(rows, cols):
        n = rows * cols
        G = nx.grid_2d_graph(rows, cols, periodic=True)
        G = nx.convert_node_labels_to_integers(G)
        return RegularGraphGenerator._nx_to_adj_list(G, n)

    @staticmethod
    def generate_random_regular_connected(n, k):
        max_attempts = 100
        for _ in range(max_attempts):
            G = nx.random_regular_graph(k, n)
            if nx.is_connected(G):
                return RegularGraphGenerator._nx_to_adj_list(G, n)
            
        raise RuntimeError("Could not generate connected graph after multiple attempts.")

    @staticmethod
    def generate_hypercube(dimension):
        G = nx.hypercube_graph(dimension)
        n = 2**dimension
        G = nx.convert_node_labels_to_integers(G)
        return RegularGraphGenerator._nx_to_adj_list(G, n)
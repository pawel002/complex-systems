import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

class LayoutGenerator:
    @staticmethod
    def layout_grid_2d(rows, cols):
        n = rows * cols
        coords = np.zeros((n, 2))
        
        for r in range(rows):
            for c in range(cols):
                index = r * cols + c
                coords[index] = [c, -r] 
        return coords

    @staticmethod
    def layout_circular(n, radius=1.0):
        coords = np.zeros((n, 2))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        coords[:, 0] = radius * np.cos(theta)
        coords[:, 1] = radius * np.sin(theta)
        
        return coords
    
    @staticmethod
    def layout_random(n):
        return np.random.uniform(-1, 1, (n, 2))

    @staticmethod
    def layout_force_directed(adj_list, iterations=50):
        G = nx.Graph()
        n = len(adj_list)
        G.add_nodes_from(range(n))
        
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v:
                    G.add_edge(u, v)
        
        pos_dict = nx.spring_layout(G, iterations=iterations)
        coords = np.zeros((n, 2))
        for i in range(n):
            coords[i] = pos_dict[i]
        
        print("Completed finding graph positions")
        return coords

def plot_graph(ax, adj_list, coords, title="Graph"):
    
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if u < v:
                p1 = coords[u]
                p2 = coords[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                         color='gray', alpha=0.3, linewidth=0.5)
    
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c='blue', zorder=5)
    ax.set_title(title)
    ax.set_axis_off()

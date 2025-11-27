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

class VectorizedForceLayout:
    def __init__(self, adj_list, width=1000.0, height=1000.0, seed=42):
        self.adj_list = adj_list
        self.n = len(adj_list)
        self.width = width
        self.height = height
        
        np.random.seed(seed)
        self.pos = np.random.rand(self.n, 2) * width
        
        edge_pairs = []
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v: 
                    edge_pairs.append([u, v])
        self.edges = np.array(edge_pairs)
        
        self.area = width * height
        self.k = np.sqrt(self.area / self.n)
        
        self.iterations = 50
        self.initial_temp = width / 10.0
        
    def run(self, iterations=None):
        if iterations is None: iterations = self.iterations
        
        temp = self.initial_temp
        dt = temp / (iterations + 1)
        
        for i in range(iterations):
            delta = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
            dist_sq = np.sum(delta**2, axis=-1)
            
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(dist_sq)
            
            repulsion = (self.k**2 / dist_sq)[:, :, np.newaxis] * delta
            disp = np.sum(repulsion, axis=1)

            start_nodes = self.pos[self.edges[:, 0]]
            end_nodes = self.pos[self.edges[:, 1]]
            
            delta_e = start_nodes - end_nodes
            dist_e = np.linalg.norm(delta_e, axis=1)
            
            force_attr = (dist_e / self.k)[:, np.newaxis] * delta_e
            
            np.add.at(disp, self.edges[:, 0], -force_attr)
            np.add.at(disp, self.edges[:, 1], +force_attr)
            
            disp_len = np.linalg.norm(disp, axis=1)
            disp_len[disp_len < 0.0001] = 0.1 
            
            capped_disp = disp * (np.minimum(disp_len, temp) / disp_len)[:, np.newaxis]

            self.pos += capped_disp
            self.pos = np.clip(self.pos, 0, [self.width, self.height])
            
            temp -= dt
                
        return self.pos

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

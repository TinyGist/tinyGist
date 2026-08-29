import os
import pickle
import networkx as nx
import pathlib
import matplotlib.pyplot as plt
import logging
import matplotlib
import numpy as np

matplotlib.use("Agg")

log = logging.getLogger(__name__)


class CommunicationSimulator:
    def __init__(self, num_nodes=10, shape="mesh_full",
                 com_stability_mean=0.8, com_stability_std=0.1,
                 highest_stability=1.0, lowest_stability=1.0
                 ):
        if num_nodes <= 1:
            raise ValueError(
                f"Cannot create a graph, the number of nodes [{num_nodes}] is invalid"
            )
        self.__support_shapes = [
            "ring",
            "line",
            "star",
            "mesh_full",
            "tree_random",
            "tree_binary",
            "custom_clustered"
        ]
        if shape not in self.__support_shapes:
            raise ValueError(
                f"current target shape is [{shape}]\n"
                f"Not supported, currently supports {self.__support_shapes}"
            )

        self.__num_nodes = num_nodes
        self.__shape = shape
        self.__graph = None
        self.__adjacent_matrix = None
        self.__connectivity_dict = None

        self.__com_stability_mean = com_stability_mean
        self.__com_stability_std = com_stability_std

        if not 0 <= lowest_stability <= highest_stability <= 1:
            raise ValueError(
                "stability bounds must satisfy "
                "0 <= lowest_stability <= highest_stability <= 1"
            )
        self.__lowest_stability = lowest_stability
        self.__highest_stability = highest_stability

    def save_graph(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        if self.__graph is None:
            raise ValueError("No graph is saved")

        with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.__graph, f)

    def load_graph(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "rb") as f:
            self.__graph = pickle.load(f)
        self.__adjacent_matrix = None
        self.__connectivity_dict = None

    def __generate_graph_custom_clustered(self, num_cluster=2, num_connection=3, connectivity_p=1.0):
        num_nodes = self.__num_nodes
        if num_cluster >= num_nodes:
            num_cluster = num_nodes
            log.warning(f"num_cluster [{num_cluster}] is set to the number of existed nodes [{num_nodes}]")
        # split nodes nearly equally among clusters
        num_nodes_per_cluster = np.full(num_cluster, num_nodes // num_cluster, dtype=int)
        num_nodes_per_cluster[: num_nodes % num_cluster] += 1
        # 1) generate connected subgraphs (clusters)
        graph_list = []
        for number in num_nodes_per_cluster:
            # keep sampling until connected
            graph = nx.gnp_random_graph(int(number), p=connectivity_p)
            while graph.number_of_nodes() > 0 and not nx.is_connected(graph):
                graph = nx.gnp_random_graph(int(number), p=connectivity_p)
            graph_list.append(graph)
        # 2) relabel nodes to avoid collisions + record cluster membership
        new_graph_list = []
        for i, graph in enumerate(graph_list):
            new_graph = nx.relabel_nodes(graph, lambda x: f"g{i}_{x}")
            new_graph_list.append(new_graph)
        # 3) compose into a big graph
        graph_big = nx.compose_all(new_graph_list)
        # 4) add exactly `num_connection` bridge edges between each adjacent cluster pair (ring)
        if num_cluster > 1 and num_connection > 0:
            rng = np.random.default_rng()
            for i in range(num_cluster):
                g1 = new_graph_list[i]
                g2 = new_graph_list[(i + 1) % num_cluster]  # next cluster (ring)
                nodes1 = list(g1.nodes)
                nodes2 = list(g2.nodes)
                # cannot add more unique bridges than min(cluster sizes)
                k = min(int(num_connection), len(nodes1), len(nodes2))
                if k <= 0:
                    continue
                pick1 = rng.choice(nodes1, size=k, replace=False)
                pick2 = rng.choice(nodes2, size=k, replace=False)
                # add k inter-cluster edges
                for u, v in zip(pick1, pick2):
                    graph_big.add_edge(u, v)
                if num_cluster == 2:
                    break
        self.__graph = graph_big

    def __generate_graph_line(self):
        graph = nx.path_graph(self.__num_nodes)
        self.__graph = graph

    def __generate_graph_ring(self):
        graph = nx.cycle_graph(self.__num_nodes)
        self.__graph = graph

    def __generate_graph_star(self):
        graph = nx.star_graph(self.__num_nodes - 1)
        self.__graph = graph

    def __generate_graph_mesh_full(self):
        graph = nx.complete_graph(self.__num_nodes)
        self.__graph = graph

    def __generate_graph_tree_random(self):
        graph = nx.random_labeled_tree(self.__num_nodes)
        self.__graph = graph

    def __generate_graph_tree_binary(self):
        graph = nx.full_rary_tree(2, self.__num_nodes)
        self.__graph = graph

    def _generate_graph(self, shape=None):
        if shape is None:
            log.info(f"Currently used shape is {self.__shape}")
            target_shape = self.__shape
        else:
            if shape not in self.__support_shapes:
                raise ValueError(
                    f"current target shape is [{shape}]\n"
                    f"Not supported, currently supports {self.__support_shapes}"
                )
            log.info(f"Currently used shape is {shape}")
            self.__shape = shape
            target_shape = shape

        self.__adjacent_matrix = None
        self.__connectivity_dict = None

        if target_shape == "line":
            self.__generate_graph_line()
        elif target_shape == "ring":
            self.__generate_graph_ring()
        elif target_shape == "star":
            self.__generate_graph_star()
        elif target_shape == "mesh_full":
            self.__generate_graph_mesh_full()
        elif target_shape == "tree_random":
            self.__generate_graph_tree_random()
        elif target_shape == "tree_binary":
            self.__generate_graph_tree_binary()
        elif target_shape == "custom_clustered":
            self.__generate_graph_custom_clustered()

        num_edges = self.__graph.number_of_edges()
        edges = list(self.__graph.edges())

        gauss_rand_array = np.random.randn(num_edges)
        gauss_rand_array = gauss_rand_array * self.__com_stability_std
        gauss_rand_array = gauss_rand_array + self.__com_stability_mean
        gauss_rand_array[gauss_rand_array < 0] = 0
        gauss_rand_array[gauss_rand_array > 1] = 1

        gauss_rand_array[gauss_rand_array < self.__lowest_stability] = self.__lowest_stability
        gauss_rand_array[gauss_rand_array > self.__highest_stability] = self.__highest_stability

        np.random.shuffle(gauss_rand_array)

        for (u, v), stability in zip(edges, gauss_rand_array):
            self.__graph[u][v]["stability"] = float(stability)


    def get_adjacent_matrix(self, shape_regenerate=False, shape=None):
        if not shape_regenerate:
            if self.__adjacent_matrix is None:
                if self.__graph is None:
                    self._generate_graph(shape=shape)
                nodelist = sorted(self.__graph.nodes())
                self.__adjacent_matrix = nx.to_numpy_array(self.__graph, weight="stability", nodelist=nodelist)
        else:
            self._generate_graph(shape=shape)
            nodelist = sorted(self.__graph.nodes())
            self.__adjacent_matrix = nx.to_numpy_array(self.__graph, weight="stability", nodelist=nodelist)

        return self.__adjacent_matrix

    def visualize_graph(self, store_dir="./test/topology/", fig_width=20, fig_height=20):
        store_path = store_dir + "topology_shape_" + self.__shape + ".svg"
        pathlib.Path(store_path).parent.mkdir(parents=True, exist_ok=True)

        if self.__graph is None:
            self._generate_graph()

        pos = nx.spring_layout(self.__graph, seed=0)

        edges = list(self.__graph.edges())
        stabilities = np.array([self.__graph[u][v]['stability'] for u, v in edges])
        if stabilities.max() > stabilities.min():
            widths = 1 + 4 * (stabilities - stabilities.min()) / (stabilities.max() - stabilities.min())
        else:
            widths = np.ones_like(stabilities)

        plt.figure(figsize=(fig_width, fig_height))
        nx.draw_networkx_nodes(self.__graph, pos, node_size=1500, node_color="blue", edgecolors="black")
        # nx.draw_networkx_labels(self.__graph, pos, font_size=30)
        nx.draw_networkx_edges(
            self.__graph,
            pos,
            edgelist=edges,
            edge_color=stabilities,
            width=widths*10,
            edge_cmap=plt.cm.viridis,
            edge_vmin=self.__lowest_stability,
            edge_vmax=self.__highest_stability,
        )
        edge_labels = {(u, v): f"{self.__graph[u][v]['stability']:.2f}" for u, v in edges}
        # nx.draw_networkx_edge_labels(self.__graph, pos, edge_labels=edge_labels, font_size=30)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(store_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_connectivity_dict(self, prefix_name="device_", pos_regenerate=False, shape_regenerate=False, shape=None, random_mapping=False):
        if self.__graph is None or shape_regenerate:
            self._generate_graph(shape=shape)

        if self.__connectivity_dict is None or pos_regenerate or shape_regenerate:
            self.__connectivity_dict = dict()
            if random_mapping:
                permuted_device_idx_list = np.random.permutation(self.__num_nodes).tolist()
            else:
                permuted_device_idx_list = np.arange(self.__num_nodes).tolist()
            # Keep the graph's internal node ids stable. Relabeling the graph in
            # place made the second dynamic remapping try to map device ids as if
            # they were the original integer nodes, corrupting the topology.
            try:
                internal_nodes = sorted(self.__graph.nodes())
            except TypeError:
                internal_nodes = sorted(self.__graph.nodes(), key=lambda node: repr(node))
            mapping = {
                node: f"{prefix_name}{device_idx}"
                for node, device_idx in zip(internal_nodes, permuted_device_idx_list)
            }
            self.__connectivity_dict = {
                external_node: {} for external_node in mapping.values()
            }
            for u, v, edge_data in self.__graph.edges(data=True):
                external_u = mapping[u]
                external_v = mapping[v]
                stability = float(edge_data['stability'])
                self.__connectivity_dict[external_u][external_v] = stability
                self.__connectivity_dict[external_v][external_u] = stability

        return self.__connectivity_dict


if __name__ == '__main__':
    AMG = CommunicationSimulator()
    support_shapes = [
        "ring",
        "line",
        "star",
        "mesh_full",
        "tree_random",
        "tree_binary",
        "custom_clustered"
    ]
    #AMG.visualize_graph()

    for shape in support_shapes:
        AMG._generate_graph(shape=shape)
        print(AMG.get_adjacent_matrix(shape_regenerate=True))
        print(AMG.get_connectivity_dict(shape_regenerate=True, shape=shape, pos_regenerate=True))
        AMG.visualize_graph()

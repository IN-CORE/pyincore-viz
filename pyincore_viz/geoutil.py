# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import networkx as nx
import matplotlib.pyplot as plt

class GeoUtil:
    """Utility methods for georeferenced data."""

    @staticmethod
    def plot_graph_network(graph, coords):
        """Plot graph.

        Args:
            graph (obj):  A nx graph to be drawn.
            coords (dict): Position coordinates.

        """
        # nx.draw(graph, coords, with_lables=True, font_weithg='bold')

        # other ways to draw
        nx.draw_networkx_nodes(graph, coords, cmap=plt.get_cmap('jet'), node_size=100, node_color='g', with_lables=True, font_weithg='bold')
        nx.draw_networkx_labels(graph, coords)
        nx.draw_networkx_edges(graph, coords, edge_color='r', arrows=True)
        plt.show()

    @staticmethod
    def get_network_graph(filename, is_directed=False):
        """Get network graph from filename.

        Args:
            filename (str):  A name of a geo dataset resource recognized by Fiona package.
            is_directed (bool, optional (Defaults to False)): Graph type. True for directed Graph,
                False for Graph.

        Returns:
            obj, dict: Graph and node coordinates.

        """
        geom = nx.read_shp(filename)
        node_coords = {k: v for k, v in enumerate(geom.nodes())}
        # create graph
        graph = None
        if is_directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        graph.add_nodes_from(node_coords.keys())
        l = [set(x) for x in geom.edges()]
        edg = [tuple(k for k, v in node_coords.items() if v in sl) for sl in l]

        graph.add_edges_from(edg)

        return graph, node_coords



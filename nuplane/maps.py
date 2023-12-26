import os

from pathlib import Path

import networkx as nx
import osmnx as ox

from dataclasses import dataclass, field

from typing import (
    Collection,
    Dict,
    List,
    Union,
)


from xplane_airports.AptDat import AptDatLine, TaxiRouteEdge, RowCode, AptDat

from .utils.osm_writer import OSMDataWriter
from .utils.transform import haversine_distance
from .utils.airport_parser import parse_airport_taxi_network


WED_LINE_ENDING = '\n'

TAXI_WIDTH = {'A': 4.5, 'B': 6, 'C': 9, 'D': 14, 'E': 14, 'F': 16}


class AptDatModified(AptDat):
    def write_to_disk(self, path_to_write_to, airport_to_write):
        """
        Writes a complete apt.dat file containing this entire collection of airports.
        :param path_to_write_to: A complete file path (ending in .dat); if None, we'll use the path we read this apt.dat in from
        """
        if not path_to_write_to:
            path_to_write_to = self.path_to_file
        assert (
            path_to_write_to and Path(path_to_write_to).suffix == '.dat'
        ), f"Invalid apt.dat path: {path_to_write_to}"
        with Path(path_to_write_to).expanduser().open('w', encoding="utf8") as f:
            f.write("I" + WED_LINE_ENDING)
            f.write(
                f"{self.xplane_version} Generated by WorldEditor{WED_LINE_ENDING}{WED_LINE_ENDING}"
            )

            for pred in [self.search_by_id, self.search_by_name]:
                result = pred(airport_to_write)
                if result:
                    break

            f.write(str(result))
            f.write(WED_LINE_ENDING * 2)
            f.write(str(RowCode.FILE_END) + WED_LINE_ENDING)


@dataclass
class GateNode:
    """
    A node in a taxiway routing network, used for routing aircraft via ATC.
    Every node must be part of one or more edges.
    Note that taxi routing networks (beginning with TAXI_ROUTE_HEADER line types)
    may or may not correspond to taxiway pavement.
    """

    name: str  # The node identifier (must be unique within an airport)
    lon: float  # Node's longitude
    lat: float  # Node's latitude
    heading: float  # Node's  heading for parking


@dataclass
class TaxiRouteNode:
    """
    A node in a taxiway routing network, used for routing aircraft via ATC.
    Every node must be part of one or more edges.
    Note that taxi routing networks (beginning with TAXI_ROUTE_HEADER line types)
    may or may not correspond to taxiway pavement.
    """

    id: int  # The node identifier (must be unique within an airport)
    lon: float  # Node's longitude
    lat: float  # Node's latitude


@dataclass
class TaxiRouteNetwork:
    nodes: Dict[int, TaxiRouteNode] = field(default_factory=dict)
    edges: List[TaxiRouteEdge] = field(default_factory=list)

    @staticmethod
    def from_lines(apt_dat_lines: Collection[AptDatLine]) -> 'TaxiRouteNetwork':
        return TaxiRouteNetwork.from_tokenized_lines(
            [line.tokens for line in apt_dat_lines if not line.is_ignorable()]
        )

    @staticmethod
    def from_tokenized_lines(
        tokenized_lines: Collection[List[Union[RowCode, str]]]
    ) -> 'TaxiRouteNetwork':
        nodes = {
            node.id: node
            for node in map(
                lambda tokens: TaxiRouteNode(
                    id=int(tokens[4]),
                    lon=float(tokens[2]),
                    lat=float(tokens[1]),
                ),
                filter(
                    lambda line: line[0] == RowCode.TAXI_ROUTE_NODE, tokenized_lines
                ),
            )
        }

        edges = [
            TaxiRouteEdge.from_tokenized_line(tokens)
            for tokens in tokenized_lines
            if tokens[0] == RowCode.TAXI_ROUTE_EDGE
        ]
        return TaxiRouteNetwork(nodes=nodes, edges=edges)


class Map:
    def __init__(self, client, config) -> None:
        self.client = client
        self.osm_writer = OSMDataWriter()
        self._setup(config)
        return None

    def _setup(self, config):
        airport = config['experiment']['experiment_config']['airport']
        self._parse_taxi_network(config, airport)

    def _setup_depreciated(self, config):
        """Perform the initial experiment setup e.g., loading the map

        Parameters
        ----------
        experiment_config : yaml
            A yaml file providing the map configuration

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the experiment config is none, raises a file not found error
        """
        airport = config['experiment']['experiment_config']['airport']

        # Check if .apt data already exist
        if not Path(f'data/{airport}.dat').is_file():
            default_xplane_apt_dat = AptDatModified(
                config['xplane_server']['xplane_path']
                + '/Resources/default scenery/default apt dat/Earth nav data/apt.dat'
            )
            default_xplane_apt_dat.write_to_disk(
                f'data/{airport}.dat', airport_to_write=airport
            )

        lines = open(f'data/{airport}.dat', 'r').read()
        apt_lines = [AptDatLine(line) for line in lines.splitlines()]

        self.taxi_network = TaxiRouteNetwork.from_lines(apt_lines)
        self._setup_gates(apt_lines)
        self._convert_taxi_network_to_osmnx_graph(self.taxi_network, airport)
        return None

    def _parse_taxi_network(self, config, airport):
        # Parse the nodes and edges
        self.node_graph = parse_airport_taxi_network(airport)

    def _setup_gates(self, apt_lines):
        self.gates = {}
        lines = [line.tokens for line in apt_lines if not line.is_ignorable()]
        for line in lines:
            if line[0] == RowCode.START_LOCATION_NEW:
                self.gates[line[-1]] = {
                    'lat': float(line[1]),
                    'lon': float(line[2]),
                    'heading': float(line[3]),
                }

    def _convert_taxi_network_to_osmnx_graph(self, taxi_network, airport):
        nodes = [
            (node[1].id, {'lat': float(node[1].lat), 'lon': float(node[1].lon)})
            for node in taxi_network.nodes.items()
        ]
        bools = ('no', 'yes')
        edges = [
            (
                edge.node_begin,
                edge.node_end,
                {
                    'is_runway': bools[edge.is_runway],
                    'one_way': bools[edge.one_way],
                    'width': str(TAXI_WIDTH[edge.icao_width]),
                    'length': str(
                        haversine_distance(
                            nodes[edge.node_begin][1], nodes[edge.node_end][1]
                        )
                    ),
                },
            )
            for edge in taxi_network.edges
        ]
        # Check if path exists
        if not os.path.isfile(f'data/{airport}.osm'):
            self.osm_writer.write(f'data/{airport}.osm', nodes, edges)
            self.osm_writer.file_close()

        # Read the graph
        G = ox.graph_from_xml(f'data/{airport}.osm', simplify=False, retain_all=True)
        self.node_graph = nx.convert_node_labels_to_integers(G)

    def find_shortest_path(self, start, end, weight='length'):
        route = nx.shortest_path(self.node_graph, start, end, weight=weight)
        return route

    def draw_map(self, node_size=15, with_labels=False):
        ox.plot_graph(self.node_graph)

    def get_node_graph(self):
        """Get the node graph of the world

        Returns
        -------
        networkx graph
            A node graph of the world map
        """
        return self.node_graph

    def get_node_info(self, node_index):
        """Get the information about a node.

        Parameters
        ----------
        id : int
            Node ID

        Returns
        -------
        dict
            A dictionary containing all the information about the node.
        """
        if isinstance(node_index, list):
            node_index = node_index[0]
        return self.node_graph.nodes[node_index]

    def convert_lat_lon_node_number(self, lat_lon):
        """Get the latitude and longitude spawn points

        Parameters
        ----------
        n_points : int, optional
            Number of points to random latitude and longitude points, by default 5

        Returns
        -------
        array
            An array of cartesian spawn points
        """
        raise NotImplementedError

    def convert_node_number_to_lat_lon(self, node_num):
        """Get the latitude and longitude spawn points

        Parameters
        ----------
        n_points : int, optional
            Number of points to random latitude and longitude points, by default 5

        Returns
        -------
        array
            An array of cartesian spawn points
        """
        if isinstance(node_num, list):
            node_num = node_num[0]
        return self.node_graph.nodes[node_num]
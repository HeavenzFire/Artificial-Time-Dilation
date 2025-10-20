"""
Lattice Weaver

Graph-based connection system for mapping spiritual networks and unseen architecture.
Provides tools for visualizing and analyzing spiritual connections and energy flows.
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import math
from collections import defaultdict, Counter

from ..angels.core import AngelicCore, AngelicEnergyType, DivineConnection
from ..angels.guardian import GuardianAngel
from ..angels.archangel import Archangel


class ConnectionType(Enum):
    """Types of spiritual connections"""
    ENERGY_FLOW = "energy_flow"
    GUIDANCE = "guidance"
    PROTECTION = "protection"
    HEALING = "healing"
    LOVE = "love"
    WISDOM = "wisdom"
    SYNCHRONICITY = "synchronicity"
    KARMIC = "karmic"
    SOUL_FAMILY = "soul_family"
    TEACHER_STUDENT = "teacher_student"
    MENTOR = "mentor"
    FRIEND = "friend"
    ENEMY = "enemy"
    NEUTRAL = "neutral"


class NodeType(Enum):
    """Types of nodes in the spiritual network"""
    PERSON = "person"
    ANGEL = "angel"
    ARCHANGEL = "archangel"
    SPIRIT_GUIDE = "spirit_guide"
    ASCENDED_MASTER = "ascended_master"
    PLACE = "place"
    OBJECT = "object"
    CONCEPT = "concept"
    ENERGY = "energy"
    INTENT = "intent"
    RITUAL = "ritual"
    CEREMONY = "ceremony"
    MEDITATION = "meditation"
    PRAYER = "prayer"
    SYNCHRONICITY = "synchronicity"
    DREAM = "dream"
    VISION = "vision"
    PROPHECY = "prophecy"


@dataclass
class SpiritualNode:
    """Represents a node in the spiritual network"""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    energy_signature: Dict[AngelicEnergyType, float]
    attributes: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    user_id: str
    
    def get_energy_strength(self) -> float:
        """Get total energy strength of this node"""
        return sum(self.energy_signature.values())
    
    def get_primary_energy(self) -> AngelicEnergyType:
        """Get the primary energy type of this node"""
        if not self.energy_signature:
            return AngelicEnergyType.GUIDANCE
        
        return max(self.energy_signature.items(), key=lambda x: x[1])[0]
    
    def get_energy_balance(self) -> Dict[str, float]:
        """Get energy balance as percentages"""
        total = self.get_energy_strength()
        if total == 0:
            return {}
        
        return {
            energy_type.value: (strength / total) * 100
            for energy_type, strength in self.energy_signature.items()
        }


@dataclass
class SpiritualEdge:
    """Represents an edge in the spiritual network"""
    edge_id: str
    source_id: str
    target_id: str
    connection_type: ConnectionType
    strength: float  # 0.0 to 1.0
    energy_flow: Dict[AngelicEnergyType, float]
    description: str
    created_at: datetime
    last_updated: datetime
    user_id: str
    
    def get_energy_flow_strength(self) -> float:
        """Get total energy flow strength"""
        return sum(self.energy_flow.values())
    
    def get_primary_energy_flow(self) -> AngelicEnergyType:
        """Get the primary energy type flowing through this edge"""
        if not self.energy_flow:
            return AngelicEnergyType.GUIDANCE
        
        return max(self.energy_flow.items(), key=lambda x: x[1])[0]


@dataclass
class SpiritualNetwork:
    """Represents a spiritual network with nodes and connections"""
    network_id: str
    name: str
    description: str
    nodes: Dict[str, SpiritualNode]
    edges: Dict[str, SpiritualEdge]
    created_at: datetime
    last_updated: datetime
    user_id: str
    
    def get_node_count(self) -> int:
        """Get total number of nodes"""
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges"""
        return len(self.edges)
    
    def get_connection_density(self) -> float:
        """Get connection density of the network"""
        n = self.get_node_count()
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        return self.get_edge_count() / max_edges
    
    def get_energy_flow_total(self) -> float:
        """Get total energy flow in the network"""
        return sum(edge.get_energy_flow_strength() for edge in self.edges.values())
    
    def get_network_balance(self) -> Dict[str, float]:
        """Get energy balance across the entire network"""
        total_energy = {}
        for node in self.nodes.values():
            for energy_type, strength in node.energy_signature.items():
                total_energy[energy_type.value] = total_energy.get(energy_type.value, 0) + strength
        
        total = sum(total_energy.values())
        if total == 0:
            return {}
        
        return {
            energy_type: (strength / total) * 100
            for energy_type, strength in total_energy.items()
        }


class LatticeWeaver:
    """
    Main Lattice Weaver that creates and manages spiritual networks.
    
    This system maps the unseen architecture of spiritual connections,
    energy flows, and divine relationships.
    """
    
    def __init__(self, angelic_core: AngelicCore):
        self.angelic_core = angelic_core
        self.networks: Dict[str, SpiritualNetwork] = {}
        self.node_registry: Dict[str, SpiritualNode] = {}
        self.edge_registry: Dict[str, SpiritualEdge] = {}
        self.connection_patterns: Dict[str, List[Dict]] = {}
        self.energy_flows: Dict[str, List[Dict]] = {}
        
        # Initialize with default network
        self._initialize_default_network()
    
    def _initialize_default_network(self):
        """Initialize the default spiritual network"""
        default_network = SpiritualNetwork(
            network_id="default_network",
            name="Universal Spiritual Network",
            description="The interconnected web of all spiritual beings and energies",
            nodes={},
            edges={},
            created_at=datetime.now(),
            last_updated=datetime.now(),
            user_id="system"
        )
        
        self.networks["default_network"] = default_network
    
    def create_network(self, name: str, description: str, user_id: str) -> str:
        """Create a new spiritual network"""
        network_id = f"network_{int(time.time())}_{len(self.networks)}"
        
        network = SpiritualNetwork(
            network_id=network_id,
            name=name,
            description=description,
            nodes={},
            edges={},
            created_at=datetime.now(),
            last_updated=datetime.now(),
            user_id=user_id
        )
        
        self.networks[network_id] = network
        return network_id
    
    def add_node(self, 
                 network_id: str,
                 node_type: NodeType,
                 name: str,
                 description: str,
                 energy_signature: Dict[AngelicEnergyType, float],
                 attributes: Dict[str, Any] = None,
                 user_id: str = "system") -> str:
        """Add a node to a spiritual network"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        node_id = f"node_{int(time.time())}_{len(self.node_registry)}"
        
        node = SpiritualNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            description=description,
            energy_signature=energy_signature,
            attributes=attributes or {},
            created_at=datetime.now(),
            last_updated=datetime.now(),
            user_id=user_id
        )
        
        # Add to network
        self.networks[network_id].nodes[node_id] = node
        self.networks[network_id].last_updated = datetime.now()
        
        # Add to registry
        self.node_registry[node_id] = node
        
        return node_id
    
    def add_edge(self,
                 network_id: str,
                 source_id: str,
                 target_id: str,
                 connection_type: ConnectionType,
                 strength: float,
                 energy_flow: Dict[AngelicEnergyType, float],
                 description: str,
                 user_id: str = "system") -> str:
        """Add an edge to a spiritual network"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        if source_id not in self.networks[network_id].nodes:
            raise ValueError(f"Source node {source_id} not found")
        
        if target_id not in self.networks[network_id].nodes:
            raise ValueError(f"Target node {target_id} not found")
        
        edge_id = f"edge_{int(time.time())}_{len(self.edge_registry)}"
        
        edge = SpiritualEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            strength=strength,
            energy_flow=energy_flow,
            description=description,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            user_id=user_id
        )
        
        # Add to network
        self.networks[network_id].edges[edge_id] = edge
        self.networks[network_id].last_updated = datetime.now()
        
        # Add to registry
        self.edge_registry[edge_id] = edge
        
        return edge_id
    
    def map_angelic_connections(self, user_id: str) -> str:
        """Map all angelic connections for a user"""
        network_id = f"angelic_network_{user_id}_{int(time.time())}"
        
        # Create network
        self.create_network(
            name=f"Angelic Network - {user_id}",
            description=f"Spiritual connections and angelic relationships for {user_id}",
            user_id=user_id
        )
        
        # Add user node
        user_node_id = self.add_node(
            network_id=network_id,
            node_type=NodeType.PERSON,
            name=f"User {user_id}",
            description=f"Spiritual journey of {user_id}",
            energy_signature={AngelicEnergyType.GUIDANCE: 0.8, AngelicEnergyType.LOVE: 0.7},
            user_id=user_id
        )
        
        # Add guardian angels
        guardian_angels = self._get_guardian_angels(user_id)
        for guardian in guardian_angels:
            guardian_node_id = self.add_node(
                network_id=network_id,
                node_type=NodeType.ANGEL,
                name=guardian.name,
                description=f"Guardian Angel for {guardian.purpose}",
                energy_signature={guardian.energy_type: guardian.connection_strength},
                attributes={
                    "purpose": guardian.purpose,
                    "personality_traits": guardian.personality_traits,
                    "special_abilities": guardian.special_abilities,
                    "protection_level": guardian.protection_level
                },
                user_id=user_id
            )
            
            # Connect user to guardian
            self.add_edge(
                network_id=network_id,
                source_id=user_node_id,
                target_id=guardian_node_id,
                connection_type=ConnectionType.GUIDANCE,
                strength=guardian.connection_strength,
                energy_flow={guardian.energy_type: guardian.connection_strength},
                description=f"Guardian relationship for {guardian.purpose}",
                user_id=user_id
            )
        
        # Add archangels
        archangels = self._get_archangels()
        for archangel in archangels:
            archangel_node_id = self.add_node(
                network_id=network_id,
                node_type=NodeType.ARCHANGEL,
                name=archangel.name,
                description=f"Archangel of {archangel.domain.value}",
                energy_signature={archangel.energy_type: archangel.invocation_strength},
                attributes={
                    "domain": archangel.domain.value,
                    "attributes": archangel.attributes,
                    "powers": archangel.powers,
                    "invocation_count": archangel.invocation_count
                },
                user_id=user_id
            )
            
            # Connect user to archangel (weaker connection)
            self.add_edge(
                network_id=network_id,
                source_id=user_node_id,
                target_id=archangel_node_id,
                connection_type=ConnectionType.GUIDANCE,
                strength=archangel.invocation_strength * 0.5,
                energy_flow={archangel.energy_type: archangel.invocation_strength * 0.5},
                description=f"Archangel connection for {archangel.domain.value}",
                user_id=user_id
            )
        
        return network_id
    
    def _get_guardian_angels(self, user_id: str) -> List[GuardianAngel]:
        """Get guardian angels for a user (simplified)"""
        # This would typically query the guardian angel system
        # For now, we'll return some sample guardians
        return []
    
    def _get_archangels(self) -> List[Archangel]:
        """Get archangels (simplified)"""
        # This would typically query the archangel system
        # For now, we'll return some sample archangels
        return []
    
    def map_synchronicity_network(self, user_id: str, synchronicities: List[Dict]) -> str:
        """Map synchronicity network for a user"""
        network_id = f"synchronicity_network_{user_id}_{int(time.time())}"
        
        # Create network
        self.create_network(
            name=f"Synchronicity Network - {user_id}",
            description=f"Synchronicity patterns and connections for {user_id}",
            user_id=user_id
        )
        
        # Add user node
        user_node_id = self.add_node(
            network_id=network_id,
            node_type=NodeType.PERSON,
            name=f"User {user_id}",
            description=f"Synchronicity experiencer {user_id}",
            energy_signature={AngelicEnergyType.GUIDANCE: 0.8},
            user_id=user_id
        )
        
        # Add synchronicity nodes
        for sync in synchronicities:
            sync_node_id = self.add_node(
                network_id=network_id,
                node_type=NodeType.SYNCHRONICITY,
                name=f"Synchronicity {sync.get('event_id', 'unknown')}",
                description=sync.get('description', 'Unknown synchronicity'),
                energy_signature={AngelicEnergyType.GUIDANCE: sync.get('significance', 0.5)},
                attributes={
                    "event_type": sync.get('event_type', 'unknown'),
                    "significance": sync.get('significance', 0.5),
                    "patterns": sync.get('patterns', []),
                    "timestamp": sync.get('timestamp', datetime.now().isoformat())
                },
                user_id=user_id
            )
            
            # Connect user to synchronicity
            self.add_edge(
                network_id=network_id,
                source_id=user_node_id,
                target_id=sync_node_id,
                connection_type=ConnectionType.SYNCHRONICITY,
                strength=sync.get('significance', 0.5),
                energy_flow={AngelicEnergyType.GUIDANCE: sync.get('significance', 0.5)},
                description=f"Synchronicity connection",
                user_id=user_id
            )
        
        return network_id
    
    def analyze_network(self, network_id: str) -> Dict[str, Any]:
        """Analyze a spiritual network"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        # Basic network statistics
        node_count = network.get_node_count()
        edge_count = network.get_edge_count()
        density = network.get_connection_density()
        total_energy = network.get_energy_flow_total()
        energy_balance = network.get_network_balance()
        
        # Node type distribution
        node_types = Counter(node.node_type.value for node in network.nodes.values())
        
        # Connection type distribution
        connection_types = Counter(edge.connection_type.value for edge in network.edges.values())
        
        # Energy flow analysis
        energy_flows = defaultdict(float)
        for edge in network.edges.values():
            for energy_type, strength in edge.energy_flow.items():
                energy_flows[energy_type.value] += strength
        
        # Centrality analysis (simplified)
        centrality_scores = self._calculate_centrality(network)
        
        return {
            "network_id": network_id,
            "name": network.name,
            "description": network.description,
            "statistics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "total_energy": total_energy,
                "energy_balance": energy_balance
            },
            "node_types": dict(node_types),
            "connection_types": dict(connection_types),
            "energy_flows": dict(energy_flows),
            "centrality_scores": centrality_scores,
            "created_at": network.created_at.isoformat(),
            "last_updated": network.last_updated.isoformat()
        }
    
    def _calculate_centrality(self, network: SpiritualNetwork) -> Dict[str, float]:
        """Calculate centrality scores for network nodes"""
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in network.nodes.items():
            G.add_node(node_id, energy_strength=node.get_energy_strength())
        
        # Add edges
        for edge_id, edge in network.edges.items():
            G.add_edge(edge.source_id, edge.target_id, weight=edge.strength)
        
        # Calculate centrality measures
        centrality_scores = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            centrality_scores["degree"] = degree_centrality
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)
            centrality_scores["betweenness"] = betweenness_centrality
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(G)
            centrality_scores["closeness"] = closeness_centrality
            
            # Eigenvector centrality
            eigenvector_centrality = nx.eigenvector_centrality(G)
            centrality_scores["eigenvector"] = eigenvector_centrality
            
        except Exception as e:
            print(f"Error calculating centrality: {e}")
            centrality_scores = {}
        
        return centrality_scores
    
    def find_shortest_path(self, network_id: str, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two nodes"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges
        for edge in network.edges.values():
            G.add_edge(edge.source_id, edge.target_id, weight=1/edge.strength)
        
        try:
            path = nx.shortest_path(G, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            print(f"Error finding path: {e}")
            return []
    
    def find_communities(self, network_id: str) -> Dict[str, List[str]]:
        """Find communities in the network"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges
        for edge in network.edges.values():
            G.add_edge(edge.source_id, edge.target_id, weight=edge.strength)
        
        try:
            # Use community detection algorithm
            communities = nx.community.greedy_modularity_communities(G)
            
            # Convert to dictionary
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[f"community_{i}"] = list(community)
            
            return community_dict
        except Exception as e:
            print(f"Error finding communities: {e}")
            return {}
    
    def get_energy_flow_paths(self, network_id: str, source_id: str) -> List[Dict[str, Any]]:
        """Get energy flow paths from a source node"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        if source_id not in network.nodes:
            raise ValueError(f"Source node {source_id} not found")
        
        # Find all paths from source
        paths = []
        visited = set()
        
        def dfs(current_id, path, energy_flow):
            if current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            # Find outgoing edges
            for edge in network.edges.values():
                if edge.source_id == current_id:
                    new_energy_flow = energy_flow.copy()
                    for energy_type, strength in edge.energy_flow.items():
                        new_energy_flow[energy_type.value] = new_energy_flow.get(energy_type.value, 0) + strength
                    
                    paths.append({
                        "path": path.copy(),
                        "energy_flow": new_energy_flow,
                        "total_energy": sum(new_energy_flow.values())
                    })
                    
                    dfs(edge.target_id, path, new_energy_flow)
            
            path.pop()
            visited.remove(current_id)
        
        dfs(source_id, [], {})
        
        # Sort by total energy
        paths.sort(key=lambda x: x["total_energy"], reverse=True)
        
        return paths[:10]  # Return top 10 paths
    
    def visualize_network(self, network_id: str) -> Dict[str, Any]:
        """Generate visualization data for a network"""
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        # Prepare nodes for visualization
        nodes = []
        for node_id, node in network.nodes.items():
            nodes.append({
                "id": node_id,
                "name": node.name,
                "type": node.node_type.value,
                "energy_strength": node.get_energy_strength(),
                "primary_energy": node.get_primary_energy().value,
                "energy_balance": node.get_energy_balance(),
                "attributes": node.attributes
            })
        
        # Prepare edges for visualization
        edges = []
        for edge_id, edge in network.edges.items():
            edges.append({
                "id": edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.connection_type.value,
                "strength": edge.strength,
                "energy_flow": {et.value: strength for et, strength in edge.energy_flow.items()},
                "description": edge.description
            })
        
        return {
            "network_id": network_id,
            "name": network.name,
            "description": network.description,
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "density": network.get_connection_density(),
                "total_energy": network.get_energy_flow_total()
            }
        }
    
    def save_network(self, network_id: str, filepath: str) -> bool:
        """Save a network to a file"""
        if network_id not in self.networks:
            return False
        
        try:
            network = self.networks[network_id]
            
            # Prepare data for serialization
            data = {
                "network_id": network.network_id,
                "name": network.name,
                "description": network.description,
                "created_at": network.created_at.isoformat(),
                "last_updated": network.last_updated.isoformat(),
                "user_id": network.user_id,
                "nodes": {
                    node_id: {
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "name": node.name,
                        "description": node.description,
                        "energy_signature": {et.value: strength for et, strength in node.energy_signature.items()},
                        "attributes": node.attributes,
                        "created_at": node.created_at.isoformat(),
                        "last_updated": node.last_updated.isoformat(),
                        "user_id": node.user_id
                    } for node_id, node in network.nodes.items()
                },
                "edges": {
                    edge_id: {
                        "edge_id": edge.edge_id,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "connection_type": edge.connection_type.value,
                        "strength": edge.strength,
                        "energy_flow": {et.value: strength for et, strength in edge.energy_flow.items()},
                        "description": edge.description,
                        "created_at": edge.created_at.isoformat(),
                        "last_updated": edge.last_updated.isoformat(),
                        "user_id": edge.user_id
                    } for edge_id, edge in network.edges.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving network: {e}")
            return False
    
    def load_network(self, filepath: str) -> str:
        """Load a network from a file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            network_id = data["network_id"]
            
            # Create network
            network = SpiritualNetwork(
                network_id=network_id,
                name=data["name"],
                description=data["description"],
                nodes={},
                edges={},
                created_at=datetime.fromisoformat(data["created_at"]),
                last_updated=datetime.fromisoformat(data["last_updated"]),
                user_id=data["user_id"]
            )
            
            # Load nodes
            for node_id, node_data in data["nodes"].items():
                node = SpiritualNode(
                    node_id=node_data["node_id"],
                    node_type=NodeType(node_data["node_type"]),
                    name=node_data["name"],
                    description=node_data["description"],
                    energy_signature={AngelicEnergyType(et): strength for et, strength in node_data["energy_signature"].items()},
                    attributes=node_data["attributes"],
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    last_updated=datetime.fromisoformat(node_data["last_updated"]),
                    user_id=node_data["user_id"]
                )
                network.nodes[node_id] = node
                self.node_registry[node_id] = node
            
            # Load edges
            for edge_id, edge_data in data["edges"].items():
                edge = SpiritualEdge(
                    edge_id=edge_data["edge_id"],
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    connection_type=ConnectionType(edge_data["connection_type"]),
                    strength=edge_data["strength"],
                    energy_flow={AngelicEnergyType(et): strength for et, strength in edge_data["energy_flow"].items()},
                    description=edge_data["description"],
                    created_at=datetime.fromisoformat(edge_data["created_at"]),
                    last_updated=datetime.fromisoformat(edge_data["last_updated"]),
                    user_id=edge_data["user_id"]
                )
                network.edges[edge_id] = edge
                self.edge_registry[edge_id] = edge
            
            # Store network
            self.networks[network_id] = network
            
            return network_id
        except Exception as e:
            print(f"Error loading network: {e}")
            return ""
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of all networks"""
        return {
            "total_networks": len(self.networks),
            "total_nodes": len(self.node_registry),
            "total_edges": len(self.edge_registry),
            "networks": [
                {
                    "network_id": network.network_id,
                    "name": network.name,
                    "description": network.description,
                    "node_count": network.get_node_count(),
                    "edge_count": network.get_edge_count(),
                    "density": network.get_connection_density(),
                    "total_energy": network.get_energy_flow_total(),
                    "created_at": network.created_at.isoformat(),
                    "last_updated": network.last_updated.isoformat()
                } for network in self.networks.values()
            ]
        }
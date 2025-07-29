import json
import os
from typing import Self

import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from networkx import DiGraph, shortest_path
from networkx.classes import Graph

from kg_builder.utils import get_triplet_string, rescale_figsize


class RelationsData:
    def __init__(self,
                 annotated_passages: dict[str, list[tuple[str, str, str]]],
                 allowed_entity_types: list[str]):
        self.annotated_passages = annotated_passages
        self.allowed_entity_types = [t.strip().upper() for t in allowed_entity_types]

    def add_relation(self, triplet: tuple[str, str, str], reference_passage: str) -> None:
        if reference_passage in self.annotated_passages:
            self.annotated_passages[reference_passage].append(triplet)
        else:
            self.annotated_passages[reference_passage] = [triplet]

    @classmethod
    def empty(cls, allowed_entity_types: list[str]) -> Self:
        return cls(annotated_passages={}, allowed_entity_types=allowed_entity_types)

    @property
    def flattened_triplets(self) -> list[tuple[str, str, str]]:
        all_triplets = []
        for triplets in self.annotated_passages.values():
            all_triplets.extend(triplets)
        return all_triplets

    @property
    def networkx_graph(self) -> DiGraph:
        graph = nx.DiGraph()
        for triplets in self.annotated_passages.values():
            for t in triplets:
                node1 = t[0].split(":")[0]
                type1 = t[0].split(":")[1]
                node2 = t[2].split(":")[0]
                type2 = t[2].split(":")[1]
                graph.add_node(node1, entity_type=type1)
                graph.add_node(node2, entity_type=type2)
                graph.add_edge(node1, node2, relation=t[1])
        return graph

    @property
    def weakly_connected_components(self) -> list[Graph]:
        g = self.networkx_graph
        nodes_generator = nx.weakly_connected_components(g)
        return [g.subgraph(nodes) for nodes in nodes_generator]

    def save_graph_plot(self,
                        save_dir: str,
                        split_components: bool = True,
                        base_figsize: int = 3,
                        font_size: int = 10,
                        node_base_size: int = 1000,
                        node_color: str = 'lightblue',
                        edge_color: str = 'gray',
                        label_pos: float = 0.5,
                        vertical_alignment: str = 'center',
                        k: float = 0.9
                        ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if split_components:
            to_plot = self.weakly_connected_components
        else:
            to_plot = [self.networkx_graph]
        for i, graph in enumerate(to_plot):
            label = '' if len(to_plot) == 1 else f"component_{i}"
            file_path = os.path.join(save_dir, f"graph_plot_{label}.png")
            pos = nx.spring_layout(graph, seed=42, k=k)
            labels = nx.get_node_attributes(graph, 'entity_type')
            labels = {key: f"{value}\n{key}" for key, value in labels.items()}
            relations = nx.get_edge_attributes(graph, 'relation')
            plt.figure(figsize=rescale_figsize(base_figsize=base_figsize, n_nodes=len(graph)))
            nx.draw(graph,
                    pos,
                    labels=labels,
                    with_labels=True,
                    font_size=font_size,
                    node_size=[len(n) * node_base_size for n in graph.nodes()],
                    node_color=node_color,
                    edge_color=edge_color)
            nx.draw_networkx_edge_labels(graph,
                                         pos,
                                         edge_labels=relations,
                                         font_size=font_size,
                                         label_pos=label_pos,
                                         verticalalignment=vertical_alignment)
            plt.title(f'Knowledge Graph - {label}')
            plt.savefig(file_path)

    def save_json(self, filepath: str) -> None:
        data = {
            'annotated_passages': self.annotated_passages,
            'allowed_entity_types': self.allowed_entity_types
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_json(cls, filepath: str) -> Self:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(annotated_passages=data['annotated_passages'],
                   allowed_entity_types=data['allowed_entity_types'])

    def get_undirected_shortest_path(self, source: str, target: str) -> list[tuple[str, str, str]]:
        undirected_graph = self.networkx_graph.to_undirected()
        node_sequence = shortest_path(undirected_graph, source=source, target=target)
        triplet_sequence = []
        for i in range(len(node_sequence) - 1):
            if self.networkx_graph.get_edge_data(node_sequence[i], node_sequence[i + 1]) is not None:
                triplet_sequence.append((node_sequence[i],
                                         self.networkx_graph.get_edge_data(node_sequence[i], node_sequence[i + 1])[
                                             "relation"],
                                         node_sequence[i + 1]))
            else:
                triplet_sequence.append(
                    (node_sequence[i + 1],
                     self.networkx_graph.get_edge_data(node_sequence[i + 1], node_sequence[i])["relation"],
                     node_sequence[i]))
        return triplet_sequence

    def plot_graph(self, figsize: tuple[float, float] = (10, 8), save_path: str = None) -> None:
        """Create a simple visualization of the knowledge graph."""
        graph = self.networkx_graph.to_undirected()
        
        if len(graph.nodes()) == 0:
            print("No nodes in graph to plot.")
            return
            
        pos = nx.spring_layout(graph, seed=42, k=0.9)
        labels = nx.get_node_attributes(graph, 'entity_type')
        labels = {key: f"{value}\n{key}" for key, value in labels.items()}
        relations = nx.get_edge_attributes(graph, 'relation')
        
        plt.figure(figsize=figsize)
        nx.draw(graph,
                pos,
                labels=labels,
                with_labels=True,
                font_size=8,
                node_size=[len(n) * 300 for n in graph.nodes()],
                node_color='lightblue',
                edge_color='gray')
        nx.draw_networkx_edge_labels(graph,
                                   pos,
                                   edge_labels=relations,
                                   font_size=6,
                                   label_pos=0.5)
        plt.title('Knowledge Graph')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class SearchableRelations:
    def __init__(self,
                 embeddings: HuggingFaceEmbeddings,
                 relations: RelationsData):
        self.embeddings = embeddings
        self.relations_data = relations
        if len(relations.annotated_passages) > 0:
            documents = [Document(get_triplet_string(t), idx=i)
                         for i, t in enumerate(self.relations_data.flattened_triplets)]
            self.triplet_vdb = FAISS.from_documents(documents, embedding=embeddings)
        else:
            self.triplet_vdb = None
        self.embeddings = embeddings

    def add_triplets(self, triplets: list[tuple[str, str, str]], reference_passage: str) -> None:
        previous_length = len(self.relations_data.flattened_triplets)
        if self.triplet_vdb is None:
            self.triplet_vdb = FAISS.from_documents([Document(get_triplet_string(t), idx=i)
                                                     for i, t in enumerate(triplets)],
                                                    embedding=self.embeddings)
        else:
            documents = [Document(get_triplet_string(t), idx=i + previous_length) for i, t in enumerate(triplets)]
            self.triplet_vdb.add_documents(documents)
        for t in triplets:
            self.relations_data.add_relation(t, reference_passage=reference_passage)

    def retrieve_similar_triplets(self, triplet: tuple[str, str, str], k: int = 10) -> list[str]:
        if len(self.relations_data.annotated_passages) == 0:
            return []
        query = get_triplet_string(triplet)
        # TODO: add rrf hybrid search
        similar_docs = self.triplet_vdb.similarity_search(query, k)
        similar_triplets = [doc.page_content for doc in similar_docs]
        return similar_triplets

    def add_relation(self, entity1: str, relation: str, entity2: str, reference_passage: str) -> str:
        triplet = (entity1, relation, entity2)
        if triplet in self.relations_data.flattened_triplets:
            raise RuntimeError(f"The relation {triplet} already exists! Don't add repeated relations.")
        self.add_triplets([triplet], reference_passage)
        return f"Successfully added relation {get_triplet_string(triplet)} to the knowledge graph."

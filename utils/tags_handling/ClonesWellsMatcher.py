import os
from typing import Dict, List, Tuple


import constants
import itertools
import sys
import anndata as ad

import utils as genu

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx

import plotly.graph_objects as go

from tags_handling.AmpliconReader import AmpliconReader
from tags_handling.Tags10XReader import Tags10XReader
import metacells as mc 


font_size = 20

class ClonesWellsMatcher(object):
    """
    A class to match clones to wells, using both the tags reader and the amplicon reader.
    The class works by the clones and not directly by the cells, as the cells might have low number of tags 
    """

    def __init__(self, exp_id:str, tags_reader: Tags10XReader, amplicon_reader:AmpliconReader, output_folder:str = "") -> None:
        """
        Initiate the class with the relevant readers and the experiment id

        :param exp_id: This is just the id for the experiment, used for saving the output
        :type exp_id: str
        :param tags_reader: The tag reader object after clustering the cells into clones
        :type tags_reader: Tags10XReader
        :param amplicon_reader: The amplicon reader object after extracting the tags from the amplicon data
        :type amplicon_reader: AmpliconReader
        :param output_folder: Optional path for saving the data
        :type output_folder: str, optional
        """
        self.tags_reader = tags_reader
        self.amplicon_reader = amplicon_reader
        self.exp_id = exp_id
        
        self.output_folder = output_folder.format(exp_id=exp_id)
        # assert os.path.exists(self.output_folder), f"Output folder {self.output_folder} does not exist"


        self.cells_wells_components: List[List[str]] = [] 
        self.wells_to_cells_mapping: Dict[str, List[str]] = {}
        self.wells_to_clone_mapping: Dict[str, int] = {}
        self.cells_to_wells_mapping: Dict[str,str] = {}
        self.cells_clone_series: pd.Series = pd.Series()
        self.clones_wells_df: pd.DataFrame = None

        self.wells_info_df = pd.DataFrame(data="", index=amplicon_reader.raw_wells_to_tags_mapping.keys(), columns = ["status"])
        self.cells_info_df = pd.DataFrame(data="doublet", index=self.tags_reader.full_cells_addata.obs_names, columns=["status"])
        self.cells_info_df.loc[self.tags_reader.filtered_cells_addata.obs_names, "status"] = ""

        cells_to_match = self.cells_info_df.index[self.cells_info_df.status == ""]
        self.cells_info_df.loc[cells_to_match.difference(self.tags_reader.filtered_cells_tags_df.index), "status"] = "low tag information"



    def calculate_cells_wells_matching_df(self, wells_minimum_reads_for_valid_tag: int = constants.MIN_READS_FOR_VALID_TAG, wells_minimum_number_of_tags: int = constants.MIN_NUMBER_OF_TAGS_IN_WELL):
        """
        Create a graph of the wells and cells, with the cells colored by the number of tags in the cell

        :param wells_minimum_reads_for_valid_tag: The minimum number of reads in a well to use as a valid tag for the matching, defaults to constants.MIN_READS_FOR_VALID_TAG
        :type wells_minimum_reads_for_valid_tag: int, optional
        """
        valid_wells_tags_counter = self.amplicon_reader.get_valid_wells_tags_counter(wells_minimum_reads_for_valid_tag, wells_minimum_number_of_tags)
        self.wells_without_enough_reads = list(self.wells_info_df.index.difference(valid_wells_tags_counter.keys()))
        self.wells_info_df.loc[self.wells_without_enough_reads, "status"] = "low reads"
        
        if len(self.wells_info_df) == 0:
            self.cells_wells_score_df = pd.DataFrame()
            self.cells_wells_umi_frac_df = pd.DataFrame()
            self.cells_well_tags_frac_df = pd.DataFrame()
            return 
        tags_score_series = pd.Series(self.amplicon_reader.tags_to_score)
        
        tags_to_wells_mapping = self.amplicon_reader.tags_to_wells_mapping
        wells_tags_names = tags_to_wells_mapping.keys()
        wells_names = self.amplicon_reader.wells_to_tags_mapping.keys()
        tags_wells_df = pd.DataFrame(0, index=wells_tags_names, columns=wells_names)

        for i, tag in enumerate(tags_to_wells_mapping):
            tags_wells_df.iloc[i].loc[tags_to_wells_mapping[tag]] = 1

        cells_tags_df = self.tags_reader.filtered_cells_tags_df.pivot_table(index=self.tags_reader.filtered_cells_tags_df.index, columns='tag', values='umi', aggfunc='size', fill_value=0)
        cells_tags_df_bool = cells_tags_df.applymap(lambda x: 1 if x > 0 else 0)
        cell_tags_umis_df = self.tags_reader.filtered_cells_tags_df.pivot_table(index=self.tags_reader.filtered_cells_tags_df.index, columns='tag', values='umi',  fill_value=0)

        shared_tags = tags_score_series.index.intersection(tags_wells_df.index).intersection(cells_tags_df_bool.columns)
        tags_wells_df = tags_wells_df.loc[shared_tags]
        tags_score_series = tags_score_series.loc[shared_tags]
        cells_tags_df_bool = cells_tags_df_bool.loc[:,shared_tags]
        cell_tags_umis_df = cell_tags_umis_df.loc[:,shared_tags]

        cells_tags_score_df = cells_tags_df_bool * tags_score_series
        cells_wells_score_df = cells_tags_score_df @ tags_wells_df

        cells_well_umis_df = cell_tags_umis_df @ tags_wells_df
        cells_wells_umi_frac_df = cells_well_umis_df.div(cell_tags_umis_df.sum(axis=1),axis=0)

        cells_well_tags_df = cells_tags_df_bool @ tags_wells_df
        cells_well_tags_frac_df = cells_well_tags_df.div(cells_tags_df_bool.sum(axis=1), axis=0)

        self.cells_wells_score_df = cells_wells_score_df
        self.cells_wells_umi_frac_df = cells_wells_umi_frac_df
        self.cells_well_tags_frac_df = cells_well_tags_frac_df
        
    def get_cells_wells_graph(self,  
                              cell_cell_connection_threshold: float = 0.7, 
                              cell_well_threshold:float=1,
                              cell_multi_well_ratio_threshold:float=1.5,
                              cell_well_min_threshold:float=0.5,
                              wells_minimum_reads_for_valid_tag: int = constants.MIN_READS_FOR_VALID_TAG, 
                              wells_minimum_number_of_tags: int = constants.MIN_NUMBER_OF_TAGS_IN_WELL,
                              min_tags_umis_frac_explained_by_well:float = 0.5,
                              min_tags_frac_explained_by_well:float = 0.5
                                ) -> nx.Graph:

        cells_wells_matching_df = self.cells_wells_score_df
        cells_cells_edges = []
        cells_wells_graph = nx.Graph()

        if len(cells_wells_matching_df) != 0:
            cells_wells_graph.add_nodes_from(cells_wells_matching_df.index, node_type="cell")
            self._calculate_cells_best_possible_score(cells_wells_matching_df)   
            cells_wells_graph.add_nodes_from(cells_wells_matching_df.columns, node_type="well")
            cells_best_score = cells_wells_matching_df.max(axis=1)
            cells_second_best_score = cells_wells_matching_df.apply(lambda x:x.nlargest(2)[1], axis=1)

            cells_second_best_score = cells_wells_matching_df.apply(second_largest_with_column, axis=1)
            cells_second_best_score = cells_second_best_score.loc[cells_wells_matching_df.index]
            cells_second_best_well = cells_second_best_score["second_largest_column"]
            cells_second_best_score = cells_second_best_score["second_largest_value"]

            cells_wells_edges = []
            cells_with_score_ambiguity = []
            wells_with_score_ambiguity = []
            cells_with_tag_ambiguity = []
            wells_with_tag_ambiguity = []

            ### Cells wells edges ### 
            for i, cell_node in enumerate(cells_wells_matching_df.index):
                best_score = cells_best_score.iloc[i]
                second_best_score = cells_second_best_score.iloc[i]
                second_best_well = cells_second_best_well.iloc[i]
                best_possible_score = self.cells_best_possible_score[cell_node] 
                best_well = cells_wells_matching_df.columns[np.argmax(cells_wells_matching_df.iloc[i])]
                
                best_to_second_well_score_ratio = best_score / second_best_score
                best_to_second_umi_frac_ratio = self.cells_wells_umi_frac_df.loc[cell_node, best_well] / self.cells_wells_umi_frac_df.loc[cell_node, second_best_well]
                best_to_second_tags_frac_ratio = self.cells_well_tags_frac_df.loc[cell_node, best_well] / self.cells_well_tags_frac_df.loc[cell_node, second_best_well]

                # the ratio to the next best well is too high, we have ambiguity
                if best_to_second_well_score_ratio < cell_multi_well_ratio_threshold:
                    cells_with_score_ambiguity.append(cell_node)
                    wells_with_score_ambiguity.append(best_well)
                    continue  

                # the fraction of the umis explained by the well is too low
                if self.cells_wells_umi_frac_df.loc[cell_node, best_well] < min_tags_umis_frac_explained_by_well :
                    cells_with_tag_ambiguity.append(cell_node)
                    wells_with_tag_ambiguity.append(best_well)
                    continue

                # also if the ratio is not good enough 
                if self.cells_wells_umi_frac_df.loc[cell_node, best_well] != 1 and best_to_second_umi_frac_ratio < 1.5:
                    cells_with_tag_ambiguity.append(cell_node)
                    wells_with_tag_ambiguity.append(best_well)
                    continue


                # the fraction of the tags explained by the well is too low
                if self.cells_well_tags_frac_df.loc[cell_node, best_well] < min_tags_frac_explained_by_well:
                    cells_with_tag_ambiguity.append(cell_node)
                    wells_with_tag_ambiguity.append(best_well)
                    continue

                # also if the ratio is not good enough 
                if self.cells_well_tags_frac_df.loc[cell_node, best_well] != 1 and best_to_second_tags_frac_ratio < 1.5:
                    cells_with_tag_ambiguity.append(cell_node)
                    wells_with_tag_ambiguity.append(best_well)
                    continue

                # if we are here - this is the best well, and it explain the cell matching enough
                # we have two options: either the score is above the threshold, or it's the best possible score so we will keep it
                if best_score > cell_well_threshold:
                    cells_wells_edges.append((cell_node, best_well, best_score))

                elif best_score == best_possible_score and best_possible_score > cell_well_min_threshold:
                    cells_wells_edges.append((cell_node, best_well, best_score))

            cells_wells_graph.add_weighted_edges_from(cells_wells_edges)
            self.wells_info_df.loc[wells_with_score_ambiguity, "status"] = "possible cell matched multi wells"
            self.wells_info_df.loc[wells_with_tag_ambiguity, "status"] = "possible cell tags explained poorly by well"
            self.cells_info_df.loc[cells_with_score_ambiguity, "status"] = "cell matched multi wells"
            self.cells_info_df.loc[cells_with_tag_ambiguity, "status"] = "cell tags explained poorly by well"


        else:
            cells_wells_graph.add_nodes_from(self.tags_reader.cells_cells_matching_score_df.index, node_type="cell")

        #### Cell-cell edges   ### 
        # based on the cell-cell matching score
        rows_part, col_part = np.where(self.tags_reader.cells_cells_matching_score_df >= cell_cell_connection_threshold)
        cells_cells_connections = zip(self.tags_reader.cells_cells_matching_score_df.index[rows_part],
                                      self.tags_reader.cells_cells_matching_score_df.columns[col_part])
        
        for i, cells in enumerate(cells_cells_connections):
            cell_1, cell_2 = cells
            # We don't want inner loops
            if cell_1 == cell_2:
                continue

            cell_cell_score = self.tags_reader.cells_cells_matching_score_df.iloc[rows_part[i], col_part[i]]
            if cell_cell_score >= cell_cell_connection_threshold:
                cells_cells_edges.append((cell_1, cell_2, cell_cell_score))

        
        cells_wells_graph.add_weighted_edges_from(cells_cells_edges)
        
        # remove components to wells with low number of tags, this is done seperatly because we want to track the reason we removed the well-cell connection
        cells_wells_graph = self.remove_wells_from_components_if_not_enough_tags(cells_wells_graph, wells_minimum_reads_for_valid_tag, wells_minimum_number_of_tags)
        return cells_wells_graph
    
    
    def remove_wells_from_components_if_not_enough_tags(self, cells_wells_graph: nx.Graph, wells_minimum_reads_for_valid_tag: int = constants.MIN_READS_FOR_VALID_TAG, wells_minimum_number_of_tags: int = constants.MIN_NUMBER_OF_TAGS_IN_WELL) -> nx.Graph:
        valid_wells_tags_counter = self.amplicon_reader.get_valid_wells_tags_counter(wells_minimum_reads_for_valid_tag, wells_minimum_number_of_tags)
        wells_with_low_number_of_tags = [k for k,v in valid_wells_tags_counter.items() if v < wells_minimum_number_of_tags]
        
        self.wells_info_df.loc[wells_with_low_number_of_tags, "status"] = "low number of tags"
        graph_connected_components =  [list(i) for i in nx.connected_components(cells_wells_graph)]
        for comopnent in graph_connected_components:
            wells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "well"]
            if len(wells_nodes) == 0 or len(comopnent) == 1:
                continue
            
            well = wells_nodes[0]
            if well in wells_with_low_number_of_tags:
                cells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "cell"]

                for cell in cells_nodes:
                    cells_wells_graph.remove_edge(cell, well)
                    cells_wells_graph.remove_edge(well, cell)
                
                self.cells_info_df.loc[cells_nodes, "status"] = "matched a well with low number of tags"

        return cells_wells_graph

    def dissolve_unmatched_clones_with_multi_wells_potential(self,cells_wells_graph: nx.Graph, component_max_ambiguity_threshold=0.5)-> nx.Graph:
        """
        If we have a clone that doesn't match a well, but it has the potential to match multiple wells, we need to dissolve it because we are not sure that the cells come form the same clone
        """
        graph_connected_components =  [list(i) for i in nx.connected_components(cells_wells_graph)]
        for h, comopnent in enumerate(graph_connected_components):
            # One well or one cell, we have nothing to do
            if len(comopnent) == 1:
                continue

            wells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "well"]
            cells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "cell"]

            # Only cells, if they have potential to multi-wells --> need to dissolve them
            if len(wells_nodes) == 0 and len(self.cells_wells_score_df) !=0:
                best_match_score = self.cells_wells_score_df.loc[cells_nodes].max(axis=1)
                number_of_wells_with_same_score = self.cells_wells_score_df.loc[cells_nodes].ge(best_match_score, axis=0).sum(axis=1)
                if sum(number_of_wells_with_same_score>1) / number_of_wells_with_same_score.shape[0] > component_max_ambiguity_threshold:
                    self.cells_info_df.loc[cells_nodes, "status"] = "dissolved clone because matched multi wells"
                    for cell1, cell2 in itertools.combinations(cells_nodes, 2):
                        cells_wells_graph = self._remove_edge_between_nodes(cells_wells_graph, cell1, cell2)
                        
        return cells_wells_graph

    def disconnect_components_with_multi_wells_potential(self,cells_wells_graph: nx.Graph,  component_max_ambiguity_threshold=0.25) -> nx.Graph:
        """
        If we have components in which all the cells can still match with high likelihood to another well, we need to disconnect them
        """
        # this is for comopnents matching multiple wells
        cells_wells_matching_df = self.cells_wells_score_df
        graph_connected_components =  [list(i) for i in nx.connected_components(cells_wells_graph)]
        for h, comopnent in enumerate(graph_connected_components):
            # One well or one cell, we have nothing to do
            if len(comopnent) == 1:
                continue

            wells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "well"]
            cells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "cell"]
            
            # Only cells, we have nothing to do
            if len(wells_nodes) ==0:
                continue
                
            
            # We have one well in this component - validate that the majority of cells don't match another well
            elif len(wells_nodes) == 1:

                # this make sure that the cell doesn't match another well with the same confidance 
                # best_match_score = np.maximum(cells_wells_matching_df.loc[cells_nodes].max(axis=1), cell_well_threshold)
                best_match_score = cells_wells_matching_df.loc[cells_nodes].max(axis=1)
                number_of_wells_with_same_score = cells_wells_matching_df.loc[cells_nodes].ge(best_match_score, axis=0).sum(axis=1)
                if sum(number_of_wells_with_same_score>1) / number_of_wells_with_same_score.shape[0] > component_max_ambiguity_threshold:
                    # this clone can match several wells, disconnect all elements - including the clone itself because we don't know the origion
                    for cell_node in cells_nodes:
                        self._remove_edge_between_nodes(cells_wells_graph,cell_node, wells_nodes[0])

                        # dissolve to clone itself because it might originated from two different wells
                        for cell_node_2 in cells_nodes:
                            cells_wells_graph = self._remove_edge_between_nodes(cells_wells_graph, cell_node, cell_node_2)
                            
                    self.cells_info_df.loc[cells_nodes, "status"] = "clone matched multi wells"
                    self.wells_info_df.loc[wells_nodes, "status"] = "possible clone matched multi wells"


            # Go over each sub component and make sure that the majority of cells inside of it doesn't match two wells
            else:
                print("THIS SHOULD NOT HAPPEN")
                # subgraph = nx.subgraph(cells_wells_graph, comopnent)
                # subgraph2 = subgraph.copy()
                # wells = [i for i in subgraph2.nodes() if subgraph2.nodes[i]["node_type"] == "well"]
                # subgraph2.remove_nodes_from(wells)
                # single_components = {i:j for i,j in enumerate(nx.connected_components(subgraph2))}
                # for i in single_components:
                #     smaller_component = list(single_components[i])

                #     # Check that component doesn't match more than one well in potential
                #     # best_match_score = np.maximum(cells_wells_matching_df.loc[smaller_component].max(axis=1), cell_well_threshold)
                #     best_match_score = cells_wells_matching_df.loc[smaller_component].max(axis=1)
                #     number_of_wells_with_same_score = cells_wells_matching_df.loc[smaller_component].ge(best_match_score, axis=0).sum(axis=1)
                #     if sum(number_of_wells_with_same_score>1) / number_of_wells_with_same_score.shape[0] > component_max_ambiguity_threshold:
                #         for cell_node in smaller_component:
                #             for well_node in wells:
                #                 cells_wells_graph = self._remove_edge_between_nodes(cells_wells_graph,cell_node, well_node)

                #         self.cells_info_df.loc[smaller_component, "status"] = "clone matched multi wells"

        return cells_wells_graph
                                
    def cut_cells_which_have_no_direct_connection_to_well_if_clone_cannot_explain_them(self, cells_wells_graph: nx.Graph, cell_clone_threshold:float=0.7, cells_clone_ratio:float=1.5) -> nx.Graph:
        # Get all connected components
        cells_to_tags_dict = self.tags_reader.get_cells_tags_mapping()
        graph_connected_components =  [list(i) for i in nx.connected_components(cells_wells_graph)]
        for comopnent in graph_connected_components:
            wells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "well"]

            # No wells in this component, nothing to check
            if len(wells_nodes) < 1:
                continue
            
            # only well without cells, nothing to check
            elif len(wells_nodes) == 1 and len(comopnent) == 1:
                if (wells_nodes[0] not in self.wells_without_enough_reads) and (self.wells_info_df.loc[wells_nodes[0], "status"] == ""):
                    self.wells_info_df.loc[wells_nodes, "status"] = "no cells matched"
                continue
                
            
            cells_nodes = set([i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "cell"])
            wells_to_cells_mapping = {i:[] for i in wells_nodes}
            unconnected_cells_nodes = cells_nodes.copy()
            
            # get all the cells that have no direct connection to the well
            for well_node in wells_nodes:
                connected_cells = self._get_node_neighbors_with_type(cells_wells_graph, well_node, "cell")
                wells_to_cells_mapping[well_node].extend(connected_cells)
                unconnected_cells_nodes = unconnected_cells_nodes - set(connected_cells)
            
            # try to connect the unconnected cell not to another cell but to a whole clone --> increasing the signal 
            # because we know that the clone is explained by the well, if the clone explain the cell well enough then the well will also explain 
            # this shouldn't happen a lot, but happen sometime
            cells_connected_to_the_clone = []
            for cell_node in unconnected_cells_nodes:
                cell_node_tags = cells_to_tags_dict[cell_node]
                cell_node_total_tags = sum(cell_node_tags.values())
                wells_clone_cell_score_series = pd.Series(0, index=wells_to_cells_mapping.keys())
                for well_candidate in wells_to_cells_mapping:
                    possible_clone_cells = wells_to_cells_mapping[well_candidate]
                    clone_tags = {}
                    for clone_cell in possible_clone_cells:
                        clone_cell_tags = cells_to_tags_dict[clone_cell]
                        for tag in clone_cell_tags:
                            if tag not in clone_tags:
                                clone_tags[tag] = 0
                            clone_tags[tag] += clone_cell_tags[tag]
                    
                    clones_total_tags = sum(clone_tags.values())
                    cell_node_explain_clone = 0.0
                    clone_explain_cell_node = 0.0
                    for tag in cell_node_tags:
                        if tag in clone_tags:
                            cell_node_explain_clone += cell_node_tags[tag] / cell_node_total_tags
                            clone_explain_cell_node += clone_tags[tag] / clones_total_tags
                    
                    wells_clone_cell_score_series.loc[well_candidate] = cell_node_explain_clone * clone_explain_cell_node
                
                
                # the cell have a good match to the clone itself, even if not to all the cell in it --> we want to keep it connected to the cell
                # delete connection to all other clones unless we have more than one good match
                possible_clones = wells_clone_cell_score_series[wells_clone_cell_score_series >= cell_clone_threshold]
                if len(possible_clones) != 0:
                    best_score = possible_clones.max()
                    if len(possible_clones) >1 :
                        second_best_score = possible_clones.nlargest(2).iloc[1]
                        
                        # both connections are good, can't decide which one to keep
                        if best_score / second_best_score <= cells_clone_ratio:
                            continue
                    
                    cells_connected_to_the_clone.append(cell_node)
                    
                    # we can just keep the best match
                    for well in wells_clone_cell_score_series.index:
                        if well == possible_clones.idxmax():
                            continue

                        possible_clone_cells = wells_to_cells_mapping[well]
                        for cell in possible_clone_cells:
                            self._remove_edge_between_nodes(cells_wells_graph, cell, cell_node)
                            self._remove_edge_between_nodes(cells_wells_graph, cell_node, cell)

                    
            # disconnect all the remaining unconnected cells from every cell which is connected to the well
            unconnected_cells = list(set(unconnected_cells_nodes) - set(cells_connected_to_the_clone))
            self.cells_info_df.loc[unconnected_cells, "status"] = "cell with low score to clone-well"
            
            for cell_node in unconnected_cells:
                for well in wells_clone_cell_score_series.index:
                    possible_clone_cells = wells_to_cells_mapping[well]
                    for cell in possible_clone_cells:
                        self._remove_edge_between_nodes(cells_wells_graph, cell, cell_node)
                        self._remove_edge_between_nodes(cells_wells_graph, cell_node, cell)
                
            
        return cells_wells_graph
    
    def cut_cell_cell_connection_if_it_combine_wells(self, cells_wells_graph: nx.Graph) -> nx.Graph:
        graph_connected_components =  [list(i) for i in nx.connected_components(cells_wells_graph)]
        for comopnent in graph_connected_components:
            wells_nodes = [i for i in comopnent if cells_wells_graph.nodes[i]["node_type"] == "well"]

            # not connecting several wells, nothing to do
            if len(wells_nodes) < 2:
                continue 

            for well_id1 in wells_nodes:
                well_cells1 = self._get_node_neighbors_with_type(cells_wells_graph, well_id1, "cell")
                
                for well_id2 in wells_nodes:
                    if well_id1 == well_id2:
                        continue

                    well_cells2 = self._get_node_neighbors_with_type(cells_wells_graph, well_id2, "cell")
                    
                    for cell_node1 in well_cells1:
                        for cell_node2 in well_cells2:
                            self._remove_edge_between_nodes(cells_wells_graph, cell_node1, cell_node2)
                            self._remove_edge_between_nodes(cells_wells_graph, cell_node2, cell_node1)
                

        return cells_wells_graph

    
    def extract_cells_wells_components_from_graph(self, cells_wells_graph, min_component_size:int=2):
        self.cells_wells_components = [list(i) for i in nx.connected_components(cells_wells_graph) if len(i) >= min_component_size]    
        return self.cells_wells_components
    
    def extract_clones_mapping(self, cells_wells_graph: nx.Graph):
        assert self.cells_wells_components, "You need to run extract_cells_wells_components_from_graph first"

        cells_wells_matching_df = self.cells_wells_score_df
        
        wells_without_cells = []
        wells_with_cells = []
        cells_with_clone_without_well = []
        singleton_cells_without_wells = []
        
        for i, component in enumerate(self.cells_wells_components):
            wells_nodes = [i for i in component if cells_wells_graph.nodes[i]["node_type"] == "well"]
            cells_nodes = [i for i in component if cells_wells_graph.nodes[i]["node_type"] == "cell"]

            if len(wells_nodes):
                well_id = wells_nodes[0]
                self.wells_to_cells_mapping[well_id] = cells_nodes
                self.wells_to_clone_mapping[well_id] = i

                if len(cells_nodes):
                    wells_with_cells.append(well_id)
                else:
                    wells_without_cells.append(well_id)
            
            for cell_node in cells_nodes:
                self.cells_clone_series.loc[cell_node] = i
                if len(wells_nodes):
                    self.cells_to_wells_mapping[cell_node] = well_id

            if len(wells_nodes) == 0:
                if len(cells_nodes) == 1:
                    singleton_cells_without_wells.extend(cells_nodes)
                else:
                    cells_with_clone_without_well.extend(cells_nodes)

        self.clones_wells_df = pd.DataFrame(0, index=self.wells_to_clone_mapping.keys(), columns=self.wells_to_clone_mapping.values())
        
        for well in self.wells_to_clone_mapping:
            clone_id = self.wells_to_clone_mapping[well]
            cells = self.wells_to_cells_mapping[well]
            self.clones_wells_df.loc[well, clone_id] = cells_wells_matching_df.loc[cells, well].mean()

        # Add wells info
        valid_wells_for_match = self.wells_info_df.index[self.wells_info_df.status == ""]
        self.wells_info_df.loc[valid_wells_for_match.intersection(wells_without_cells), "status"] = "no cells matched"
        self.wells_info_df.loc[valid_wells_for_match.intersection(wells_with_cells), "status"] = "matched to a clone"
        well_best_score = cells_wells_matching_df.max(axis=0)
        self.wells_info_df.loc[well_best_score.index, "best_score"] = well_best_score
        
        valid_cells_for_match = self.cells_info_df.index[self.cells_info_df.status == ""]
        self.cells_info_df.loc[valid_cells_for_match.intersection(singleton_cells_without_wells), "status"] = "singleton without well"
        self.cells_info_df.loc[valid_cells_for_match.intersection(cells_with_clone_without_well), "status"] = "clone without well"
        self.cells_info_df.loc[self.cells_to_wells_mapping.keys(), "status"] = "matched to well"

        cells_best_score = cells_wells_matching_df.max(axis=1)
        self.cells_info_df.loc[cells_best_score.index, "best_score"] = cells_best_score.values

        cells_second_best_score = cells_wells_matching_df.apply(second_largest_with_column, axis=1)
        if len(cells_second_best_score) ==0:
            cells_second_best_score = pd.DataFrame(data=np.nan, index=cells_best_score.index, columns=["second_largest_value", "second_largest_column"])

        self.cells_info_df.loc[cells_second_best_score.index, "second_best_score"] = cells_second_best_score["second_largest_value"]
        self.cells_info_df.loc[cells_best_score.index, "ratio"] = self.cells_info_df.loc[cells_best_score.index, "best_score"] / self.cells_info_df.loc[cells_best_score.index, "second_best_score"]

        # This is a bit tricky, because we can have a best well with the same score as the second best well, but we decided to match it to the first or second wells 
        # this is based on the previous calculation, like matching to the rest of the clone, or how well it match the well compare to other cells in the well
        # so we need to make sure what is the best well that we put
        cells_to_best_well = pd.Series("nan", index=self.cells_info_df.index, dtype=object)
        cells_to_second_best_well = pd.Series("nan", index=self.cells_info_df.index, dtype=object)
        cells_to_well_series = pd.Series(self.cells_to_wells_mapping)
        
        # this put the wells that we were able to match and that we trust
        cells_to_best_well.loc[cells_to_well_series.index] = cells_to_well_series.values

        best_possible_well = cells_wells_matching_df.idxmax(axis=1)        
        second_best_well = cells_second_best_score["second_largest_column"]
        best_second_best_df = pd.concat([best_possible_well,second_best_well], axis=1) 
        best_second_best_df.columns=["best_well","second_best_well"]
        
        # this just put the ones that we are unable to map
        cells_to_best_well[best_second_best_df.index.difference(cells_to_well_series.index)] = best_second_best_df.loc[best_second_best_df.index.difference(cells_to_well_series.index), "best_well"]
        cells_to_second_best_well[best_second_best_df.index.difference(cells_to_well_series.index)] = best_second_best_df.loc[best_second_best_df.index.difference(cells_to_well_series.index), "second_best_well"]

        # now just the one we are able to map, make sure we didn't flip the order
        cells_with_mapping_subset = best_second_best_df.loc[cells_to_well_series.index]
        best_option = cells_with_mapping_subset["best_well"].values
        second_best_option = cells_with_mapping_subset["second_best_well"].values

        cells_to_second_best_well[cells_to_well_series.index[cells_to_well_series == best_option]] = second_best_option[cells_to_well_series == best_option]
        cells_to_second_best_well[cells_to_well_series.index[cells_to_well_series != best_option]] = best_option[cells_to_well_series != best_option]
        
        self.cells_info_df.loc[cells_to_best_well.index, "best_well"] = cells_to_best_well
        self.cells_info_df.loc[cells_to_second_best_well.index, "second_best_well"] = cells_to_second_best_well
                    
    def _get_node_neighbors_with_type(self, graph: nx.Graph, node: str, node_type: str) -> List[str]:
        """
        Get all the neighbors of a node with a specific type

        :param graph: The graph to search in
        :param node: The node to get the neighbors of
        :param node_type: The type of the neighbors to get
        :return: A list of the neighbors of the node with the requested type
        """

        return [i for i in nx.neighbors(graph, node) if graph.nodes[i]["node_type"] == node_type]

    def _remove_edge_between_nodes(self, graph: nx.Graph, node: str, node2: str) -> nx.Graph:
        if graph.has_edge(node, node2):
            graph.remove_edge(node, node2)

        if graph.has_edge(node2,node):
            graph.remove_edge(node2, node)

        return graph
        

    def _calculate_cells_best_possible_score(self, cells_wells_matching_df:pd.DataFrame):
        self.cells_best_possible_score = {cell:0 for cell in cells_wells_matching_df.index}
        
        cells_tags_mapping = self.tags_reader.get_cells_tags_mapping()
        for cell in cells_wells_matching_df.index:
            for tag in cells_tags_mapping[cell]:
                if cell not in self.cells_best_possible_score:
                    self.cells_best_possible_score[cell] = 0
                self.cells_best_possible_score[cell] += self.amplicon_reader.tags_to_score[tag]
            

    def add_clones_wells_information_to_anndata(self, cells_ad:ad.AnnData, only_matches_clones:bool=False, constant_condition_name:str = "") ->ad.AnnData:
        """
        Add the information of the wells to the cells anndata, this will add a column to the cells anndata with the well id, and a column with the clone id

        :param wells_to_clones_dict: A mapping between wells and it's possible clones
        :type wells_to_clones_dict: Dict[str, list[int]]
        :param cells_ad: The cells anndata
        :type cells_ad: ad.AnnData
        :param only_matches_clones: If true, will only show clone id for clones which had a well matching, defaults to True
        :type only_matches_clones: bool, optional
        :param extract_plate_id: Extract and add the plate id from the well name , defaults to True
        :type extract_plate_id: bool, optional
        :return: The cells ad with the added columns
        :rtype: ad.AnnData
        """
        self._convert_clones_id_to_unique()
        # Create a series of clones and wells id
        cells_clones_series = pd.Series(np.nan, index=cells_ad.obs_names)
        cells_wells_series = pd.Series(np.nan, index=cells_ad.obs_names)
        
        for cell_id in self.cells_to_wells_mapping:
            well_plate_id = self.cells_to_wells_mapping[cell_id]
            well_plate_id_s = well_plate_id.split("_")
            well_id = well_plate_id_s[0]
            if len(well_plate_id_s) > 2:
                suffix = well_plate_id_s[2]
                well_id  = well_id + "_" + suffix
            
            cells_wells_series.loc[cell_id] = well_id

        for cell_id in self.cells_clone_series.index:
            cells_clones_series.loc[cell_id] = self.cells_clone_series.loc[cell_id]

        if only_matches_clones:
            cells_clones_series = cells_clones_series.copy()
            cells_clones_series[cells_wells_series[cells_wells_series.isna()].index] = np.isnan

        mc.ut.set_o_data(cells_ad, "clone_id", cells_clones_series)
        mc.ut.set_o_data(cells_ad, "well_id", cells_wells_series)

        cells_condition_series = pd.Series("unknown", index=cells_ad.obs.index)
        for cell_id in self.cells_to_wells_mapping:
            cells_condition_series.loc[cell_id] = self.cells_to_wells_mapping[cell_id].split("_")[1]
        
        # In the first two experiment we have only one condition, so we can just put it as a constant
        if constant_condition_name:
            cells_condition_series = pd.Series(constant_condition_name, index=cells_ad.obs.index)
            
        mc.ut.set_o_data(cells_ad, "condition", cells_condition_series)
        mc.ut.set_o_data(cells_ad, "matching_status", self.cells_info_df.status)
        mc.ut.set_o_data(cells_ad, "matching_best_score", self.cells_info_df.best_score)
        mc.ut.set_o_data(cells_ad, "matching_second_best_score", self.cells_info_df.second_best_score)
        mc.ut.set_o_data(cells_ad, "matching_best_well", self.cells_info_df.best_well)
        mc.ut.set_o_data(cells_ad, "second_best_well", self.cells_info_df.second_best_well)
        mc.ut.set_o_data(cells_ad, "well_matching_ratio", self.cells_info_df.ratio)


        best_well_umi_frac_list = []
        second_best_well_umi_frac_list = []
        best_well_tag_frac_list = []
        second_best_well_tag_frac_list = []

        for i in cells_ad.obs.index:
            if i in self.cells_wells_umi_frac_df.index:
                best_well_umi_frac_list.append(self.cells_wells_umi_frac_df.loc[i, self.cells_info_df.best_well.loc[i]])
                second_best_well_umi_frac_list.append(self.cells_wells_umi_frac_df.loc[i, self.cells_info_df.second_best_well.loc[i]])
                best_well_tag_frac_list.append(self.cells_well_tags_frac_df.loc[i, self.cells_info_df.best_well.loc[i]])
                second_best_well_tag_frac_list.append(self.cells_well_tags_frac_df.loc[i, self.cells_info_df.second_best_well.loc[i]])
            else:
                best_well_umi_frac_list.append(np.nan)
                second_best_well_umi_frac_list.append(np.nan)
                best_well_tag_frac_list.append(np.nan)
                second_best_well_tag_frac_list.append(np.nan)

        
        mc.ut.set_o_data(cells_ad, "best_well_umi_frac", best_well_umi_frac_list)
        mc.ut.set_o_data(cells_ad, "second_best_well_umi_frac", second_best_well_umi_frac_list)
        mc.ut.set_o_data(cells_ad, "well_umi_frac_ratio", cells_ad.obs.best_well_umi_frac / cells_ad.obs.second_best_well_umi_frac)
        mc.ut.set_o_data(cells_ad, "best_well_tag_frac", best_well_tag_frac_list)
        mc.ut.set_o_data(cells_ad, "second_best_well_tag_frac", second_best_well_tag_frac_list)
        mc.ut.set_o_data(cells_ad, "well_tag_frac_ratio", cells_ad.obs.best_well_tag_frac / cells_ad.obs.second_best_well_tag_frac)
        
        cells_ad = self.add_clone_exp_id(cells_ad)

        return cells_ad


    def add_clone_exp_id(self, cells_adata):
        clone_exp_id = pd.Series(cells_adata.obs.clone_id.astype(str) , index=cells_adata.obs.index)
        has_wells = cells_adata.obs_names[cells_adata.obs.well_id.notna()]
        has_clones = cells_adata.obs_names[clone_exp_id.notna()]        
        clone_exp_id[has_wells] = cells_adata.obs.well_id[has_wells].astype(str)
        mc.ut.set_o_data(cells_adata, "well_clone_id", clone_exp_id)  # this is just the well id or clone id 
        
        clone_exp_id = clone_exp_id.copy().astype(str)
        clone_exp_id[has_clones] = clone_exp_id + "_" + cells_adata.obs.condition[has_clones].astype(str) + "_" + cells_adata.obs.exp_id[has_clones].astype(str)
        mc.ut.set_o_data(cells_adata, "clone_exp_id", clone_exp_id)  # this is the clone/well id + the exp id
        
        # now to have a shorter version of the clone id if we don't have the well id
        clone_exp_id_shorten = clone_exp_id.copy()
        has_clones_but_no_wells = has_clones.difference(has_wells)
        clone_exp_id_shorten[has_clones_but_no_wells] = clone_exp_id[has_clones_but_no_wells].str[:15]
        
        mc.ut.set_o_data(cells_adata, "clone_exp_id_short", clone_exp_id_shorten)
        return cells_adata

    
    def get_qc_over_wells(self, cells_ad:ad.AnnData, well_id_col:str="well_id", ax_list:list=[]):
        print("{number_of_cells}({fraction_of_cells:.2f}%) cells matched".format(number_of_cells=cells_ad.obs.shape[0] - np.sum(cells_ad.obs[well_id_col].isnull()), fraction_of_cells=100 * np.sum(~cells_ad.obs[well_id_col].isnull()) / cells_ad.obs.shape[0]))
        print("{number_of_wells} wells matched".format(number_of_wells=len(cells_ad.obs[well_id_col].unique()) - 1))
        
        if ax_list:
            self._plot_number_of_clones_by_clone_size(cells_ad, well_id_col= well_id_col, ax=ax_list[0])
        else:
            self._plot_number_of_clones_by_clone_size(cells_ad, well_id_col= well_id_col)
       
        show = False if ax_list  else True
        # plot number of cells per condition
        ax = ax_list[1] if ax_list else plt.subplots(figsize=(4,6))[1]
        ax.barh(width=cells_ad.obs.condition.value_counts(), y=cells_ad.obs.condition.value_counts().index)
        ax.set_xscale("log", base=2)
        min_value = int(np.log2(np.min(cells_ad.obs.condition.value_counts()))) 
        max_value = int(np.log2(np.max(cells_ad.obs.condition.value_counts()))) + 2
        ax.set_xticks([2**i for i in range(min_value, max_value)])
        ax.set_title("Number of cells per condition")
        ax.set_ylabel("Condition")
        ax.set_xlabel("Number of cells")
        ax.grid(axis="x")

        if show:
            plt.show()

        # plot distribution of success matching based on UMI depth
        ax = ax_list[2] if ax_list else plt.subplots(figsize=(8,4)) [1]
        ax = sb.histplot(x=mc.ut.sum_per(cells_ad.X, per="row"), hue=~cells_ad.obs[well_id_col].isna(), multiple="fill", ax=ax, log_scale=(2,False))
        # ax.set_xscale("log", basex=2)
        ax.set_xlabel("Umi deph")
        ax.set_ylabel("Fraction of cells")
        ax.set_title("well-clone matching as a function of UMI depth")
        sb.move_legend(ax, "upper right")

        if show:
            plt.show()

        # plot distribution of success matching based on number of reads in wells
        if self.amplicon_reader.raw_wells_to_tags_mapping != {}:
            number_of_tags_reads_in_wells = {i : len(self.amplicon_reader.raw_wells_to_tags_mapping[i]) for i in self.amplicon_reader.raw_wells_to_tags_mapping}
            number_of_tags_reads_in_wells_df = pd.DataFrame(pd.Series(number_of_tags_reads_in_wells), columns=["number_of_reads"])
            number_of_tags_reads_in_wells_df["matched_well"] = np.isin(number_of_tags_reads_in_wells_df.index, cells_ad.obs.well_id.unique())

            ax = ax_list[3] if ax_list else plt.subplots(figsize=(8,4))[1]
            ax = sb.histplot(data=number_of_tags_reads_in_wells_df[number_of_tags_reads_in_wells_df.number_of_reads > 0] , x="number_of_reads", hue="matched_well", multiple="fill", ax=ax, log_scale=(2,False))
            # ax.set_xscale("log", basex=2)
            ax.set_xlabel("Number of reads in well")
            ax.set_ylabel("Fraction of wells")
            ax.set_title("well-clone matching as a function of #reads in wells")
            sb.move_legend(ax, "upper right")

            if show:
                plt.show()

        # plot distribution of success matching to clone based on UMI depth
        ax = ax_list[4] if ax_list  else plt.subplots(figsize=(8,4))[1]
        ax = sb.histplot(x=mc.ut.sum_per(cells_ad.X, per="row"), hue=~cells_ad.obs.clone_id.isna(), multiple="fill", ax=ax, log_scale=(2,False))
        # ax.set_xscale("log", basex=2)
        ax.set_xlabel("Umi deph")
        ax.set_ylabel("Fraction of cells")
        ax.set_title("cell-clone matching as a function of UMI depth")
        sb.move_legend(ax, "upper right")

        if show:
            plt.show()

    ### Plotting functions ###
    def plot_number_of_reads_per_clone(self, clones_minimum_valid_size: int = constants.MIN_SIZE_OF_VALID_CLONE, common_tag_index:int = 0):
        """
        Plot the number of reads in the top tag in each clone, common tag zero is the most common tag, 1 is the second most common and etc.
        In general we want at least 2 tags with lots of reads, as we want to be able to match the clones to the wells

        :param clones_minimum_valid_size: Only plot information for clones with at least this number of cells, defaults to constants.MIN_SIZE_OF_VALID_CLONE
        :type clones_minimum_valid_size: int, optional
        :param common_tag_index: The index to plot, common tag zero is the most common tag, 1 is the second most common and etc., defaults to 0
        :type common_tag_index: int, optional
        """
        valid_clones = self.tags_reader.cells_clones_series.value_counts()[self.tags_reader.cells_clones_series.value_counts() >= clones_minimum_valid_size].index
        data = []
        for clone_id in valid_clones:
            tags_per_clone = self.clones_to_counted_tags_mapping[clone_id]
            number_of_reads_per_tag = sorted(list(tags_per_clone.values()), reverse=True)
            data.append(number_of_reads_per_tag[common_tag_index])

        plt.figure(figsize=(8,4))
        sb.ecdfplot(data)
        plt.title("#reads in top %s tag" %common_tag_index)
        plt.xlabel("#reads")
        plt.xscale('log', base=2)

    def plot_clones_wells_score(self, min_value_to_mask: float=1, max_value: float=4, interactive:bool=False,  save_output:bool=False):
        """
        Plot the clones wells score, the score is the sum over all the tags in the well, with each score  1/#wells with this tag

        :param min_value_to_mask: Masking results below this score, defaults to 1
        :type min_value_to_mask: float, optional
        :param max_value: Coloring in the same manner results with a score above this one, defaults to 4
        :type max_value: float, optional
        :param interactive: Generate an interactive plot, this will prevent from saving, defaults to False
        :type interactive: bool, optional
        :param save_output: If true, the results will be saved to the output_folder of the objects, defaults to False
        :type save_output: bool, optional
        """
        assert self.clones_wells_df is not None, "You need to run extract_clones_mapping first"

        ordered_data = genu.order_by_slanter(self.clones_wells_df)

        if interactive:            
            fig = go.Figure(data=go.Heatmap(z=ordered_data.mask(ordered_data < min_value_to_mask),x=ordered_data.columns,y=ordered_data.index.astype(str),
                    hoverongaps = False, colorscale='YlGnBu', zmin=min_value_to_mask, zmax=max_value))

            fig.update_layout(title_text='Clones wells match',title_x=0.5,  template='plotly_white',)
            fig.show()

        else:
            plt.figure(figsize=(100,100))
            sb.heatmap(ordered_data, mask=ordered_data<min_value_to_mask, cmap="Reds", vmax=max_value, vmin=min_value_to_mask)
            plt.tight_layout() 

            if save_output:
                plt.savefig(os.path.join(self.output_folder, "clones_clusters_full.pdf"), format="pdf")
            else:
                plt.show()

    def plot_score_of_matching_ratio(self,cells_wells_matching_df:pd.DataFrame, ax=None, score_to_plot:list[float]=[1]):
        """
        Plot the ratio between the top clone-wells match and the second, we want to see a big ratio, as we want to be able to match the clones to the wells

        :param clones_wells_df: The matching dataframe, this is the first result from match_clones_to_wells
        :type clones_wells_df: pd.DataFrame
        """
        most_highest = cells_wells_matching_df.apply(lambda x: x.sort_values(ascending=False).iloc[0], axis=1)
        second_highest = cells_wells_matching_df.apply(lambda x: x.sort_values(ascending=False).iloc[1], axis=1)
        ratio = most_highest / second_highest
        ratio[np.isinf(ratio)] = np.max(ratio[~np.isinf(ratio)])

        show = False if ax else True
        ax = plt.subplots(1,1,figsize=(8,4))[1] if show else ax
        
        ax = sb.ecdfplot(ratio, label="All clones", complementary=True,ax = ax)

        for i in score_to_plot:
            ax = sb.ecdfplot(ratio[most_highest >=i], label=f"Score => %s" %i, complementary=True,ax =ax)
        ax.set_xlabel("Score ratio")
        ax.set_ylabel("Fraction of clones")
        ax.set_title("Ratio between 1st and 2nd highest score")
        ax.set_xscale("log", base=2)
        ax.legend()

        if show:
            plt.show()

    def plot_score_distribution(self, save_output:bool=False, min_value_to_mask:float=-1):
        assert self.clones_wells_df is not None, "You need to run extract_clones_mapping first"

        data = self.clones_wells_df.values.flatten()
        data = data[data>min_value_to_mask]
        plt.figure(figsize=(8,4))
        sb.ecdfplot(data)
        plt.ylim(0,1.1)
        plt.xticks(ticks=range(0, int(max(data))+1, 1))
        plt.grid()
        
        if min_value_to_mask >=0:
            plt.title("Score distribution (score > {min_value_to_mask})".format(min_value_to_mask=min_value_to_mask))
        else:
            plt.title("Score distribution")
        
        plt.tight_layout() 

        if save_output:
            plt.savefig(os.path.join(self.output_folder, "score_distribution.png"), dpi=1200)
        else:
            plt.show()

    def _plot_number_of_clones_by_clone_size(self, cells_ad:ad.AnnData, ax=None, well_id_col:str="well_id"):
         # plot distribution of number of cells per well
        show = False if ax  else True
        ax = ax  if ax  else plt.subplots(1,1,figsize=(8,4))[1]
        ax = sb.ecdfplot(cells_ad[~cells_ad.obs.well_id.isnull()].obs.well_id.value_counts(), log_scale=(False,2), stat="count", complementary=True, label="Matched clones", ax=ax)
        ax = sb.ecdfplot(cells_ad[cells_ad.obs.well_id.isnull()].obs.clone_id.value_counts(), log_scale=(False,2), stat="count", complementary=True, label="Unmatched clones", ax=ax)
        ax.set_xscale("log", base=2)
        ax.grid(axis="y")
        ax.set_title("Number of clones by clone size")
        ax.set_xlabel("Clone size")
        ax.set_ylabel("Number of clones")
        max_value = cells_ad.obs[well_id_col].value_counts()
        if max_value.shape[0] > 0:
            max_value = int(np.log2(np.max(cells_ad.obs[well_id_col].value_counts()))) + 2
        else:
            max_value = int(np.log2(np.max(cells_ad.obs["clone_id"].value_counts()))) + 2
        ax.set_xticks([2**i for i in range(0, max_value)])
        ax.legend()
        ax.grid(axis="x")

        if show:
            plt.show()

    def plot_plate_matching_matrix(self, condition:str, cells_ad:ad.AnnData ,ax=None):
        assert condition in self.amplicon_reader.plate_to_wells_mapping, "condition not found in amplicon reader object"
        
        plate_wells_df = pd.DataFrame(0, index=list(map(chr, range(65, 81))), columns=[str(i) for i in range(1,25)])
        
        for well in cells_ad.obs.well_id[cells_ad.obs.condition == condition]:
            well_position = well.split("_")[-1]
            plate_wells_df.loc[well_position[0], well_position[1:]] = 1

        ax = ax if ax  else plt.subplots(figsize=(8,8))[1]
        ax = sb.heatmap(1- plate_wells_df, cbar=False, linewidths=.1, yticklabels=list(map(chr, range(65, 81))), xticklabels=[str(i) for i in range(1,25)], ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

    def plot_clones_wells_matching_based_on_score(self, top_score=1.6):
        assert self.clones_wells_df is not None, "You need to run extract_clones_mapping first"
        
        data_list = []
        for i in np.arange(0.1,top_score,0.1):
            data_list.append([sum(np.sum(self.clones_wells_df >i) ==0), sum(np.sum(self.clones_wells_df >i) ==1), sum(np.sum(self.clones_wells_df >i) >1)])

        df = pd.DataFrame(data_list, columns=["no clones", "one clone", "multiple clones"])
        df = df.div(df.sum(axis=1), axis=0)
        df.index = ["%.1f" %i for i in np.arange(0.1,top_score,0.1)]
        df.plot.bar(stacked=True, figsize=(3,4))
        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel("Score")
        plt.ylabel("Fraction of wells")
        plt.title("#clones matching a well")
        

        data_list = []
        for i in np.arange(0.1,top_score,0.1):
            data_list.append([sum(np.sum(clones_wells_df >i, axis=1) ==0), sum(np.sum(clones_wells_df >i,axis=1) ==1), sum(np.sum(clones_wells_df >i,axis=1) >1)])

        df = pd.DataFrame(data_list, columns=["zero matches", "one match", "multiple matches"])
        df = df.div(df.sum(axis=1), axis=0)
        df.index = ["%.1f" %i for i in np.arange(0.1,top_score,0.1)]
        df.plot.bar(stacked=True, figsize=(3,4))
        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel("Score")
        plt.ylabel("Fraction of clones")
        plt.title("#wells matching a clone")

    def plot_single_clone_well_component(self, graph: nx.Graph, component:List[str], output_folder:str=None, output_name:str=None, figsize=24, font_size=font_size):
        node_type_to_color = {"well":"red", "cell":"blue"}

        subgraph = nx.subgraph(graph, component)

        edges_colors = []
        edges_labels = {}
        edges_style = []
        for edge in subgraph.edges():
            if subgraph.edges[edge]["weight"] >= 0.5:
                edges_style.append("-")
            else:
                edges_style.append("--")
                
            if subgraph.nodes[edge[0]]["node_type"] == "well" or subgraph.nodes[edge[1]]["node_type"] == "well":
                edges_colors.append("red")
                edges_labels[(edge[0], edge[1])] = "%.2f" %subgraph.edges[(edge[0], edge[1])]['weight']
            else:
                edges_colors.append("black")
                edges_labels[(edge[0], edge[1])] = "%.2f" %subgraph.edges[(edge[0], edge[1])]['weight']
                
                

        wells = [i for i in subgraph.nodes if subgraph.nodes[i]["node_type"] == "well"]
        if len(wells) ==1:
            pos = nx.circular_layout(subgraph, scale=2)
            pos[wells[0]] = np.array([0,0])
        else:
            pos = nx.spring_layout(subgraph, scale=2)
            
        plt.figure(figsize=(figsize,figsize))
        nx.draw_networkx(subgraph, 
                         with_labels=False,
                         node_color=[node_type_to_color[subgraph.nodes[i]["node_type"]] for i in subgraph.nodes()],
                         style=edges_style, 
                         pos = pos,
                         edge_color=edges_colors,
                         node_size=200,
                         )

        _= nx.draw_networkx_edge_labels(subgraph, pos=pos, edge_labels=edges_labels,font_size=font_size)
        plt.title("Clone has {} cells".format(len(component) - len(wells)))
        if output_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_folder, "%s.png" % output_name))
            plt.close()

    def saves_clones_wells_graphs(self, graph: nx.Graph, components:List[List[str]], output_folder:str):           
        output_folder = output_folder.format(exp_id=self.exp_id)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, component in enumerate(components):
            if len(component) ==1:
                continue
            
            self.plot_single_clone_well_component(graph, component, output_folder, "clone_{}".format(i))

    def _convert_clones_id_to_unique(self):
        self.cells_clone_series_copy = self.cells_clone_series
        cells_clone_series_copy = pd.Series("", index=self.cells_clone_series.index)
        for i in self.cells_clone_series.unique():
            cells_in_clone = self.cells_clone_series[self.cells_clone_series == i].sort_index().index
            clone_id = hashlib.md5( "_".join(cells_in_clone.astype(str)).encode()).digest()
            clone_number = int.from_bytes(clone_id, "big")
            base36 = np.base_repr(clone_number, 36)
            cells_clone_series_copy.loc[cells_in_clone] = base36

        self.cells_clone_series = cells_clone_series_copy

# Add clone info
def second_largest_with_column(row):
    # Get the two largest values
    two_largest = row.nlargest(2)
    
    # Extract the second largest value and corresponding column name
    second_largest_value = two_largest.iloc[1]
    second_largest_column = two_largest.index[1]
    return pd.Series({'second_largest_value': second_largest_value, 'second_largest_column': second_largest_column})

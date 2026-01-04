import os
import re
from typing import Dict, List, Tuple, Set

import networkx as nx
import multiprocessing
import constants
import time 

import sys

import utils as genu


import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sb
import tqdm

from scipy.cluster.hierarchy import fcluster, ward


import metacells as mc

import plotly.graph_objects as go


# BAM_FILES_TEMPLATE = (
#     "{project_path}/cellranger_output/{sample}/outs/possorted_genome_bam.bam"
# )

font_size = 20

class Tags10XReader:
    """
    A tag reader object for 10X data, will read and extract the tags
    """
    import time

    def __init__(
        self,
        exp_id: str,
        full_cells_addata: ad.AnnData,
        data_path: str = "",
        output_path: str = "",
        filtered_cells_addata: ad.AnnData = None,
        cells_umis_tags_df = None,
        min_umis_for_valid_tag:int = constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG, 
        minimum_number_of_tags_in_cell:int =constants.MIN_TAGS_IN_CELL,
        min_reads_for_valid_tag_umis:int = constants.MIN_READS_FOR_VALID_TAG_UMIS,
        samples_to_ignore: list[str] = ["cd34"], 
        approved_tags:list[str] = [],
        hamming_dict_path:str = "./data/lib1_tags_hamming_1_dict.pkl",
        sample_format:str = "{exp_id}_{sample_day}_cell_tags_information.parquet",
        min_umis_ratio_for_valid_tag:int = 10,
    ):
        """
        Read 10x fastq files and extract the tags from them using the prefix and suffix of the tags.
        Will try to read the tags from a file if it exists, if not will extract the tags from the fastq files.
        Will also try to read the hamming cloud dict from a file if it exists, if not will generate it from the tags.

        :param exp_id: Only the date from the proejct name
        :type exp_id: str
        :param full_cells_addata: The full cells addata object, this is before any filtering to make sure we only work on tags of cells which we might be interested in
        :type full_cells_addata: ad.AnnData
        :param filtered_cells_addata: The cells addata object, will be used to filter the cells that didn't pass the 10x QC so we don't care about tags
        :type filtered_cells_addata: ad.AnnData
        :param data_path: Path for the celranger output, defaults to constants.DATA_PATH
        :type data_path: str, optional
        :param output_path: Path to save the output objects, defaults to constants.PROJECT_OUTPUT_PATH
        :type output_path: str, optional
        :param cells_cells_match_score_df_path: Path for the hamming distance which was calculated before, defaults to constants.HAMMING_01_DF_PATH
        :type cells_cells_match_score_df_path: str, optional
        :param barcode_prefix: The prefix of the tag, defaults to "GGT"
        :type barcode_prefix: str, optional
        :param barcode_suffix: The suffix of the tag, defaults to "GAATTC"
        :type barcode_suffix: str, optional
        :param minimal_distance_for_clones: Minimum score between cells to have a connection between them, defaults to 0.5
        :type minimal_distance_for_clones: float
        """
        self.output_folder = output_path.format(exp_id=exp_id)
        # assert os.path.exists(self.output_folder), "Output folder doesn't exists"

        self.hamming_cloud_dict = genu.read_from_pickle(hamming_dict_path)
        self.exp_id = exp_id
        self.full_cells_addata = full_cells_addata
        self.approved_tags = approved_tags
        
        if filtered_cells_addata is None:
             filtered_cells_addata = mc.ut.slice(full_cells_addata, obs=~full_cells_addata.obs.excluded_cell, vars=~full_cells_addata.var.excluded_gene)

        self.filtered_cells_addata = filtered_cells_addata
        
        if cells_umis_tags_df.shape[0] == 0:
            self.project_path = os.path.join(data_path, exp_id)
            self.samples = [i for i in os.listdir(os.path.join(self.project_path, "cellranger_output"))]
            self.samples = [i for i in self.samples if (i.lower() not in samples_to_ignore and i.upper() not in samples_to_ignore)]
            # Read and filtered out the cells umis tags based on the full cells addata
            tags_information_paths = {sample:sample_format.format(exp_id=exp_id, sample_day=sample) for sample in self.samples}
            self.cells_umis_tags_df = self._get_tags_information(tags_information_paths)
        else:
            self.cells_umis_tags_df = cells_umis_tags_df
        

        # Filter the tags based on thresholds, approved tags, and cells
        self.filtered_cells_tags_df =  self._filter_cells_umis_tags_df(min_umis_for_valid_tag = min_umis_for_valid_tag, 
                                                                            minimum_number_of_tags_in_cell = minimum_number_of_tags_in_cell, 
                                                                            min_reads_for_valid_tag_umis=min_reads_for_valid_tag_umis,
                                                                            min_umis_ratio_for_valid_tag=min_umis_ratio_for_valid_tag)
        

        print("Running cell-cell matching score algorithm - this might take a few minutes")
        self.cells_cells_matching_score_df = self.get_cells_cells_match_score()

        self.cells_tags_counter_df = self.cells_umis_tags_df[["umi", "tag"]].groupby(["cell_id", "tag"]).umi.nunique()


    def run_clone_calling(self, minimal_distance_for_clones: float = 0.7):
        self.cells_clones_series = self._extract_clones_base_on_cliques(self.cells_cells_matching_score_df, minimal_distance_for_clones=minimal_distance_for_clones)
        self.clone_to_tags_count_dict = self.get_clones_to_counted_tags_mapping()

    def update_cells_clones_series(self, cells_clones_series: pd.Series):
        """
        Update the cells clones series

        :param cells_clones_series: The new cells clones series
        :type cells_clones_series: pd.Series
        """
        self.cells_clones_series = cells_clones_series
        # self.clone_to_tags_count_dict = self.get_clones_to_counted_tags_mapping()

    def get_cells_cells_match_score(self, tags_as_bool:bool=False):
        """
        Calculate the match score between cells based on the hamming distance between their tags

        :param cells_cells_match_score_df_path: The output path for the cell cell match score dataframe
        :type cells_cells_match_score_df_path: str
        :param tags_as_bool: If true, we only care if the tags exists or not , and are uneffected by the number of umis in those tags, use True with care
        :type tags_as_bool: bool
        :param min_umis_for_valid_tag: Only tags with more than this number of UMIs will be used to calculate the matching score, defaults to constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG
        :type min_umis_for_valid_tag: int
        :param minimum_number_of_tags_in_cell: Only cells with at least this number of tags will be considered valid cells, other cells will be just ignored, defaults to 2
        :type minimum_number_of_tags_in_cell: int
        """
        # Calculate the distance matrix of hamming 0 and hamming 1 and combine tham to symmatrical matrix
        hamming_0_matrix, hamming_1_matrix = self._calculate_cells_hamming_matrix(
            tags_as_bool=tags_as_bool, 
        )

        hamming_0_01_frac_sym_df = self._calculate_cell_cell_score(
            hamming_0_matrix,
            hamming_1_matrix,
            tags_as_bool=tags_as_bool,
        )
        
        cells_cells_matching_score_df = pd.DataFrame(
            np.tril(hamming_0_01_frac_sym_df) + np.tril(hamming_0_01_frac_sym_df).T,
            index=hamming_0_01_frac_sym_df.index,
            columns=hamming_0_01_frac_sym_df.columns,
        )
        np.fill_diagonal(cells_cells_matching_score_df.values, 1)

        cells_cells_matching_score_df_0_hamming = pd.DataFrame(
            np.triu(hamming_0_01_frac_sym_df) + np.triu(hamming_0_01_frac_sym_df).T,
            index=hamming_0_01_frac_sym_df.index,
            columns=hamming_0_01_frac_sym_df.columns,
        )
        np.fill_diagonal(cells_cells_matching_score_df_0_hamming.values, 1)
        self.cells_cells_matching_score_df_0_hamming = cells_cells_matching_score_df_0_hamming
        
        return cells_cells_matching_score_df

    def _filter_cells_umis_tags_df(self, min_umis_for_valid_tag:int, minimum_number_of_tags_in_cell:int,  tags_reads_thrshold:float = 0.75,min_reads_for_valid_tag_umis:int=2, min_umis_ratio_for_valid_tag: int = 10):       
        
        # If provided, use the approved tags only
        if len(self.approved_tags):
            filtered_cells_umis_tags_df = self.cells_umis_tags_df.iloc[np.where(self.cells_umis_tags_df.tag.isin(self.approved_tags))[0]]

        else:
            filtered_cells_umis_tags_df = self.cells_umis_tags_df.copy()
        
        
        filtered_cells_umis_tags_df["cell"] = filtered_cells_umis_tags_df.index

        # Take only cells from the filtered anndata 
        filtered_cells_umis_tags_df = filtered_cells_umis_tags_df.iloc[np.where(filtered_cells_umis_tags_df.cell.isin(self.filtered_cells_addata.obs_names))[0]]     


        # Group by 'cell', 'umi', and 'tag' and count occurrences
        grouped_cells_umis_tags_df = filtered_cells_umis_tags_df.groupby(['cell', 'umi', 'tag']).size().reset_index(name='reads')

        # Calculate the total counts for each 'cell' and 'umi' combination
        total_counts = grouped_cells_umis_tags_df.groupby(['cell', 'umi'])['reads'].transform('sum')

        # Compute the ratio of each tag's count to the total counts
        grouped_cells_umis_tags_df['umi_tag_reads_ratio'] = grouped_cells_umis_tags_df['reads'] / total_counts

        
        # When we have multiple tags with the same UMI, take only the one with the highest ratio and only if the number of reads is above min_reads_for_valid_tag_umis
        cells_umis_tags_df_with_enough_reads = grouped_cells_umis_tags_df[(grouped_cells_umis_tags_df["umi_tag_reads_ratio"] >= tags_reads_thrshold) & (grouped_cells_umis_tags_df["reads"] >= min_reads_for_valid_tag_umis)]

        # We only wants tags with enough UMIs in them, meaning they were expressed enough so we will be sure it's not noise
        cell_tag_umi_count_df = cells_umis_tags_df_with_enough_reads.groupby(["cell","tag"]).umi.count().reset_index()
        cell_tag_umi_count_df = cell_tag_umi_count_df.groupby('cell').apply(self._filter_tags, threshold_factor=min_umis_ratio_for_valid_tag, min_umis_for_valid_tag=min_umis_for_valid_tag).reset_index(drop=True)

        # We filtred out the cells which doesn't have enough tags or umis per tag
        cells_tags_count = cell_tag_umi_count_df.groupby("cell").size()
        cells_with_enough_tags = cells_tags_count[cells_tags_count >= minimum_number_of_tags_in_cell]

        cell_tag_umi_count_df = cell_tag_umi_count_df.set_index(cell_tag_umi_count_df.cell)
        filtered_cell_tag_umi_count_df = cell_tag_umi_count_df.loc[cells_with_enough_tags.index]

        filtered_cell_tag_umi_count_df["sample_id"] = self.filtered_cells_addata[filtered_cell_tag_umi_count_df.index].obs.sample_day
        
        number_of_cells = len(filtered_cell_tag_umi_count_df.index.unique())
        print("Filtered cells with %d valid tag and at least %d umis per tag: %d cells (%.2f%%)" % (minimum_number_of_tags_in_cell , min_umis_for_valid_tag,number_of_cells, number_of_cells / self.filtered_cells_addata.obs.shape[0] * 100))

        number_of_cells_with_tags_per_day = filtered_cell_tag_umi_count_df.groupby("sample_id").cell.nunique()
        print(number_of_cells_with_tags_per_day / self.filtered_cells_addata.obs.sample_day.value_counts() * 100)

        return filtered_cell_tag_umi_count_df

    def _get_tags_information(self, output_files:Dict[str,str]) -> pd.DataFrame:
        """
        Go over a group of bam files from different samples extract the tag information from all of them based on the given pattern.
        """
        cell_tags_information_list = []
        number_of_cells_per_sample = {}
        for sample in output_files:
            file_path = output_files[sample]
            if not os.path.exists(file_path):
                print("No infomration for sample file %s" %sample)

            cell_tags_information = pd.read_parquet(file_path) #, engine="fastparquet")
            
            sample_name = sample
            if self.exp_id == "CD34_GFP_BULK":
                sample_name = 'cd34_gfp_bulk'
            else:
                cells_sample_name = self.full_cells_addata.obs_names[0].split("_")[1]
                if cells_sample_name == cells_sample_name.upper():
                    sample_name = sample_name.upper()
                elif cells_sample_name == cells_sample_name.lower():
                    sample_name = sample_name.lower()

            cell_tags_information["sample_id"] = sample_name
            cell_tags_information.index = genu.add_exp_and_sample_info_to_cell_barcode(cell_barcode=cell_tags_information.cell, exp_id=self.exp_id, sample_name=sample_name)
            cell_tags_information.index.name = "cell_id"


            # Only work on cells which passed the 10x pipeline qc 
            shared_cells = set(self.full_cells_addata.obs_names) & set(cell_tags_information.index)
            cell_tags_information = cell_tags_information.iloc[cell_tags_information.index.isin(shared_cells)]
            number_of_cells_per_sample[sample_name] = len(shared_cells)
            cell_tags_information_list.append(cell_tags_information)

        # concat all the samples 
        cells_umis_tags_df = pd.concat(cell_tags_information_list)
        shared_cells = set(self.full_cells_addata.obs_names) & set(cells_umis_tags_df.index)


        print("Full cells tags statistics:")
        print("Out of %d cells from 10x, %d has any tag(%.2f%%)" % (self.full_cells_addata.obs.shape[0], len(shared_cells), len(shared_cells) / self.full_cells_addata.obs.shape[0] * 100))       

        # number_of_cells_with_tags_per_day = cells_umis_tags_df.groupby("sample_id").cell.nunique()
        number_of_cells_with_tags_per_day = pd.Series(number_of_cells_per_sample)
        print(number_of_cells_with_tags_per_day / self.full_cells_addata.obs.sample_day.value_counts() * 100)

        return cells_umis_tags_df

    def _calculate_cells_hamming_matrix(
        self, tags_as_bool=False, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the hamming distrance matrixes between the cells base on the tags information

        :return: Two 2D arrays, the first is the zero-hamming matrix between all the cells and the second is the one-hamming matrix between the cells
                The array is arrange such that each value is the number of tags umi's from cell i that can explain cell j tags umi's, given an unsymmetrical matrix.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        cells_tags_df = self.filtered_cells_tags_df.pivot(index="cell", columns="tag", values="umi")
        cells_tags_df[cells_tags_df.isna()] = 0

        # Fully explained by the barcodes
        hamming_0_matrix = np.zeros(
            (cells_tags_df.shape[0], cells_tags_df.shape[0]), dtype=int
        )

        tags_for_each_cell = self.filtered_cells_tags_df.groupby(self.filtered_cells_tags_df.index).tag.unique()

        for i, cell_barcode in enumerate(
            tqdm.tqdm(cells_tags_df.index, desc="0 hamming")
        ):
            
            cell_tags = tags_for_each_cell[cell_barcode]

            hamming_0_matrix[:, i] = (
                (cells_tags_df[cell_tags] != 0).astype(int).sum(axis=1)
                if tags_as_bool
                else cells_tags_df[cell_tags].sum(axis=1)
            )


        # 1 hamming explained by the barcodes
        possible_cells_tags = set(cells_tags_df.columns)

        hamming_1_matrix = np.zeros(
            (cells_tags_df.shape[0], cells_tags_df.shape[0]), dtype=int
        )

        for i, cell_barcode in enumerate(
            tqdm.tqdm(cells_tags_df.index, desc="1 hamming")
        ):
            cell_tags = tags_for_each_cell[cell_barcode]

            # Get all those from the hamming 1
            hamming_cell_tags = []
            for tag in cell_tags:
                hamming_cell_tags.extend(self.hamming_cloud_dict[tag])

            # Make sure not to have the original tags
            hamming_cell_tags_set = set(hamming_cell_tags) - set(cell_tags)
            hamming_cell_tags_set = possible_cells_tags & hamming_cell_tags_set
            hamming_cell_tags_set = list(hamming_cell_tags_set)
            hamming_1_matrix[:, i] = (
                (cells_tags_df[hamming_cell_tags_set] != 0).astype(int).sum(axis=1)
                if tags_as_bool
                else cells_tags_df[hamming_cell_tags_set].sum(axis=1)
            )

        return hamming_0_matrix, hamming_1_matrix

    def _calculate_cell_cell_score(
        self,
        hamming_0_matrix: np.ndarray,
        hamming_1_matrix: np.ndarray,
        tags_as_bool=False,
    ) -> pd.DataFrame:
        """
        Calculate a 2-way measurement between cells, using the hamming distance.
        This is used to make sure the shared distance between two of the cells

        :param hamming_0_matrix: zero-hamming matrix between all the cells
        :type hamming_0_matrix: np.ndarray

        :param hamming_1_matrix: one-hamming matrix between all the cells
        :type hamming_1_matrix: np.ndarray

        :param min_tags_for_valid_cell: Minimum number of tags in a cell to be considered a valid cell to check, defaults to 0
        :type min_tags_for_valid_cell: int, optional

        :return: Represent the shared precentage of tags umi's which can be explain from cell i to cell j.
        :rtype: pd.DataFrame
        """
        # the hamming matrix 1 can't be larger then the hamming matrix 0
        hamming_1_matrix = np.minimum(hamming_1_matrix, hamming_0_matrix)      
        
        # Cells with less than X number of tags are invalid for matching 

        cells_tags_df = self.filtered_cells_tags_df.pivot(index="cell", columns="tag", values="umi")
        cells_tags_df[cells_tags_df.isna()] = 0

        # generate mapping between a cell and it's tags
        number_of_umis_per_cells = (
            (cells_tags_df!=0).sum(axis=1)
            if tags_as_bool
            else cells_tags_df.sum(axis=1)
        )

        # move to fractions
        hamming_0_frac = hamming_0_matrix / number_of_umis_per_cells.values[:, None]
        hamming_0_frac[np.isnan(hamming_0_frac)] = 0

        hamming_1_frac = hamming_1_matrix / number_of_umis_per_cells.values[:, None]
        hamming_1_frac[np.isnan(hamming_1_frac)] = 0

        # if the hamming 1 is bigger than hamming 0, remove hamming 1
        hamming1_higher_than0 = np.where(hamming_1_frac >= hamming_0_frac)
        hamming_1_frac[hamming1_higher_than0] = 0
        
        hamming_01_frac = hamming_0_frac + hamming_1_frac

        # # make the graph 2-sided
        hamming_0_frac_sym = hamming_0_frac * hamming_0_frac.T
        hamming_01_frac_sym = hamming_01_frac * hamming_01_frac.T
        
        # Convert to df
        hamming_0_01_frac_sym_df = pd.DataFrame(
            np.triu(hamming_0_frac_sym )
            + np.tril(hamming_01_frac_sym),
            index=cells_tags_df.index, 
            columns=cells_tags_df.index
        )

        np.fill_diagonal(hamming_0_01_frac_sym_df.values, 1)
        return hamming_0_01_frac_sym_df
    
    def _extract_clones_base_on_cliques(self, cells_cells_score_df: pd.DataFrame, minimal_distance_for_clones: float) -> pd.Series:
        """
        Cluster cells if they are on the same clique, i.e. they are connected to each other.
        If one cell is connected to two different cliques, it will be assigned to the clique with the highest number of cells.

        :param cells_cells_score_df: The score between each pair of cells
        :type cells_cells_score_df: pd.DataFrame
        :param minimal_distance_for_clones: Minimum score between cells to have a connection between them
        :type minimal_distance_for_clones: float
        :return: The series of cells to clones matching
        :rtype: pd.Series
        """
        rows_part, col_part = np.where(cells_cells_score_df>= minimal_distance_for_clones)
        edges = [(cells_cells_score_df.index[rows_part[i]], cells_cells_score_df.columns[col_part[i]]) for i in range(len(rows_part)) ]

        cells_graph = nx.Graph()
        cells_graph.add_nodes_from(cells_cells_score_df.index)
        cells_graph.add_edges_from(edges)

        cells_clusters_series = pd.Series(-1, index=cells_cells_score_df.index, dtype=int)
        connected_components = nx.connected_components(cells_graph)
        connected_components = sorted(connected_components, key=len, reverse=False)

        # First work on all the singlets and couples clones
        clone_index = 0

        for component in connected_components:
            if len(component) >2:
                break
                
            cells_clusters_series.loc[component] = clone_index
            clone_index +=1

        # Now work on the larger clones
        larger_components = [component for component in connected_components if len(component) >2]
        for components in larger_components:
            cells_left_to_match = components
            
            subgraph = nx.subgraph(cells_graph, cells_left_to_match)
            cells_to_cliques_index = {i:[] for i in cells_left_to_match}

            cliques = list(set(i) for i in nx.find_cliques(subgraph))

            # for every cell, we get the cliques index it is in
            for i, clique in enumerate(cliques):
                for cell in clique:
                    cells_to_cliques_index[cell].append(i)

            # For every cell, find the minial cliques that are in agreemeent 
            cells_to_cliques = {i: cliques[cells_to_cliques_index[i][0]] for i in cells_to_cliques_index}

            for cell in cells_to_cliques:
                cliques_list = [cliques[i] for i in cells_to_cliques_index[cell]]
                cells_to_cliques[cell] = cells_to_cliques[cell].intersection(*cliques_list)
                    
            while len(cells_left_to_match):
                smallest_size_to_work = min([len(cells_to_cliques[i]) for i in cells_to_cliques])

                for cell in cells_to_cliques:
                    if len(cells_to_cliques[cell]) != smallest_size_to_work:
                        continue

                    cells_clusters_series.loc[cells_to_cliques[cell]] = clone_index
                    clone_index +=1
                
                new_matched_cells = set(cells_left_to_match) & set(cells_clusters_series[cells_clusters_series!=-1].index)
                cells_left_to_match = list(set(cells_left_to_match) & set(cells_clusters_series[cells_clusters_series==-1].index))
                
                # update the cliques mapping by removing those old cells --> we should still get cliques
                for cell in new_matched_cells:
                    del cells_to_cliques[cell]
                
                for cell in cells_to_cliques:
                    cells_to_cliques[cell] = cells_to_cliques[cell] - new_matched_cells
                

        return cells_clusters_series
    
    def get_clones_to_tags_mapping(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of clones to tags mapping

        :return: The mapping between clones and tags
        :rtype: Dict[str, List[str]]
        """
        clone_to_tags_dict = {}

        tags_per_cell = self.filtered_cells_tags_df.groupby(self.filtered_cells_tags_df.index).tag.unique()
        for clone_id in self.cells_clones_series.unique():
            clone_cells = self.cells_clones_series[self.cells_clones_series == clone_id].index
            clone_to_tags_dict[clone_id] = np.unique(np.concatenate(tags_per_cell.loc[clone_cells].values))

        return clone_to_tags_dict

    def get_clones_to_counted_tags_mapping(self) -> Dict[str, Dict[str, int]]:
        """
        Get a dictionary of clones to tags mapping, where the tags are counted

        :return: A mapping between clones to the tags, and for each tag we write 
        :rtype: Dict[str, Dict[str, int]]
        """
        clone_to_tags_dict = {}
        
        tags_per_cell = self.cells_tags_counter_df
        
        for clone_id in tqdm.tqdm(
            self.cells_clones_series.unique(),
            desc="Collecting tags from all the cells in the clone",
        ):
            if np.isnan(clone_id):
                continue

            clone_cells = self.cells_clones_series[self.cells_clones_series == clone_id].index
            clone_to_tags_dict[clone_id] = tags_per_cell.loc[clone_cells].groupby("tag").sum().to_dict()
            
        return clone_to_tags_dict

    def get_cells_tags_mapping(self, min_number_of_tags_per_cell: int = constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG) -> Dict[str, Dict[str, int]]:
        """
        Get a dictionary of cells to tags mapping

        :param min_number_of_tags_per_cell: Cells with less than this number of tags will be ignored, defaults to constants.MIN_NUMBER_OF_TAGS_IN_WELL
        :type min_number_of_tags_per_cell: int, optional
        :return: Mappig between the cells and it's tags and the number of times each tag appeared in the cell
        :rtype: Dict[str, Dict[str, int]]
        """
        cells_to_tags_dict : Dict[str, Dict[str, int]] = {}

        cells_tags_count = self.filtered_cells_tags_df[["umi","tag"]].groupby(["cell","tag"]).sum()

        cells_tags_count = cells_tags_count[cells_tags_count >=min_number_of_tags_per_cell]
        cells_tags_count_as_dict = cells_tags_count.umi.to_dict()

        for cell, tag in cells_tags_count_as_dict:
            if cell not in cells_to_tags_dict:
                cells_to_tags_dict[cell] = {}
            if cells_tags_count_as_dict[(cell,tag)] !=0 and not np.isnan([cells_tags_count_as_dict[(cell,tag)] ])[0] :
                cells_to_tags_dict[cell][tag] = cells_tags_count_as_dict[(cell,tag)] 

        return cells_to_tags_dict
        
    def add_clones_information_to_anndata(self, cells_ad:ad.AnnData) -> ad.AnnData:
        """
        Add the clone information to the anndata object as a clone id column

        :param cells_ad: The anndata object to add the clone information to
        :type cells_ad: ad.Anndata
        :return: The anndata object with the clone information
        :rtype: ad.Anndata
        """
        cells_clones_series = self.cells_clones_series.copy()
        mc.ut.set_o_data(cells_ad, "clone_id", cells_clones_series)
        return cells_ad

    ### Plotting functions ###
    def plot_number_of_tags_per_clone(self, clones_minimum_reads_for_valid_tag: int = constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG, min_number_of_tags_per_clone:int = constants.MIN_TAGS_IN_CELL, ax=None):
        show = True if ax is None else False
        clones_tags_mapping = self.clone_to_tags_count_dict
        clones_size = self.cells_clones_series.value_counts()
        number_of_tags_per_clone = pd.Series(0,index=list(clones_tags_mapping.keys()))
        
        for clone, tags in clones_tags_mapping.items():
            number_of_tags_per_clone[clone] = len([i for i in tags if tags[i] >= min(clones_minimum_reads_for_valid_tag, clones_size[clone] * 1/2)])
        
        ax = plt.subplots(1,1,figsize=(8,4))[1] if ax is None else ax
        for i in range(1, 7):
            clones_large_enough = clones_size >= 2**i
            ax = sb.ecdfplot(number_of_tags_per_clone[clones_large_enough], label="{}".format(2**i), complementary=True, stat="count", log_scale=(2,False), ax=ax)
            

        ax.set_title("Number of tags per clone")
        ax.set_xlabel("Number of tags")
        ax.set_ylabel("Number of clones")
        ax.grid()
        ax.legend(title="Clone size", loc="upper right")
        ax.axvline(min_number_of_tags_per_clone, color="black", linestyle="--")
    
        if show:
            plt.show()

    def plot_number_of_tags_per_cell(self, cell_minimum_reads_for_valid_tag: int = constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG, min_number_of_tags_per_clone:int = constants.MIN_TAGS_IN_CELL, ax=None,
    font_size:int=font_size):
        show = True if ax is None else False
        cells_tags_counter_df = self.cells_tags_counter_df.unstack(fill_value=0)
        number_of_tags_per_cell = np.sum(cells_tags_counter_df >= cell_minimum_reads_for_valid_tag,axis=1)
        
        ax = plt.subplot() if ax is None else ax
        ax = sb.ecdfplot(number_of_tags_per_cell, complementary=True, log_scale=(2,False), ax=ax)
            
        ax.set_title("Tags per cell", fontsize=font_size)
        ax.set_xlabel("Number of tags", fontsize=font_size)
        ax.set_ylabel("Fraction of cells" , fontsize=font_size)
        ax.grid(axis="both", visible=True)
        ax.axvline(min_number_of_tags_per_clone, color="black", linestyle="--")
        ax.tick_params(axis='both', which='major', labelsize=font_size)
    
        if show:
            plt.show()

    def plot_number_of_umis_per_tag_in_cells(self, ax = None, cell_minimum_reads_for_valid_tag: int = constants.MIN_NUMBER_OF_UMIS_FOR_VALID_TAG,
                                             font_size:int=font_size):
        cells_tags_counter_df = self.cells_tags_counter_df.unstack(fill_value=0)
        max_number_of_tags_per_cell = cells_tags_counter_df.max(axis=1)
        second_max_number_of_tags_per_cell = cells_tags_counter_df.apply(lambda x:x.nlargest(2)[1], axis=1)

        show = True if ax is None else False
        ax = plt.subplot() if ax is None else ax
        
        ax = sb.ecdfplot(max_number_of_tags_per_cell, ax=ax, label="1st tag", log_scale=2, complementary=True)
        ax = sb.ecdfplot(second_max_number_of_tags_per_cell, ax=ax, label="2nd tag", log_scale=2, complementary=True)
        
        ax.set_xlabel("Number of UMIs", fontsize=font_size)
        ax.set_ylabel("Fractions of cells", fontsize=font_size)
        ax.set_title("Umis for the most common tags", fontsize=font_size)
        max_value = int(np.log2(np.max(max_number_of_tags_per_cell))) + 1
        ax.set_xticks([2**i for i in range(max_value)])
        ax.grid(axis="both", visible=True)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        plt.legend()
        ax.axvline(cell_minimum_reads_for_valid_tag, color="red", linestyle="--")

    
        if show:
            plt.show()
            


    def plot_clones_size_distribution(self, ignore_singleton: bool = True, ax_list:list=None, font_size=12):
        show = True if ax_list is None else False
        ax = plt.subplot() if ax_list is None else ax_list[0]
        max_value = int(np.log2(np.max(self.cells_clones_series.value_counts()[self.cells_clones_series]))) + 2
        ax = sb.histplot(self.cells_clones_series.value_counts()[self.cells_clones_series], stat="probability", bins=[2**i for i in range(max_value)], ax=ax)
        ax.set_xscale("log",base=2)
        ax.set_xticks([2**i for i in range(max_value)])
        ax.set_xlabel("Clones size", fontsize=font_size)
        ax.set_ylabel("Probability", fontsize=font_size)
        ax.set_title("Clones size probability", fontsize=font_size)
        ax.set_xticks([2**i for i in range(max_value)])
        ax.grid(axis="y")
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # center the xticks
        plt.xticks(rotation=90, ha='center')

        plt.legend()
    
        if show:
            plt.show()


        data = (
            self.cells_clones_series.value_counts()[
                self.cells_clones_series.value_counts() > 1
            ]
            if ignore_singleton
            else self.cells_clones_series.value_counts()
        )
        ax = plt.subplot() if ax_list is None else ax_list[1]
        ax = sb.ecdfplot(data, log_scale=(2, False), stat="count", complementary=True, ax=ax)
        ax.set_ylabel("Number of clones")
        ax.set_xlabel("#Cells")
        ax.set_title(
            "Non singleton clones size (%d)"
            % (
                sum(self.cells_clones_series.value_counts() > 1),
            )
        )
        max_value = int(np.log2(np.max(self.cells_clones_series.value_counts()))) + 1
        ax.set_xticks([2**i for i in range(max_value)])
        ax.grid()
        if show:
            plt.show()
            

    def plot_cells_clusters(
        self,
        ignore_singleton: bool = True,
        plot_each_sample: bool = False,
        save_output: bool = False,
        minimal_distance_for_clones: float = 0.5,
    ):
        assert (
            not self.cells_clones_series.empty
        ), "No clones were found. Please run the clone detection first"
        if save_output:
            assert os.path.exists(self.output_folder), "Output folder does not exist"

        filtered_cells_clones_series = (
            self.cells_clones_series[
                self.cells_clones_series.isin(
                    self.cells_clones_series.value_counts()[
                        self.cells_clones_series.value_counts() > 1
                    ].index
                )
            ]
            if ignore_singleton
            else self.cells_clones_series
        )

        def _format_distance_df_as_heatmap(
            cells_clones_series: pd.Series,
        ) -> pd.DataFrame:
            cells_clones_series_with_size = pd.DataFrame(
                cells_clones_series, columns=["cluster"]
            )
            cells_clones_series_with_size[
                "clones_size"
            ] = cells_clones_series.value_counts()[cells_clones_series].values
            cells_orders = cells_clones_series_with_size.sort_values(
                by=["clones_size", "cluster"]
            )

            # cells_orders = cells_clones_series.argsort()
            distance_df_copy = self.cells_cells_matching_score_df.loc[
                cells_orders.index, cells_orders.index
            ].copy()
            np.fill_diagonal(distance_df_copy.values, 1)
            distance_df_copy[distance_df_copy == 0] = np.nan
            return distance_df_copy


        plt.figure(figsize=(30, 30))
        sb.heatmap(
            _format_distance_df_as_heatmap(filtered_cells_clones_series),
            xticklabels=False,
            yticklabels=False,
            cmap="YlGnBu",
            vmin=minimal_distance_for_clones,
            vmax=1,
        )
        plt.tight_layout()

        if save_output:
            plt.savefig(
                os.path.join(
                    self.output_folder,
                    "clones_clusters_full.png",
                ),
                dpi=1200,
            )
        else:
            plt.show()

        if plot_each_sample:
                
            samples = set(
                [i[1] for i in filtered_cells_clones_series.index.str.split("_")]
            )
            for sample in samples:
                sample_cells_clones_series = filtered_cells_clones_series[
                    filtered_cells_clones_series.index.str.endswith(sample)
                ]
                plt.figure(figsize=(30, 30))
                sb.heatmap(
                    _format_distance_df_as_heatmap(sample_cells_clones_series),
                    xticklabels=False,
                    yticklabels=False,
                    cmap="YlGnBu",
                    vmin=minimal_distance_for_clones,
                    vmax=1,
                )
                plt.title("Clones for sample day %s" % sample)
                plt.tight_layout()

                if save_output:
                    plt.savefig(
                        os.path.join(
                            self.output_folder,
                            "clones_clusters_{sample}.png",
                        ).format(sample=sample),
                        dpi=1200,
                    )
                else:
                    plt.show()

    def _filter_tags(self, group, threshold_factor, min_umis_for_valid_tag):
        # Sort tags by UMI in descending order
        # sorted_tags = group.sort_values(by='umi', ascending=False)
        
        # # Get the second most common tag's UMI
        # if len(sorted_tags) > 1:
        #     second_most_umi = sorted_tags.iloc[1]['umi']
        # else:
        #     second_most_umi = sorted_tags.iloc[0]['umi']
        
        # # Calculate threshold as 1/10 of the second most UMI
        # threshold = int(max(min_umis_for_valid_tag, second_most_umi / threshold_factor))
        threshold = int(max(min_umis_for_valid_tag, group['umi'].max() / threshold_factor))
        
        # Filter tags with UMI greater than or equal to the threshold
        return group[group['umi'] >= threshold]

    def save_individual_clusters(self, output_path:str = None, number_of_clusters_to_save:int = 100, minimal_distance_for_clones: float = 0.5,):
        """
        plot the individual clusters matching score

        :param output_path: The folder to save the plots to, defaults to None in which case,taking the output folder from the object and creating a new folder named "clusters"
        :type output_path: str

        :param number_of_clusters_to_save: Number of clusters to save, defaults to 100
        :type number_of_clusters_to_save: int
        """
        if output_path is None:
            output_path = os.path.join(self.output_folder, "clusters")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        cells_tags_mapping = self.get_cells_tags_mapping()
        samples_to_colors = {j:sb.color_palette("tab10")[i] for i, j in enumerate(self.filtered_cells_addata.obs.sample_day.unique())}
        
        top_clusters_to_plot = self.cells_clones_series.value_counts().head(number_of_clusters_to_save).index
        for cluster in top_clusters_to_plot:
            cluster_cells = self.cells_clones_series[self.cells_clones_series == cluster].index
            cluster_cells_matching_score_df = self.cells_cells_matching_score_df.loc[cluster_cells, cluster_cells]

            cells_colors = [samples_to_colors[x.split("_")[1]] if x in cluster_cells else "white" for x in cluster_cells]

            cg = sb.clustermap(cluster_cells_matching_score_df, cmap="YlGnBu", vmin=minimal_distance_for_clones, vmax=1, 
                               row_colors=cells_colors, col_colors=cells_colors, cbar_pos=(0.02, 0.4, 0.05, 0.4))
            plt.title(len(cluster_cells))
            
            # Remove labels and add number of tags
            ax = cg.ax_heatmap
            ax.set_ylabel("")
            ax.set_xlabel("")
            new_labels = []

            for i,l in enumerate(ax.axes.get_yticklabels()):
                number_of_tags_for_cell = len(cells_tags_mapping[l._text])
                l.set_text(number_of_tags_for_cell)
                new_labels.append(l)
            ax.axes.set_yticklabels(new_labels)
            try:
                ax.axes.set_xticklabels(new_labels)
            except:
                print("some problem with the x labels")
            

            # Add legend for samples
            handles = [Patch(facecolor=samples_to_colors[name]) for name in samples_to_colors]
            ax.legend(handles, samples_to_colors, title='Samples',
                bbox_to_anchor=(1.1, 1), loc='upper left')

            cg.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
            cg.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "cluster_%s.png" % cluster), dpi=1200)

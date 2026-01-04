import os
import re
import sys
import pathlib
import gzip
from typing import Dict, List, Union, Tuple

import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import seaborn as sb
from scipy.spatial import distance


import utils as genu
import constants
import tags_utils

font_size = 20

class MockAmpliconReader(object):
    def __init__(self):
        self.raw_wells_to_tags_mapping = {}
        self.wells_tags_counter = {}
        self.wells_to_tags_mapping_full = {}
        self.wells_to_tags_mapping = {}
        self.tags_to_wells_mapping = {}
        self.wells_tags_counter = {}
        self.tags_to_score = {}

        self.plate_to_wells_mapping = {}

    def get_non_prevalent_tags(self):
        return []
    
    def get_valid_wells_tags_counter(self, *args,**kwargs):
        return {}

class AmpliconReader(object):
    """
    This class is used to read the amplicon fastq files and extract the tags from them, provide several plots to decide the correct threshold for the tags for each well and will be part of the clones-wells matcher
    """
    def __init__(
        self, amplicon_fastq_folder: str = "", barcode_prefix="CCGGT", barcode_suffix="GAATTCG",
        min_reads_for_common_tag: int = constants.MIN_NUMBER_OF_READS_FOR_COMMON_TAG, min_reads_for_valid_tag:int = constants.MIN_READS_FOR_VALID_TAG,
        inner_pattern: str="[ACGT]{8}", allow_seq_error:bool=False, min_tags_for_valid_well: int = constants.MIN_NUMBER_OF_TAGS_IN_WELL,
        hamming_dict_path:str = "./lib1_tags_hamming_1_dict.pkl", raw_wells_to_tags_mapping=None) -> None:
        """
        1. Read the fastq files from amplicon_fastq_folder and extract the tags base on the prefix and suffix provided.
        2. Merge tags using 1-hamming distance based on the min_reads_for_common_tag paramater.
        3. Filter out tags based on the min_reads_for_valid_tag paramater, creating a mapping between tags to wells and vice versa.
        4. Also provide mapping between wells to the number of valid tags, and between plate and the wells inside of it 
        
        :param amplicon_fastq_folder: The path for the amplicon fastq folder
        :type amplicon_fastq_folder: str
        :param barcode_prefix: Prefix for the tag , defaults to "GGT"
        :type barcode_prefix: str, optional
        :param barcode_suffix: Suffix for the tag, defaults to "GAATTC"
        :type barcode_suffix: str, optional
        :param min_reads_for_common_tag: Tags with less than this number of reads will be candidated for merger for more common tags, defaults to constants.MIN_NUMBER_OF_READS_FOR_COMMON_TAG
        :type min_reads_for_common_tag: int, optional
        :param min_reads_for_valid_tag: Only tags with at least this amount of reads will be considered relaible tags, defaults to constants.MIN_READS_FOR_VALID_TAG
        :type min_reads_for_valid_tag: int, optional
        """
        self.allow_seq_error = allow_seq_error
        self.barcode_prefix = barcode_prefix
        self.inner_pattern = inner_pattern
        self.barcode_suffix = barcode_suffix
        self.amplicon_fastq_folder = amplicon_fastq_folder
        self.min_reads_for_valid_tag = min_reads_for_valid_tag
        self.min_tags_for_valid_well = min_tags_for_valid_well
        self.min_reads_for_common_tag = min_reads_for_common_tag
        self.hamming_cloud_dict = genu.read_from_pickle(hamming_dict_path)

        if raw_wells_to_tags_mapping is None:
            self.raw_wells_to_tags_mapping = self._extract_amplicon_tags_from_fastq_paths()
        else:
            self.raw_wells_to_tags_mapping = raw_wells_to_tags_mapping.to_dict()["values"]
        self.wells_tags_counter = self._count_number_of_reads_per_tags_in_wells()
        self.wells_to_tags_mapping_full = self._merge_uncommon_tags_with_common_per_well()
        self.wells_to_tags_mapping = self.get_filtered_well_to_tags_mapping()
        self.tags_to_wells_mapping = self.get_tags_to_wells_mapping()
        self.wells_tags_counter = self.get_valid_wells_tags_counter(wells_minimum_reads_for_valid_tag=self.min_reads_for_valid_tag, min_tags_for_valid_well=self.min_tags_for_valid_well)
        self.tags_to_score = {tag: 1/len(self.tags_to_wells_mapping[tag]) for tag in self.tags_to_wells_mapping}

        self.plate_to_wells_mapping = self._get_plate_to_wells_mapping()

    def get_valid_wells_tags_counter(self, wells_minimum_reads_for_valid_tag:int = constants.MIN_READS_FOR_VALID_TAG, min_tags_for_valid_well:int = constants.MIN_NUMBER_OF_TAGS_IN_WELL) -> Dict[str, int]:
        """
        Return a counter of the number of tags per well, after filtering out tags with less than some threshold of min reads

        :param wells_minimum_reads_for_valid_tag: The minimum number of reads for a tag to be considered valid, defaults to constants.MIN_READS_FOR_VALID_TAG
        :type wells_minimum_reads_for_valid_tag: int
        :return: A counter of the number of tags per well, after filtering out tags with less than some threshold of min reads
        :rtype: Dict[str, int]
        """
        temp_wells_tags_counter = {}
        for tag in self.tags_to_wells_mapping:
            for clone_id in self.tags_to_wells_mapping[tag]:
                if clone_id not in temp_wells_tags_counter :
                    temp_wells_tags_counter[clone_id] = 0
                temp_wells_tags_counter[clone_id] +=1

        wells_tags_counter = {k:v for k,v in temp_wells_tags_counter.items() if v >= min_tags_for_valid_well}

        print("Only {number_of_wells_with_reads} wells out of {total_number_of_wells} has a valid tag above {wells_minimum_reads_for_valid_tag} reads".format(number_of_wells_with_reads=len(temp_wells_tags_counter),
                                                                                                                                                       total_number_of_wells=len(self.raw_wells_to_tags_mapping),
                                                                                                                                                       wells_minimum_reads_for_valid_tag=wells_minimum_reads_for_valid_tag))


        print("Out of those, {number_of_wells_with_tags} wells have at least {wells_minimum_tags} tags".format(number_of_wells_with_tags = len(wells_tags_counter), wells_minimum_tags=min_tags_for_valid_well))
        return temp_wells_tags_counter

    def get_filtered_well_to_tags_mapping(self, pcr_cycle_bias: int = 4,
                                          min_reads_for_valid_tag: int = constants.MIN_READS_FOR_VALID_TAG, 
                                          min_tags_for_valid_well:int = constants.MIN_NUMBER_OF_TAGS_IN_WELL) -> Dict[str, List[str]]:
        assert (
            self.wells_to_tags_mapping
        ), "Must run merge_uncommon_tags_with_common_per_well"
        
        min_reads_for_valid_tag = min_reads_for_valid_tag if min_reads_for_valid_tag else self.min_reads_for_valid_tag
        min_tags_for_valid_well = min_tags_for_valid_well if min_tags_for_valid_well else self.min_tags_for_valid_well

        self.wells_to_max_tag_mapping = {}
        self.wells_to_second_max_tag_mapping = {}
        for well_id in self.wells_to_tags_mapping:
            sorted_wells_to_tags_mapping = sorted(
                self.wells_to_tags_mapping[well_id].values(), reverse=True
            )
            max_value = sorted_wells_to_tags_mapping[0]
            second_max_value = sorted_wells_to_tags_mapping[1] if len(sorted_wells_to_tags_mapping) > 1 else max_value
            
            self.wells_to_max_tag_mapping[well_id] = max_value
            self.wells_to_second_max_tag_mapping[well_id] = second_max_value   # this is to remove pcr bias

        
        wells_to_tags_mapping_temp: Dict[str, List[str]] = {}
        for well_id in self.wells_to_tags_mapping:
            for tag_id in self.wells_to_tags_mapping[well_id]:
                if (
                    self.wells_to_tags_mapping[well_id][tag_id]
                    < max(min_reads_for_valid_tag, int(self.wells_to_second_max_tag_mapping[well_id] * (1/2**pcr_cycle_bias)))  # two pcr rounds below the max
                ):
                    continue
                
                if well_id not in wells_to_tags_mapping_temp:
                    wells_to_tags_mapping_temp[well_id] = []
                
                wells_to_tags_mapping_temp[well_id].append(tag_id)

        return wells_to_tags_mapping_temp

    def get_tags_to_wells_mapping(self) -> Dict[str, List[str]]:
        """
        Flip the wells_to_tags_mapping to tags_to_wells_mapping, and filter out tags with less than some threshold of min reads

        :param min_reads_for_valid_tag: Tags with less than this amount of reads will be ignored, defaults to None
        :type min_reads_for_valid_tag: int, optional
        :return: Mapping between tags and wells
        :rtype: Dict[str, List[str]]
        """
        assert (
            self.wells_to_tags_mapping
        ), "Must run merge_uncommon_tags_with_common_per_well"
        
        
        tags_to_wells_mapping: Dict[str, List[str]] = {}
        for well_id in self.wells_to_tags_mapping:
            for tag_id in self.wells_to_tags_mapping[well_id]:
                if tag_id not in tags_to_wells_mapping:
                    tags_to_wells_mapping[tag_id] = []

                tags_to_wells_mapping[tag_id].append(well_id)

        return tags_to_wells_mapping

    def _extract_amplicon_tags_from_fastq_paths(self) -> Dict[str, List[str]]:
        """
        Read the amplicon tags from the fastq files and match each well(file) the list of tags seen here, duplicated might exists

        :return: The mapping between wells and all the tags reads we've seen
        :rtype: Dict[str, List[str]]
        """
        amplicon_files = [i for i in pathlib.Path(self.amplicon_fastq_folder).rglob("*.fastq.gz")]
        well_to_tags_dict: Dict[str, List[str]] = {}

        if self.allow_seq_error:
            _, reg_short_tag_list = tags_utils.get_regex_with_seq_errors(self.barcode_prefix, self.inner_pattern,self.barcode_suffix)
    
        else:
            reg_short_tag = "%s(%s)%s" % (self.barcode_prefix, self.inner_pattern,self.barcode_suffix)
            reg_short_tag_list = [re.compile(reg_short_tag)]


        pbar = tqdm.tqdm(amplicon_files, total=len(amplicon_files))
        for amplicon_file in pbar:
            pbar.set_description("Processing fastq files")
            if "Undetermined" in amplicon_file.name:
                continue
            
            possible_tags = []

            amplicon_parts = amplicon_file.name.split("_")
            for i in range(len(amplicon_parts)):
                if amplicon_parts[i][1:].isdigit():
                    break

            well_name =  "_".join(amplicon_parts[:i+1])
            
            if well_name not in well_to_tags_dict:
                well_to_tags_dict[well_name] = []

            with gzip.open(amplicon_file, "r") as f:
                for i, line in enumerate(f):
                    if i % 4 != 1:
                        continue
                    for reg_short_tag in reg_short_tag_list:
                        if len(reg_short_tag.findall(str(line))):
                            possible_tags.append((i, reg_short_tag.findall(str(line))[0]))
            
            possible_tags = pd.DataFrame(possible_tags, columns=["row_index", "tag"])
            possible_tags = possible_tags.drop_duplicates()
            well_to_tags_dict[well_name].extend(possible_tags.tag)
        
        return well_to_tags_dict

    def _count_number_of_reads_per_tags_in_wells(self, show_plot:bool=False) -> pd.DataFrame:
        """
        Count the number of tags inside each well

        :return: The counted wells to tags mapping
        :rtype: pd.DataFrame
        """

        temp_list = []
        for well in self.raw_wells_to_tags_mapping:
            for tag in self.raw_wells_to_tags_mapping[well]:
                temp_list.append((well, tag, 1))

        wells_tags_counter = pd.DataFrame(temp_list, columns=["wells", "tags", "num"])
        wells_tags_counter = (
            wells_tags_counter.groupby(["wells", "tags"])
            .sum()
            .sort_values("num")
            .reset_index()
        )

        if show_plot:
            plt.figure(figsize=(8,8))
            sb.histplot(wells_tags_counter.num, log_scale=(2, 2))
            plt.title("Number of reads per tag")
            plt.xlabel("#Reads")
        
        return wells_tags_counter
    
    def get_non_prevalent_tags(self, max_number_of_wells_per_tag:int = 10):
        return pd.Series(self.tags_to_score)[pd.Series(self.tags_to_score) > 1/max_number_of_wells_per_tag].index
    
    def _get_plate_to_wells_mapping(self) -> Dict[str, List[str]]:
        """
        Return a mapping between plate to wells based on the well_id, we expect the well_id to be in the format of plate_id_well_number

        :return: Mapping between plate to wells
        :rtype: Dict[str, List[str]]
        """
        plate_to_wells_mapping:  Dict[str, List[str]] = {}
        for well_id in self.wells_to_tags_mapping:
            plate_id = "_".join(well_id.split("_")[:-1])
            if plate_id not in plate_to_wells_mapping:
                plate_to_wells_mapping[plate_id] = []
            plate_to_wells_mapping[plate_id].append(well_id)
        return plate_to_wells_mapping
    
    def _merge_uncommon_tags_with_common_per_well(
        self, min_reads_for_common_tag: int = None, 
    ) -> Dict[str,dict[str, int]]:
        """
        Merge uncommon tags with common tags based on the hamming distance between them, and return a mapping between wells to tags.
        We are using 1-hamming distance and trying to merge uncommon tags with common tags, based on the threshold for common tags

        :param min_reads_for_common_tag: Every tag with less than this number of reads will be considered uncommon and a potential for merge, defaults to None
        :type min_reads_for_common_tag: int, optional
        :return: Mapping between welsl and it's tags
        :rtype: Dict[str,dict[str, int]]
        """
        
        min_reads_for_common_tag = min_reads_for_common_tag if min_reads_for_common_tag else self.min_reads_for_common_tag
        
        wells_to_tags_dict = {}
        pbar = tqdm.tqdm(self.wells_tags_counter.wells.unique())
        for well_id in pbar:
            pbar.set_description("Merging common tags in well %s" % well_id)
            well_data = self.wells_tags_counter[
                self.wells_tags_counter.wells == well_id
            ]
            
            common_tags_info = well_data[well_data.num >= min_reads_for_common_tag]
            common_tags_info = common_tags_info.sort_values("num", ascending=False)

            wells_to_tags_dict[well_id] = (
                common_tags_info[["tags", "num"]].set_index("tags").to_dict()["num"]
            )

            non_common_tags_info = well_data[well_data.num < min_reads_for_common_tag]

            for non_common_tag in non_common_tags_info.tags:
                found_match = False
                optional_tags_based_on_hamming = set(self.hamming_cloud_dict[non_common_tag])

                for common_tag in common_tags_info.tags.values:
                    if common_tag in optional_tags_based_on_hamming:
                        wells_to_tags_dict[well_id][common_tag] += non_common_tags_info[non_common_tags_info.tags == non_common_tag].num.values[0]
                        found_match = True
                        break

                if not found_match:
                    wells_to_tags_dict[well_id][non_common_tag] = non_common_tags_info[non_common_tags_info.tags == non_common_tag].num.values[0]

        self.wells_to_tags_mapping = wells_to_tags_dict
        return wells_to_tags_dict

    #### PLOT FUNCTIONS ####
    def plot_number_of_wells_per_barcode(self, ax=None, font_size:int=font_size):
        number_of_wells_per_tag = []
        for tag_id in self.tags_to_wells_mapping:
            number_of_wells_per_tag.append(len(self.tags_to_wells_mapping[tag_id]))

        show = True if ax is None else False
        ax = plt.subplot() if ax is None else ax
        ax =sb.ecdfplot(number_of_wells_per_tag, ax=ax)
        ax.set_title("Barcode uniqueness in wells", fontsize=font_size)
        ax.set_xlabel("#Wells a barcode appear in", fontsize=font_size)
        ax.set_ylabel("Fraction of wells", fontsize=font_size)
        ax.set_xscale("log", base=2)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.grid()
        ax.set_xticks([2**i for i in range(0,int( np.log2(np.max(number_of_wells_per_tag)) + 1,))])

        if show:
            plt.show()

    def plot_max_reads_per_well(self, min_reads_for_valid_tag: int = None, plot_swarm:bool=False, ax=None, only_report_on_plate:str=None, font_size:int=font_size, skip=1):
        min_reads_for_valid_tag = min_reads_for_valid_tag if min_reads_for_valid_tag else self.min_reads_for_valid_tag

        if only_report_on_plate:
            assert only_report_on_plate in self.plate_to_wells_mapping , "Plate not found in plate_to_wells_mapping"
            max_reads_per_well = [
                np.max(list(self.wells_to_tags_mapping[well_id].values()))
                for well_id in self.wells_to_tags_mapping if well_id in self.plate_to_wells_mapping[only_report_on_plate]
            ]
        else:
            max_reads_per_well = [
                np.max(list(self.wells_to_tags_mapping[well_id].values()))
                for well_id in self.wells_to_tags_mapping
            ]

        show = True if ax is None else False
        if plot_swarm:
            plt.figure(figsize=(4, 8))
            sb.swarmplot(data=max_reads_per_well)
            plt.axhline(y=min_reads_for_valid_tag, xmin=0, xmax=1, color="red")
            plt.xlabel("")
            plt.ylabel("max_reads")
            plt.yscale("log", basey=2)
            plt.title("Number of reads per well for dominant barcode")
            plt.show()

        ax = plt.subplot() if ax is None else ax
        ax = sb.ecdfplot(max_reads_per_well, complementary=True, stat="count",ax=ax)
        ax.axvline(x=min_reads_for_valid_tag, ymin=0, ymax=np.max(max_reads_per_well), color="red")
        ax.set_title("Dominant barcode reads per well", fontsize=font_size)
        ax.set_xlabel("Number of reads", fontsize=font_size)
        ax.set_ylabel("Number of wells", fontsize=font_size)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2**i for i in range(0,int( np.log2(np.max(max_reads_per_well)) + 1,), skip)])
        
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.grid()
        if show:
            plt.show()

    def plot_number_of_valid_barcodes_per_well(
        self, min_reads_for_valid_tag: int = None, min_number_of_tags_in_well: int = constants.MIN_NUMBER_OF_TAGS_IN_WELL, ax=None, only_report_on_plate:str=None, plot_for_plate:bool=False,
        font_size:int=font_size):
        min_reads_for_valid_tag = min_reads_for_valid_tag if min_reads_for_valid_tag else self.min_reads_for_valid_tag

        values_list = []
        for well_id in self.wells_to_tags_mapping:
            if only_report_on_plate and well_id not in self.plate_to_wells_mapping[only_report_on_plate]:
                continue

            well_series = pd.Series(self.wells_to_tags_mapping[well_id])
            number_of_valid_tags = len(
                well_series[well_series >= min_reads_for_valid_tag].index
            )
            if number_of_valid_tags > 0:
                values_list.append(number_of_valid_tags)
        
        show = True if ax is None else False
        ax = plt.subplot() if ax is None else ax
        ax = sb.ecdfplot(values_list, stat="count", complementary=True, log_scale=(2,False),ax=ax)
        ax.set_title("Barcodes per well", fontsize=font_size)
        ax.set_xlabel("Number of barcodes", fontsize=font_size)
        ax.set_ylabel("Number of wells", fontsize=font_size)
        ax.axvline(x=min_number_of_tags_in_well, ymin=0, ymax=np.max(values_list), color="red")

        if plot_for_plate:
            ax.set_ylim(0, 400)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=font_size)
    
        if show:
            plt.show()


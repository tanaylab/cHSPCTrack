import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import constants
import utils as genu

import re
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform



def get_hamming_distance_cloud(whitelist_files:list[str], hamming_distance:int=1,output_path:str = "./hamming_cloud.pkl"):
    """
    Extract the tags from whitelist files and generate a dict of tags in specific hamming distance from each tag

    :param whitelist_files: List of paths for the whitelist files
    :type whitelist_files: list[str]
    :param hamming_distance: The required hamming distance between two tags to consider them in the same hamming cloud, defaults to 1
    :type hamming_distance: int, optional
    :param output_path: Path for a file to store the tags for later use, i None return it to the user, defaults to None
    :type output_path: str, optional
    :return: Either none if path was given or the hamming cloud dict
    :rtype: None or dict[str]:list
    """

    # Read and combine all the files with the tags 
    whitelists_data = []

    for whitelist_file in whitelist_files:
        library_reads = pd.read_csv(whitelist_file)
        library_reads = library_reads.set_index("v1.CellTag")
        whitelists_data.append(library_reads)
        
    whitelists_data_df = pd.concat(whitelists_data)

    # Find the require score based on the tag length and the user requirement 
    tag_sample = whitelists_data_df.index[0]
    require_score = hamming_distance / len(tag_sample)

    tags_hamming_cloud = {i:[] for i in whitelists_data_df.index}

    # Convert the tags to integers for the distance functions 
    tags_list_as_ints = []
    for tag in whitelists_data_df.index:
        tag = tag.replace("A","0")
        tag = tag.replace("C","1")
        tag = tag.replace("G","2")
        tag = tag.replace("T","3")
        tags_list_as_ints.append(list(tag))

    # Calculate hamming distance between all the tags 
    condence_hamming_distance = pdist(np.array(tags_list_as_ints), 'hamming')
    hamming_distance_matrix = squareform(condence_hamming_distance)

    # Find and store the tags which considered inside each hamming cloud
    couple_row_index, couple_col_index = (np.where(hamming_distance_matrix == require_score))

    for i in range(len(couple_row_index)):
        tags_hamming_cloud[whitelists_data_df.index[couple_row_index[i]]].append(whitelists_data_df.index[couple_col_index[i]])

    # Save the data or return it
    if output_path:
        genu.save_to_pickle(tags_hamming_cloud, output_path)        
            
    else:
        return tags_hamming_cloud



def report_number_of_wells_containing_several_patient(cells_addata: ad.AnnData, well_field_name:str = "well_id", print_entire_patients:bool=True) -> int:
    """
    Check how many wells contain cells from more than one patient

    :param cells_addata: The cells addata with the wells and genetic information
    :type cells_addata: ad.AnnData
    :param well_field_name: The well field name, defaults to "well_id"
    :type well_field_name: str, optional
    :param print_entire_patients: If true, print the entire information about the wells with those double patients, defaults to True
    :type print_entire_patients: bool, optional
    :return: Number of wells with multi donors
    :rtype: int
    """
    counter = 0 
    for well_id in cells_addata.obs[well_field_name].unique():
        if not well_id:
            continue
            
        patient_list = [patient for patient in cells_addata[cells_addata.obs[well_field_name] == well_id].obs.patient.unique() if patient not in [ "doublet","unassigned"]]
        
        if len(patient_list) > 1:
            counter +=1

    if counter and print_entire_patients:
        for well_id in cells_addata.obs[well_field_name].unique():
            if not well_id:
                continue
                
            patient_list = [patient for patient in cells_addata[cells_addata.obs[well_field_name] == well_id].obs.patient.unique() if patient not in [ "doublet","unassigned"]]
            
            if len(patient_list) > 1:
                genu.print_bold("%s - %s" %(well_id, " ".join(patient_list)))
                print("%s" % cells_addata[cells_addata.obs[well_field_name] == well_id].obs.value_counts("patient")) 

    return counter


def get_regex_with_seq_errors(prefix, pattern, suffix):    
    prefix_list = []
    suffix_list = []

    for i in range(len(prefix)):
        prefix_as_list = list(prefix).copy()
        prefix_as_list[i] = "[ACGT]"
        prefix_list.append(prefix_as_list)

    for i in range(len(suffix)):
        suffix_as_list = list(suffix).copy()
        suffix_as_list[i] = "[ACGT]"
        suffix_list.append(suffix_as_list)

    prefix_options = "|".join(["".join(x) for x in prefix_list])
    suffix_options = "|".join(["".join(x) for x in suffix_list])
    # long_regex = "(?:[%s])%s(?:[%s])" % (prefix_options, pattern, suffix_options)
    # short_regex = "(?:[%s])(%s)(?:[%s])" % (prefix_options, pattern, suffix_options)

    long_regex = []
    short_regex = []

    long_regex.append("(?:[%s])%s%s" % (prefix_options, pattern, suffix))
    short_regex.append("(?:[%s])(%s)%s" % (prefix_options, pattern, suffix))
    long_regex.append("%s%s(?:[%s])" % (prefix, pattern, suffix_options))
    short_regex.append("%s(%s)(?:[%s])" % (prefix, pattern, suffix_options))
    
    return [re.compile(i) for i in long_regex], [re.compile(i) for i in short_regex]
    
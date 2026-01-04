import numpy as np
import anndata as ad
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from collections import defaultdict
import sys
from pyvis.network import Network
from IPython.display import display, HTML
import tempfile
import os
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from graphviz import Digraph
from itertools import groupby
import math
from IPython.display import Image, display


pio.renderers.default = "notebook"  
import pandas as pd

class CloneObject(object):
    def __init__(self, adata: ad.AnnData, ordered_cell_types: list, min_transition_distance:float = 0, max_cell_types_distances:dict[str,float] = None):
        self.adata = adata  # Store the anndata object
        self.ordered_cell_types = ordered_cell_types  # Store the predefined order of cell types
        # self.added_cell_types = ["start","unknown","death"]
        # for i in self.added_cell_types:
        #     if i not in self.ordered_cell_types:
        #         self.ordered_cell_types.append(i)
        
        obs_df = adata.obs.copy()  # Work with a copy of obs
        self.obs_df=obs_df
        self.max_cell_types_distances = max_cell_types_distances if max_cell_types_distances is not None else {}
    

        # Ensure required columns exist
        required_columns = ['clone_exp_id', 'exp_id', 'condition', 'sample_day_as_int', 'cell_type', 'top_level_cell_type']
        missing_columns = [col for col in required_columns if col not in obs_df]
        if missing_columns:
            raise ValueError(f"Missing required columns in obs: {missing_columns}")
        
        # Store single-value attributes
        self.clone_exp_id = obs_df['clone_exp_id'].iloc[0]
        self.exp_id = obs_df['exp_id'].iloc[0]
        self.condition = obs_df['condition'].iloc[0]
        
        # Store unique lists and value counts
        self.sample_day_as_int = sorted(obs_df['sample_day_as_int'].unique().tolist())
        self.sample_day_as_int = self.sample_day_as_int  # Add 0 to the beginning
        self.sample_day_as_int_counts = obs_df['sample_day_as_int'].value_counts().to_dict()
        
        self.type_list = obs_df['cell_type'].unique().tolist()
        self.type_counts = obs_df['cell_type'].value_counts().to_dict()
        self.sample_day_type_count = obs_df.groupby(['sample_day_as_int', 'cell_type'], observed=True).size().unstack(fill_value=0)
        self.sample_day_top_level_type_count = obs_df.groupby(['sample_day_as_int', 'top_level_cell_type'], observed=True).size().unstack(fill_value=0)

        self.top_level_cell_type_list = obs_df['top_level_cell_type'].unique().tolist()
        self.top_level_cell_type_counts = obs_df['top_level_cell_type'].value_counts().to_dict()
        
        # Store number of rows
        self.number_of_cells = obs_df.shape[0]
        
        # Store only specific columns for each row
        trajectories_scores_columns = [i for i in obs_df.columns if "downsampled_score" in i]
        self.filtered_obs = obs_df[['sample_day_as_int', 'cell_type', 'top_level_cell_type'] + trajectories_scores_columns].copy()
        
        correlation_distance = squareform(pdist(self.filtered_obs[trajectories_scores_columns], metric='euclidean'))
        self.cells_correlation_df = pd.DataFrame(correlation_distance, index=self.filtered_obs.index, columns=self.filtered_obs.index)        
        self.filtered_obs["delta_from_start"] = self.filtered_obs["sample_day_as_int"] - self.filtered_obs["sample_day_as_int"].min()
        
        # Create transition matrices
        self.min_distance_per_cell = []
        self.transition_matrices = self._compute_transitions(min_transition_distance)
        self.total_transition_matrix = sum(self.transition_matrices.values())


        if self.is_cmp_multilineage():
            self.lineage_type = "cmp_multilineage"
        elif self.is_mebemp_multilineage():
            self.lineage_type = "mebemp_multilineage"
        elif self.has_bemp_trajectory():
            self.lineage_type = "bemp_lineage"
        elif self.has_mep_trajectory():
            self.lineage_type = "mep_lineage"
        elif self.has_gmp_trajectory():
            self.lineage_type = "gmp_lineage"
        else:
            self.lineage_type = "unknown_lineage"

    def get_cells_by_type(self,cell_type, day):
        """
        Get cells of a specific type on a specific day.
        :param cell_type: The cell type to filter by.
        :param day: The day to filter by.
        :return: DataFrame of cells of the specified type on the specified day.
        """
        return self.filtered_obs[(self.filtered_obs['cell_type'] == cell_type) & (self.filtered_obs['sample_day_as_int'] == day)].index.to_list()

    def has_connection(self,src,dst):
        if self.multi_day_clone():
            if self.total_transition_matrix.loc[src, dst] > 0:
                return True
        return False

    def multi_day_clone(self):
        """
        Check if the clone has multiple days.
        :return: True if multiple days, False otherwise.
        """
        return len(self.sample_day_as_int) > 1
    
    def has_bemp_trajectory(self):
        if len(set(self.type_list) & set(["BEMP", "BEP","Mast","MastP","Eosinophil", "Basophils"])):
            return True
        
        return False
    
    def has_mep_trajectory(self):
        if len(set(self.type_list) & set(["MEP", "ProE", "BasoE", "PolyE", "EryEnuc", "OrthoE", "Ery_progenitors", "MEP", "MK", "MKP"])):
            return True
        
        return False

    def has_mk_trajectory(self):
        if len(set(self.type_list) & set(["MK", "MKP"])):
            return True
        
        return False
    
    def has_ery_trajectory(self):
        if len(set(self.top_level_cell_type_list) & set(["Ery"])):
            return True
        
        return False
    
    def has_mast_trajectory(self):
        if len(set(self.type_list) & set(["Mast", "MastP"])):
            return True
        
        return False
    
    def has_eosinophil_trajectory(self):
        if "Eosinophil" in self.type_list:
            return True
            
        return False
    
    def has_neutorphil_trajectory(self):
        if "Neutrophils" in self.top_level_cell_type_list:
            return True
            
        return False

    def has_monocyte_trajectory(self):
        if "Monocyte" in self.top_level_cell_type_list:
            return True
            
        return False
    
    def has_basophil_trajectory(self): 
        if "Basophils" in self.type_list:
            return True
            
        return False
    
    def is_bemp_multipotent(self):
        n = 0
        n += 1 if self.has_mast_trajectory() else 0
        n += 1 if self.has_eosinophil_trajectory() else 0
        n += 1 if self.has_basophil_trajectory() else 0

        return n > 1
    
    def is_bemp_unipotent(self):
        n = 0
        n += 1 if self.has_mast_trajectory() else 0
        n += 1 if self.has_eosinophil_trajectory() else 0
        n += 1 if self.has_basophil_trajectory() else 0

        return n == 1


    def is_mep_multipotent(self):
        if self.has_mk_trajectory() and self.has_ery_trajectory():
            return True
        
        return False
        
    def is_mep_unipotent(self):
        if (self.has_mk_trajectory() or self.has_ery_trajectory()) and (not self.is_mep_multipotent()):
            return True
        
        return False
    
    def has_gmp_trajectory(self):
        if len(set(self.type_list) & set(["GMP", "MonoP", "Monocyte", "Neutrophils", "Neutrophils-L"])):
            return True
        
        return False
    
    def has_mebemp_cell_type(self):
        if "MEBEMP" in self.type_list:
            return True
        return False
    
    def is_mebemp_unipotent(self):
        return (self.has_bemp_trajectory() or self.has_mep_trajectory()) and not self.is_mebemp_multilineage()
    
    def has_cell_in_earlies_timepoint(self, cell_type):
        if cell_type not in self.sample_day_top_level_type_count.columns:
            return False
        return self.sample_day_top_level_type_count.iloc[0][cell_type] > 0
        
    def is_mebemp_multilineage(self):
        lineages = 0
        if self.has_bemp_trajectory():
            lineages += 1
        if self.has_mep_trajectory():
            lineages += 1

        return lineages > 1
    
    def is_cmp_multilineage(self):
        lineages = 0
        if self.has_bemp_trajectory() or self.has_mep_trajectory() or self.has_mebemp_cell_type():
            lineages += 1
        if self.has_gmp_trajectory():
            lineages += 1

        return lineages > 1
    
    def has_cell_type_in_day(self, cell_type, day):
        if cell_type not in self.sample_day_top_level_type_count.columns:
            return False
        return self.sample_day_top_level_type_count.loc[day, cell_type] > 0
    

    def get_cells_pairs_by_cell_type(self, cell_type1, cell_type2):
        # has several days
        if not self.multi_day_clone():
            return []
        
        clone_cells_types = self.obs_df["cell_type"].unique().tolist()

        # the cell type we want
        if cell_type1 not in clone_cells_types or cell_type2 not in clone_cells_types:
            return []
        
        days_with_cell_type = self.sample_day_type_count.index[self.sample_day_type_count[cell_type1] > 0]

        valid_cells = []
        
        for day in days_with_cell_type:
            future_days = [i for i in self.sample_day_as_int if i > day]
            
            # no future_days
            if len(future_days) ==0:
                continue 
            
            day_cells_counter = self.sample_day_type_count.loc[day]
            # we have cell type 1 in this day, we don't have cell type 2 and we will have cell type 2 future
            if day_cells_counter[cell_type1] > 0 and (day_cells_counter[cell_type2] == 0 or cell_type2 == cell_type1) and self.sample_day_type_count.loc[future_days,cell_type2].sum() > 0:
                cell_type1_cells = self.obs_df.loc[(self.obs_df['sample_day_as_int'] == day) & (self.obs_df['cell_type'] == cell_type1)].index
                cell_type2_cells = self.obs_df.loc[(self.obs_df['sample_day_as_int'].isin(future_days)) & (self.obs_df['cell_type'] == cell_type2)].index

                # get all combinations of cell type 1 and cell type 2
                for cell1 in cell_type1_cells:
                    for cell2 in cell_type2_cells:
                        valid_cells.append((cell1, cell2))
                
        return valid_cells            
            
        

    def get_cells_by_query(self, day_has_top_level_cell_type, day_forbidden_top_level_cell_types = [], previous_days_forbidden_top_level_cell_types = [], future_days_top_level_cell_types = [], future_days_forbidden_top_level_cell_types=[],
                           use_top_level_cell_type=True):
        if not self.multi_day_clone():
            return []
        
        if use_top_level_cell_type:
            cell_type_list = self.top_level_cell_type_list
            sample_day_to_cell_type_count = self.sample_day_top_level_type_count
            cell_type_column = 'top_level_cell_type'
            
        else:
            cell_type_list = self.obs_df['cell_type'].unique().tolist()
            sample_day_to_cell_type_count = self.sample_day_type_count
            cell_type_column = 'cell_type'
        
        # check we actually have this cell type
        if day_has_top_level_cell_type not in cell_type_list:
            return []
        
        days_with_cell_type = sample_day_to_cell_type_count.index[sample_day_to_cell_type_count[day_has_top_level_cell_type] > 0]
        valid_days = []
        for day in days_with_cell_type:
            day_cells_counter = sample_day_to_cell_type_count.loc[day]
            current_days_cell_types = day_cells_counter.index.intersection(day_forbidden_top_level_cell_types)
            if day_cells_counter[current_days_cell_types].sum() >0:
                continue
            
            # Check previous days if asked for
            if len(previous_days_forbidden_top_level_cell_types):
                previous_days = [i for i in self.sample_day_as_int if i < day]
                previous_days_cell_counter = sample_day_to_cell_type_count.loc[previous_days].sum(axis=0)
                previous_days_cell_types = previous_days_cell_counter.index.intersection(previous_days_forbidden_top_level_cell_types)
                if previous_days_cell_counter[previous_days_cell_types].sum() > 0 :
                    continue

            # Check future days if asked for 
            # future_days_top_level_cell_types is a list of lists, we need at least 1 cell from each sublist to appear 
            if len(future_days_top_level_cell_types):
                future_days = [i for i in self.sample_day_as_int if i  > day]
                # no future days, can't fill the requirment
                if len(future_days) == 0:
                    continue
                
                # has future days, make sure they have the cell type 
                future_days_cell_counter = sample_day_to_cell_type_count.loc[future_days].sum(axis=0)
                future_days_cell_types = future_days_cell_counter.index.intersection(future_days_forbidden_top_level_cell_types)
                if future_days_cell_counter[future_days_cell_types].sum() > 0: 
                    continue
            
                sublists_found = 0
                for sublist in future_days_top_level_cell_types:
                    future_days_cell_types = future_days_cell_counter.index.intersection(sublist)
                    if future_days_cell_counter[future_days_cell_types].sum() > 0: # don't have
                        sublists_found += 1
            
                if sublists_found != len(future_days_top_level_cell_types):
                    continue

            valid_days.append(day)

        return self.obs_df.index[np.logical_and(self.obs_df['sample_day_as_int'].isin(valid_days), self.obs_df[cell_type_column] == day_has_top_level_cell_type)]

    def _compute_transitions(self, min_transition_distance):
        transition_matrices = {}
        
        for i in range(len(self.sample_day_as_int) - 1):
            day_x, day_y = self.sample_day_as_int[i], self.sample_day_as_int[i + 1]
            df_x_cells = self.obs_df[self.obs_df['sample_day_as_int'] == day_x].index
            df_y_cells = self.obs_df[self.obs_df['sample_day_as_int'] == day_y].index           
            transition_matrix = pd.DataFrame(0, index=self.ordered_cell_types, columns=self.ordered_cell_types)

            x_y_correlation = self.cells_correlation_df.loc[df_x_cells, df_y_cells]
            x_types = self.obs_df.loc[df_x_cells, "cell_type"]
            y_types = self.obs_df.loc[df_y_cells, "cell_type"]

            # find most common ancestor for each y cell
            x_cells_per_y = x_y_correlation.idxmin(axis=0)
            self.min_distance_per_cell.append(x_y_correlation.min(axis=0))
            # this is the connection between y to x, so y is the advance
            for i in range(x_cells_per_y.shape[0]):
                y = x_cells_per_y.index[i]
                x = x_cells_per_y.values[i]
                y_type = y_types[y]
                x_type = x_types[x]
                if self.max_cell_types_distances and y_type in self.max_cell_types_distances:
                    transition_cutoff = self.max_cell_types_distances[y_type]
                else:
                    transition_cutoff = min_transition_distance

                if x_y_correlation.loc[x,y] < transition_cutoff:
                    continue

                transition_matrix.loc[x_type, y_type] += 1

            # x_cells_per_y_couples = zip(x_types[x_cells_per_y],y_types)

            # Count the transitions between source (df_x) and destination (df_y)
            # for x,y in x_cells_per_y_couples:
            #     transition_matrix.loc[x, y] += 1

            # Store the transition matrix for the pair of days
            transition_matrices[(day_x, day_y)] = transition_matrix
        
        return transition_matrices
                

    # ───────────────────────────────────────── helpers ─────────────────────────────────────────
    def _build_nodes(self, cell_type_colors):
        df = self.sample_day_type_count
        nodes = []
        for day, row in df.iterrows():
            for cell_type, count in row.items():
                if count == 0:
                    continue
                size = math.log2(count + 1) # minimum size of 1
                nodes.append({
                    "id":  f"{day}_{cell_type}",
                    "day": day,
                    "size": size,
                    "color": cell_type_colors.get(cell_type, "#999999")
                })
        return nodes

    def _build_edges(self, cell_type_colors, min_penwidth=0.5, max_penwidth=4):
        edges = []
        for (day_x, day_y), mat in self.transition_matrices.items():
            for src_ct in mat.index:
                if mat.loc[src_ct, :].sum() == 0:
                    continue
                for tgt_ct in mat.columns:
                    cnt = mat.at[src_ct, tgt_ct]
                    if cnt == 0:
                        continue
                    penwidth = max(min(math.log2(cnt), max_penwidth), min_penwidth)
                    edges.append({
                        "source_id": f"{day_x}_{src_ct}",
                        "target_id": f"{day_y}_{tgt_ct}",
                        "color": cell_type_colors.get(src_ct, "#999999"),
                        "penwidth": penwidth
                    })
        return edges

    def make_trajectory_graph(
        self,
        cell_type_colors,
        fig_size=(12, 6),
        min_node_size=0.2,
        min_edge_penwidth=0.5,
        max_edge_penwidth=4,
        output_folder=None,
        output_file=None,
        add_first_day=False,
        nodesep="0.1",
        ranksep="1"
    ):
        nodes = self._build_nodes(cell_type_colors)
        edges = self._build_edges(cell_type_colors, min_penwidth=min_edge_penwidth, max_penwidth=max_edge_penwidth)
        
        if add_first_day:
            cell_type_colors = cell_type_colors.copy()
            cell_type_colors["Start"] = "#ffffff"
            first_day = min(n["day"] for n in nodes)
            start_node_id = "0_Start"
            nodes.append({
                "id": start_node_id,
                "day": 0,
                "size": 1,
                "color": "#ffffff"
            })

            for n in nodes:
                if n["day"] == first_day:
                    edges.append({
                        "source_id": start_node_id,
                        "target_id": n["id"],
                        "color": "#aaaaaa",
                        "penwidth": str(min_edge_penwidth)
                    })

        g = Digraph("G")
        
        g.attr(rankdir="LR",
               splines="true",
               size=f"{fig_size[0]},{fig_size[1]}",
               dpi="300",
               nodesep=nodesep,
               ranksep=ranksep)

        # Group nodes by day
        for day in sorted({n["day"] for n in nodes}):
            with g.subgraph(name=f"cluster_day{day}") as c:
                c.attr(rank="same", label=f"Day {day}", labelloc="b", color="lightgrey", style="dashed")
                
                # Sort nodes for the current day based on the order of cell types in cell_type_colors.keys()
                sorted_nodes = sorted(
                    (n for n in nodes if n["day"] == day),
                    key=lambda nd: list(cell_type_colors.keys()).index(nd["id"].split("_")[1])
                )
                
                for n in sorted_nodes:
                    c.node(
                        n["id"],
                        label="",  # Removes the text label
                        style="filled",
                        fillcolor=n["color"],
                        width=str(min_node_size * n["size"]),
                        height=str(min_node_size * n["size"]),
                        fixedsize="true",
                        shape="circle"
                    )

        # Add edges
        for e in edges:
            g.edge(e["source_id"],
                   e["target_id"],
                   color=e["color"],
                   penwidth=str(e["penwidth"]))

        
        if output_folder is None or output_file is None:
            # Display on screen (PNG)
            png_bytes = g.pipe(format="png")
            display(Image(png_bytes))
            
        # ------------------
        else:
            g.render(filename=output_file, directory=output_folder, format="pdf", cleanup=True)
            
        return g

    def make_node_size_legend(
        self,
        output_file="node_size_legend",
        output_folder="./",
        min_node_size=0.2,
        base_color="#ffffff",
        edge_color="#000000",
        label_fontsize=10,
        max_size = 8
    ):
        g = Digraph("Legend")
        g.attr(rankdir="LR",          # Left to right layout
            splines="false",
            dpi="300",
            nodesep="0.05",        # Less space between nodes
            ranksep="0.1")

        # Node counts for size scaling
        node_counts = [2**i for i in range(0,max_size)]
        
        for count in node_counts:
            size = math.log2(count + 1)  # match logic from main graph
            scaled_size = min_node_size * size
            node_id = f"legend_{count}"

            g.node(
                node_id,
                label=str(count),  # No internal label
                # xlabel=str(count),  # Label shown below the node
                style="filled",
                fillcolor=base_color,
                color=edge_color,
                fontcolor="black",
                fontsize=str(label_fontsize),
                width=str(scaled_size),
                height=str(scaled_size),
                fixedsize="true",
                shape="circle"
            )

        # Connect nodes invisibly to force horizontal alignment
        for i in range(len(node_counts) - 1):
            g.edge(f"legend_{node_counts[i]}", f"legend_{node_counts[i+1]}", style="invis")

        g.render(filename=output_file, directory=output_folder, format="pdf", cleanup=True)
        
    def __repr__(self):
        return (f"CloneObject(exp_id={self.exp_id}, condition={self.condition}, clone_exp_id={self.clone_exp_id}, "
                f"unique_sample_days={self.sample_day_as_int}, type_list={self.type_list}, "
                f"top_level_cell_types={self.top_level_cell_type_list}, number_of_cells={self.number_of_cells}")
    
    def __lt__(self, other):
        return self.number_of_cells < other.number_of_cells
    
    def __gt__(self, other):
        return self.number_of_cells > other.number_of_cells



def convert_full_anndata_to_clones_objects(full_cells: ad.AnnData, ordered_cell_types, max_cell_types_distances=None) -> list:
    clone_list = []
    for clone_exp_id in tqdm.tqdm(full_cells.obs.clone_exp_id.unique()):
        clone_list.append(CloneObject(full_cells[full_cells.obs.clone_exp_id == clone_exp_id], ordered_cell_types, max_cell_types_distances=max_cell_types_distances))

    return sorted(clone_list, reverse=True)

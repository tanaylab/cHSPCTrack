import os
import pickle 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
import metacells as mc
import scipy
import tqdm
from sklearn.cluster import KMeans


import multiprocessing as mp
import concurrent.futures

import matplotlib.ticker as mticker

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


import seaborn as sb
from adjustText import adjust_text
from numpy.lib.stride_tricks import sliding_window_view
from scipy.cluster.hierarchy import linkage

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from typing import Dict, List, Tuple, Union


from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

numpy2ri.activate()  
pandas2ri.activate() 

np.random.seed(42)


EPSILON = 1e-5


def get_anndata_fp(anndata:ad.AnnData, layer:str="__x__"):
    cells_array = mc.ut.get_vo_proper(anndata, name=layer).toarray() 
    total_cells = mc.ut.get_o_numpy(anndata, name=layer, sum=True)
    fp = cells_array/total_cells[:, np.newaxis]
    fp_df = pd.DataFrame(fp, index=anndata.obs_names, columns=anndata.var_names, dtype="float32")
    return fp_df



def get_lgc(anndata, epsilon=1e-5, layer = "__x__", log_base=2):   
    fp = get_anndata_fp(anndata, layer=layer)
    lgc = np.log(fp + epsilon) / np.log(log_base)

    return lgc


def plot_generic_dict_of_colors(
    colors_dict,
    ncol=1,
    output_path=None,
    fontsize=20,
    padding=0.4,
    min_size=(2.5, 1.5),  # Minimum width, height in inches
    dpi=300,
    title=None
):
    """
    Plot just a legend based on a color dictionary.

    Parameters:
    - colors_dict: dict of {label: color}
    - ncol: number of columns in the legend
    - output_path: if given, saves to this path
    - fontsize: font size for labels
    - padding: extra padding (in inches) around the legend
    - min_size: minimum figure size in inches (width, height)
    - dpi: resolution for saving
    """
    # Create dummy figure
    fig, ax = plt.subplots()
    ax.axis("off")

    # Build legend handles
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in colors_dict.items()
    ]

    # Draw the legend
    legend = ax.legend(
        handles=legend_handles,
        loc='center',
        fontsize=fontsize,
        ncol=ncol,
        frameon=False,
        title=title,
    )
    
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(fontsize)

    # Render and measure size
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # Final figure size
    width = max(bbox.width + padding, min_size[0])
    height = max(bbox.height + padding, min_size[1])
    fig.set_size_inches(width, height)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def calc_diff_expression_from_egc(group1_lgc:pd.DataFrame, group2_lgc:pd.DataFrame, obs1:pd.Index, obs2:pd.Index, mat:pd.DataFrame, diff_thresh:float = 1, pval_thresh:float = 0.05, calculate_p_val=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    calculate rlgc between two groups 

    take only those above diff_thresh  as optional genes
    calc how much umis per metacells 
    for each genes have a 2x2
        gene_in_group1 | gene in group 2
        total group 1  | tot group 2
        
    chiseq test 

    take only those above specific threshold 
    """
    total_number_of_umis = mat.sum(axis=1)
    total_group1 = total_number_of_umis.loc[obs1].sum()
    total_group2 = total_number_of_umis.loc[obs2].sum()

    rlgc = group1_lgc - group2_lgc
    
    # high diff between them
    genes_with_high_diff = rlgc.index[abs(rlgc) >= diff_thresh]

    # for each gene calculate the chi square test
    chi_square_results = pd.Series(0, index=mat.columns)
    if calculate_p_val:
        mat1 = mat.loc[obs1]
        mat2 = mat.loc[obs2]
        genes_1_sum = mat1.sum(axis=0)
        genes_2_sum = mat2.sum(axis=0)
        for gene_i in range(mat.shape[1]):
            g1_sum = genes_1_sum.iloc[gene_i]
            g2_sum = genes_2_sum.iloc[gene_i]

            if g1_sum == 0 and g2_sum.sum() == 0:
                chi_square_results.iloc[gene_i] = None
                continue

            obs = np.array([[g1_sum, g2_sum], [total_group1 - g1_sum, total_group2 - g2_sum]])
            _, p, _, _ = scipy.stats.chi2_contingency(obs)
            # _, p= scipy.stats.fisher_exact(obs)
            chi_square_results.iloc[gene_i] = p 

    
    # combine the chi_square_results series with the rlgc series
    diff_expression_genes = pd.concat([rlgc, chi_square_results.loc[rlgc.index], group1_lgc.loc[rlgc.index], group2_lgc.loc[rlgc.index]], axis=1)
    diff_expression_genes.columns = ["rlgc", "pval", "exp_obs1", "exp_obs2"]
    diff_expression_genes["uncorrected_pval"] = diff_expression_genes.pval
    diff_expression_genes.loc[diff_expression_genes.pval.isna(),"pval"] = 1
    if calculate_p_val:
        diff_expression_genes.pval = scipy.stats.false_discovery_control(diff_expression_genes.pval)
    diff_expression_genes = diff_expression_genes.sort_values("rlgc")
    
    # now return the genes which are above the threshold and have a pval below the threshold, and also return the entire diff_expression_genes
    return diff_expression_genes[diff_expression_genes.index.isin(genes_with_high_diff) & (diff_expression_genes.pval < pval_thresh)], diff_expression_genes


def calc_diff_expression_genes_cells(cells_anndata:ad.AnnData, obs1:pd.Index, obs2:pd.Index, diff_thresh:float = 1, pval_thresh:float = 0.05, epsilon:float = 1e-5, calculate_p_val=False, cells_umis_df=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    calculate the differential expression between two groups of metacells

    first take the egc --> we need this because we are working with metacells
    calculate the lgc for each group 
    send it to the calc_diff_expression_from_egc function 
    """
    if cells_umis_df is None:
        cells_umis_df = mc.ut.get_vo_frame(cells_anndata[obs1.union(obs2)])
    group1_umis = cells_umis_df.loc[obs1].sum() 
    group2_umis = cells_umis_df.loc[obs2].sum()

    group1_fp = group1_umis / group1_umis.sum() + epsilon
    group2_fp = group2_umis / group2_umis.sum() + epsilon

    group1_lgc = np.log2(group1_fp)
    group2_lgc = np.log2(group2_fp)

    subset_mat = cells_umis_df.loc[obs1.union(obs2)]

    return calc_diff_expression_from_egc(group1_lgc, group2_lgc, obs1, obs2, subset_mat, diff_thresh=diff_thresh, pval_thresh=pval_thresh, calculate_p_val=calculate_p_val)


def plot_de_graph(de_genes_df:pd.DataFrame, min_pval:float = 0.05, min_rlgc_to_plot:float= 1, figsize=(8,8), max_genes_names=20, xylim=(-17,-6), xlabel="Group 1", ylabel="Group 2",
                  genes_to_ignore:pd.Index=pd.Index([]), min_expression = -16, output_path=None, file_name=None, exp_to_plot=-16, genes_to_show = [], title=False, fontsize=14, txt_fix_dict={},
                  axis_formatter=None):
    plt.figure(figsize=figsize)

    group1_obs = de_genes_df["exp_obs1"]
    group2_obs = de_genes_df["exp_obs2"]
    pval = de_genes_df["pval"]
    
    sign_genes_in_obs1 = de_genes_df.index[(pval < min_pval) & (de_genes_df["rlgc"] > min_rlgc_to_plot) & ((group1_obs > min_expression) | (group2_obs > min_expression))]
    sign_genes_in_obs2 = de_genes_df.index[(pval < min_pval) & (de_genes_df["rlgc"] < -1 * min_rlgc_to_plot) & ((group1_obs > min_expression) | (group2_obs > min_expression))]
    sign_genes_in_obs1 = sign_genes_in_obs1.difference(genes_to_ignore)
    sign_genes_in_obs2 = sign_genes_in_obs2.difference(genes_to_ignore)
    
    sb.scatterplot(x=group1_obs, y=group2_obs,  s=20, c="gray", alpha=0.5)
    sb.scatterplot(data=de_genes_df.loc[sign_genes_in_obs1], x="exp_obs1", y="exp_obs2", s=20, c="red")
    sb.scatterplot(data=de_genes_df.loc[sign_genes_in_obs2], x="exp_obs1", y="exp_obs2", s=20, c="blue")
    
    if max_genes_names >0:
        sign_genes_names_obs1 = de_genes_df.loc[sign_genes_in_obs1].abs().sort_values(by="rlgc").tail(max_genes_names).index
        sign_genes_names_obs2 = de_genes_df.loc[sign_genes_in_obs2].abs().sort_values(by="rlgc").tail(max_genes_names).index

        sign_genes_names = sign_genes_names_obs1.union(sign_genes_names_obs2)
        
        if exp_to_plot != -16:
            sign_genes_names_based_on_exp1 = sign_genes_in_obs1[de_genes_df.loc[sign_genes_in_obs1]["exp_obs1"] > exp_to_plot]
            sign_genes_names_based_on_exp2 = sign_genes_in_obs2[de_genes_df.loc[sign_genes_in_obs2]["exp_obs2"] > exp_to_plot]
            sign_genes_names_based_on_exp1 = sign_genes_names_based_on_exp1.difference(genes_to_ignore)
            sign_genes_names_based_on_exp2 = sign_genes_names_based_on_exp2.difference(genes_to_ignore)
            
            exp1_used_n  = max_genes_names - len(sign_genes_names_based_on_exp1)
            exp2_used_n  = max_genes_names - len(sign_genes_names_based_on_exp2)
            
            sign_genes_names = sign_genes_names_based_on_exp1.union(sign_genes_names_obs1[:exp1_used_n]).union(sign_genes_names_based_on_exp2.union(sign_genes_names_obs2[:exp2_used_n]))
            
        texts = [] 
        for name in sign_genes_names.union(genes_to_show):
            texts.append(plt.text(
                de_genes_df.loc[name]["exp_obs1"] ,   # X position (offset slightly to the right)
                de_genes_df.loc[name]["exp_obs2"] , # Y position
                name,                            # Text (e.g., species name)
                fontsize=fontsize,                      # Font size
                ha='center', va='center'
            ))

        
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.xticks(range(xylim[0], xylim[1]+1, 2))
    plt.yticks(range(xylim[0], xylim[1]+1, 2))

    plt.plot(xylim, xylim, "--", c="black", alpha=0.5)
    plt.plot([xylim[0],xylim[1]], [xylim[0]-1,xylim[1]-1], "-.", c="gray", alpha=0.5)
    plt.plot([xylim[0],xylim[1]], [xylim[0]+1,xylim[1]+1], "-.", c="gray", alpha=0.5)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if max_genes_names > 0:
        adjust_text(
            texts,
            de_genes_df["exp_obs1"].values,
            de_genes_df["exp_obs2"].values,
            force_text=5.0,
            force_points=2.5,
            expand_text=(2.0, 2.0),
            expand_points=(1.5, 1.5),
            avoid_self=True,
            lim=250,
        )
        
    # 3. Manual fix for the stubborn ones
    for txt in texts:
        if txt.get_text() in txt_fix_dict:
            x, y = txt.get_position()
            x_f, y_f = txt_fix_dict[txt.get_text()]
            txt.set_position((x + x_f,y + y_f))
            
    ax = plt.gca()
    for txt in texts:
        gene = txt.get_text()
        x0 = de_genes_df.loc[gene, "exp_obs1"]   # point
        y0 = de_genes_df.loc[gene, "exp_obs2"]
        x1, y1 = txt.get_position()             # final text position

        ax.annotate(
            "", xy=(x0, y0), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
        )

    if axis_formatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    
    if output_path:   
        plt.savefig(os.path.join(output_path, "%s.pdf" % file_name), dpi=1200, format="pdf", bbox_inches="tight")

    plt.show()
    plt.close()
    

def load_pickle(file_path, binary=True):
    r_type = "rb" if binary else "r"

    assert os.path.exists(file_path), "File: %s doesn't exists" %file_path

    with open(file_path, r_type) as f:
        data = pickle.load(f)

    return data

read_from_pickle = load_pickle

def save_to_pickle(obj_to_save, file_path, binary=True):
    r_type = "wb" if binary else "w"

    assert os.path.exists(os.path.dirname(file_path)), "Folder: %s doesn't exists" %os.path.dirname(file_path)
    with open(file_path, r_type) as f:
        pickle.dump(obj_to_save, f)





def downsample_cells(anndata: ad.AnnData, target_umis:int = None, quantile: float=None, cut_downsampled:bool = False, random_seed:int = 42, downsample_min_cell_quantile:float = 0):
    assert target_umis or quantile, "Either target_umis or quantile should be given"
    if quantile:
        target_umis = np.quantile(mc.ut.get_o_numpy(anndata, name="__x__", sum=True), quantile)

    target_umis = int(target_umis)
    
    print(f"Downsampling to {target_umis} UMIs")
    mc.tl.downsample_cells(adata = anndata, downsample_min_samples = target_umis, random_seed = random_seed, downsample_min_cell_quantile=downsample_min_cell_quantile,  downsample_max_cell_quantile=1)
    if cut_downsampled:
        number_of_cells_to_remove = sum(mc.ut.get_o_numpy(anndata, name="__x__", sum=True) < target_umis)
        print("Cutting downsampled cells, going to remove %d cells (%.2f%%)" % (number_of_cells_to_remove, 100 * number_of_cells_to_remove / anndata.shape[0]))
        anndata = anndata[mc.ut.get_o_numpy(anndata, name="__x__", sum=True) >= target_umis]
    return anndata

def get_metacells_umis_df(cells_anndata:ad.AnnData, group_name="metacell_name",layer="__x__"):
    # 1) cells×genes matrix as CSR float32
    X = cells_anndata.layers[layer] if layer in cells_anndata.layers else cells_anndata.X
    
    if scipy.sparse.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
    else:
        X = scipy.sparse.csr_matrix(np.asarray(X, dtype=np.float32, order="C"))

    # 2) group codes for cells
    grp = cells_anndata.obs[group_name]
    codes, uniques = pd.factorize(grp, sort=False)
    valid = codes >= 0
    if not np.all(valid):  # drop NaNs
        X = X[valid]
        codes = codes[valid]

    n_cells, n_genes = X.shape
    n_groups = len(uniques)

    # 3) one-hot aggregator G (groups × cells)
    G = scipy.sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (codes, np.arange(n_cells))),
        shape=(n_groups, n_cells)
    )
    G.sum_duplicates()

    # 4) grouped sums = G @ X  → groups×genes
    grouped = G @ X

    # 5) back to DataFrame (sparse)
    df = pd.DataFrame.sparse.from_spmatrix(
        grouped,
        index=pd.Index(uniques, name=group_name),
        columns=cells_anndata.var_names
    ).astype(np.float32, copy=False)
    
    df = df.loc[cells_anndata.obs[group_name].unique().sort_values()] 

    return df



def perform_2d_sliding_window_smoothing(df, sliding_window_size, func=np.mean):

    sliding_window_smoothing = func(sliding_window_view(df, 1+ 2*sliding_window_size, axis=0), axis=-1)

    sliding_window_smoothing_l = []
    for i in range(sliding_window_size):
        sliding_window_smoothing_l.append(sliding_window_smoothing[0].reshape(1, -1))

    sliding_window_smoothing_l.append(sliding_window_smoothing)
    for i in range(sliding_window_size, 0, -1):
        sliding_window_smoothing_l.append(sliding_window_smoothing[-1].reshape(1, -1))


    sliding_window_smoothing = np.concatenate(sliding_window_smoothing_l,axis=0)
    smooth_df = pd.DataFrame(sliding_window_smoothing, index=pd.Categorical(df.index, categories=df.index), columns=df.columns)
    assert df.shape[0] == smooth_df.shape[0]
    return smooth_df


def add_gene_modules_score(metacells_anndata: ad.AnnData, modules_to_score_dict: dict[str, list[str]], layer=""):
    for module in modules_to_score_dict:
        if layer == "":
            metacell_score = get_gene_module_expression(metacells_anndata, modules_to_score_dict[module])
            mc.ut.set_o_data(metacells_anndata, "%s_score" % module, metacell_score)
        else:
            metacell_score = get_gene_module_expression(metacells_anndata, modules_to_score_dict[module], layer=layer)
            mc.ut.set_o_data(metacells_anndata, "%s_%s_score" % (module, layer), metacell_score)

    return metacells_anndata

def get_summed_gene_module_expression(anndata:ad.AnnData, genes:List[str], layer:str ="__x__", epsilon:float =1e-5, log_base:int=2, fp =None):
    if fp is None:
        fp = get_anndata_fp(anndata, layer=layer) 

    diff_genes = set(genes) - set(anndata.var_names)
    if len(diff_genes):
        print("Some genes don't appear in the metacells object, ignoring them: %s" %",".join(list(diff_genes)))
        genes = anndata.var_names.intersection(genes)
    
    fp_genes = fp.loc[:,genes]
    
    expression = pd.Series(np.sum(fp_genes, axis=1) + epsilon, index=anndata.obs_names)
    if log_base == 1:
        return expression
    expression = np.log(expression) / np.log(log_base)
    return expression



def add_gene_modules_summed_score(metacells_anndata: ad.AnnData, modules_to_score_dict: dict[str, list[str]], layer="", epsilon=1e-5, log_base=2,suffix=""):
    if suffix != "":
        suffix = "_%s" % suffix
    if layer == "":
            fp = get_anndata_fp(metacells_anndata) 
            
    else:
        fp = get_anndata_fp(metacells_anndata, layer=layer)
    for module in modules_to_score_dict:
        metacell_score = get_summed_gene_module_expression(metacells_anndata, modules_to_score_dict[module], layer=layer, fp=fp, epsilon=epsilon, log_base=log_base)
            
        if layer == "":
            mc.ut.set_o_data(metacells_anndata, "%s_score%s" % (module, suffix), metacell_score)
        else:   
            mc.ut.set_o_data(metacells_anndata, "%s_%s_score%s" % (module, layer, suffix), metacell_score)
                
    return metacells_anndata




def rand_jitter(arr, jitter_size = 0.01):
    stdev = jitter_size * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=1, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, jitter_size=0.01, **kwargs):
    return plt.scatter(rand_jitter(x, jitter_size), rand_jitter(y, jitter_size), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def run_multiprocess(func, args_list, num_workers=None, thread_pool=False):
    # threads should be used in heavy IO
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 10)

    if thread_pool:
        # Use ThreadPoolExecutor for threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm.tqdm(executor.map(func, args_list), total=len(args_list)))
    else:
        # Use multiprocessing.Pool for processes
        with mp.Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap(func, args_list), total=len(args_list)))

    return results


def get_linekage(data, method="ward", metric="euclidean",optimal_ordering=True):
    """
    cg = sb.clustermap(data,
                   row_linkage=Z,
                   col_linkage=Z,
                   method=None,  # disable internal clustering
                   metric=None)  # so it does not override your linkage
    """
    return linkage(data, method=method, metric=metric, optimal_ordering=optimal_ordering)


def plot_continuous_colorbar(
    vmin, vmax, label,
    output_path=None,
    filename="colorbar",
    cmap=plt.cm.gray_r,       # default colormap
    figsize=(1.5, 4),
    fontsize=17,
    labelpad=15,
    log2_ticks=False          # if True, use integer ticks from vmin→vmax
):
    """Plots a continuous colorbar (linear or log2-style)."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax)

    if log2_ticks:
        ticks = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(t)) for t in ticks])

    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(label, rotation=90, labelpad=labelpad, fontsize=fontsize)
    cbar.ax.yaxis.set_label_position("left")
    plt.tight_layout()

    if output_path:
    
        plt.savefig(os.path.join(output_path, f"{filename}.pdf"),
                    bbox_inches="tight", dpi=1200)
    plt.show()
    
    
    
def order_by_slanter(df, cluster_rows=True, cluster_cols=True):
    rows_order, cols_order = get_slanter_order(df.to_numpy(), cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    ordered_data = df.iloc[rows_order,cols_order] 
    return ordered_data

def get_slanter_order(df, cluster_rows=True, cluster_cols=True):
    r_slanter = importr("slanter")
    df= df.copy()
    
    min_value = 0
    if df.min().min() < 0:
        min_value = df.min().min()
        df += -1 * min_value
    
    r_slanter_results = r_slanter.slanted_orders(df, order_rows=cluster_rows, order_cols=cluster_cols, max_spin_count =30)
    rows_order = np.array(r_slanter_results.rx("rows")).reshape(-1) - 1
    cols_order = np.array(r_slanter_results.rx("cols")).reshape(-1) - 1

    # from scipy.spatial.distance import pdist
    # dist_matrix = pdist(df[rows_order,cols_order].to_numpy())
    # clusters = r_slanter.oclust(dist_matrix, method="ward.D2")

    return rows_order, cols_order


def plot_umap(metacells_adata, metadata_to_color=None, max_color_value = None, metadata_color_dict=None, title="",
              figsize=(12,12), s=30, output_path=None, data_series_to_color=None, palette="vlag", vlim=(), linewidth=0.5, remove_legend=False, xlabel="", ylabel=""):
    if metadata_to_color:
        data = metacells_adata.obs[metadata_to_color].copy()

        if max_color_value:    
            data[data > max_color_value] = max_color_value
    
    sb.set_theme(style='white')
    plt.figure(figsize=figsize)
    if metadata_to_color:
        if metadata_color_dict:
            if vlim == ():
                ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data, palette=metadata_color_dict, s=s, linewidth=linewidth,edgecolor='black')
            else:
                ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data, palette=metadata_color_dict, s=s, hue_norm=vlim, linewidth=linewidth,edgecolor='black')
            sb.move_legend(ax, loc="center",bbox_to_anchor=(1.2, 0.5), frameon=False, title=None)
        else:
            if vlim == ():
                ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data, palette=palette, s=s, linewidth=linewidth,edgecolor='black')
            else:
                ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data, palette=palette, s=s, hue_norm=vlim, linewidth=linewidth,edgecolor='black')
            sb.move_legend(ax, loc="center",bbox_to_anchor=(1.2, 0.5), frameon=False, title=None)
            # sb.move_legend(ax, loc="center",bbox_to_anchor=(.5, 1), frameon=False, title=None, ncols=3)            

    elif data_series_to_color is not None:
        if data_series_to_color.dtype == "O":
            ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", c=data_series_to_color, s=s, linewidth=linewidth,edgecolor='black')
        elif len(vlim) == 0:
            ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data_series_to_color, palette=palette, s=s, linewidth=linewidth,edgecolor='black')
            sb.move_legend(ax, loc="center",bbox_to_anchor=(1.2, 0.5), title=title)
            
        else:
            ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y", hue=data_series_to_color, palette=palette, s=s, hue_norm=vlim, linewidth=linewidth,edgecolor='black')
            sb.move_legend(ax, loc="center",bbox_to_anchor=(1.2, 0.5), title=title)
        
    

    else:
        ax = sb.scatterplot(data=metacells_adata.obs, x="x", y="y")

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    sb.despine(left=True, bottom=True)
    
    if remove_legend:
        ax.get_legend().remove()
        
    if output_path:
        plt.savefig(output_path, dpi=1200, format="pdf", bbox_inches="tight")


class Log2EpsNorm(Normalize):
    def __init__(self, epsilon=0.01, vmin=None, vmax=None, clip=False):
        self.epsilon = epsilon
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # value is your data array
        value = np.asarray(value)
        log_val = np.log2(value + self.epsilon)
        return super().__call__(log_val, clip)
    
    
def plot_fractions_stacked_barplot(df, x_grouped_field, y_count_field, colors=None, x_order=[], y_order=[],
                                    x_label="", y_label="", title="", figsize=(5,5), fontsize=18,
                                    remove_legend=False, output_path=None, show_yticks=True, observed=True):
    df = df.copy()

    df = df.groupby([x_grouped_field, y_count_field], observed=observed).size().unstack(fill_value=0).astype(float)
    df = df.div(df.sum(axis=1), axis=0)
    df = df.reset_index()

    if colors is None:
        colors = sb.color_palette(colors, n_colors=len(df.columns) - 1)

    df.index = df[x_grouped_field]

    if len(x_order) > 0:
        df = df.loc[[i for i in x_order if i in df.index]]

    if len(y_order) > 0:
        df = df[[x_grouped_field] + [i for i in y_order if i in df.columns]]

    if df.shape[0] == 0 or df.shape[1] <= 1:
        print("No data to plot %s - %s." % (title, output_path if output_path else ""))
        return
    
    # Plot horizontally
    df.drop(columns=[x_grouped_field]).plot.barh(
        stacked=True,
        figsize=figsize,
        color=colors,
        edgecolor='none'
    )

    y_label = y_label if y_label else y_count_field  # X becomes Y in horizontal
    x_label = x_label if x_label else x_grouped_field

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0,1)
    
    if not show_yticks:
        plt.yticks([])

    plt.legend(title=y_count_field, fontsize=fontsize, title_fontsize=fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
    if remove_legend:
        plt.legend().remove()

    if output_path is not None:
        plt.savefig(output_path + ".pdf", dpi=1200, format="pdf", bbox_inches="tight")


def get_correlation_between_two_df(a,b):
    correlation = np.corrcoef(a.values, b.values)
    correlation_df = pd.DataFrame(correlation, index=a.index.to_list() + b.index.to_list(), columns=a.index.to_list() + b.index.to_list())
    correlation_df = correlation_df.iloc[:a.shape[0], a.shape[0]:]
    
    return correlation_df


def get_cell_score_per_genes_and_bins(cells_ad, genes_per_bin, modules_to_remove=[], layer="downsampled"):
    cells_score_df = pd.DataFrame(index=cells_ad.obs.index, columns=sorted(genes_per_bin.keys())) 
    for module_to_remove in modules_to_remove:
        cells_score_df.drop(module_to_remove, axis=1, inplace=True)

    for bins_i in genes_per_bin.keys():
        if bins_i not in cells_score_df.columns:
            continue
        
        genes = genes_per_bin[bins_i]
        genes_difference = set(genes).difference(cells_ad.var_names)
        if len(genes_difference) > 0:
            print("Genes %s not in cells adata" %genes_difference)
        
        genes_to_use = list(set(genes).intersection(cells_ad.var_names))
        cells_score_df[bins_i] = mc.ut.get_o_numpy(cells_ad[:,genes_to_use], name=layer, sum=True)

    return cells_score_df



def kmeans_with_size(X, n_clusters, size, min_size=12, random_state=42, n_init=10, max_iter=300, max_passes=10_000, max_size=None, verbose=False):
    """
    K-means with post-processing to split clusters larger than max_size
    and merge clusters smaller than min_size, without infinite loops.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_clusters : int
        Initial number of clusters for the first KMeans fit.
    size : int
        Target/typical size. If max_size is None, uses 2*size.
    min_size : int, default 12
    random_state : int
    n_init : int
        Used for the very first fit and for 2-means splits.
    max_iter : int
    max_passes : int
        Global safety cap on outer adjustments to avoid runaway loops.
    max_size : int or None
        If None, set to 2*size.
    verbose : bool
        Print progress.

    Returns
    -------
    kmeans : sklearn.cluster.KMeans
        Final fitted model.
    """
    if max_size is None:
        max_size = 2 * size
    if min_size <= 0 or max_size <= 0:
        raise ValueError("min_size and max_size must be positive.")
    if min_size >= max_size:
        raise ValueError("min_size must be < max_size.")

    X = np.asarray(X)
    n_samples = X.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError("n_clusters must be in [1, n_samples].")

    def fit_kmeans_with_inits(nc, init_centers):
        """Deterministic fit with provided centers as init."""
        init_centers = np.asarray(init_centers)
        if init_centers.shape != (nc, X.shape[1]):
            raise ValueError(f"init_centers must be shape ({nc}, {X.shape[1]})")
        return KMeans(
            n_clusters=nc,
            init=init_centers,
            n_init=1,                  # important: deterministic reuse of centers
            random_state=random_state,
            max_iter=max_iter
        ).fit(X)

    # 1) Initial vanilla k-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=random_state,
        max_iter=max_iter
    ).fit(X)

    passes = 0

    # 2) Split loop: repeatedly split clusters that are too large
    while True:
        if passes >= max_passes:
            if verbose: print("Safety cap reached during split loop.")
            break
        passes += 1

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_clusters = centers.shape[0]

        sizes = np.bincount(labels, minlength=n_clusters)
        large_clusters = np.where(sizes > max_size)[0]
        if large_clusters.size == 0:
            break  # nothing to split

        if verbose:
            print(f"Splitting pass (n_clusters={n_clusters}): large={list(large_clusters)}")

        new_centers = list(centers)
        progressed = False

        # Sort by size desc so we split the worst offenders first
        for cid in large_clusters[np.argsort(sizes[large_clusters])[::-1]]:
            pts = X[labels == cid]
            if pts.shape[0] < 2:
                # Can't split a singleton; skip (shouldn't happen if > max_size, but be safe)
                continue

            # Local 2-means split for that cluster
            sub_kmeans = KMeans(
                n_clusters=2,
                init='k-means++',
                n_init=n_init,
                random_state=random_state,
                max_iter=max_iter
            ).fit(pts)

            c0, c1 = sub_kmeans.cluster_centers_
            # Replace original center with one subcenter; append the other
            new_centers[cid] = c0
            new_centers.append(c1)
            progressed = True

        if not progressed:
            # Could not split any (degenerate case)
            if verbose: print("No viable splits; stopping split loop.")
            break

        # Refit with proposed centers deterministically
        n_clusters = len(new_centers)
        kmeans = fit_kmeans_with_inits(n_clusters, np.array(new_centers))

    # 3) Merge loop: repeatedly try to remove one small cluster per pass
    passes_merge = 0
    while True:
        if passes >= max_passes:
            if verbose: print("Safety cap reached during merge loop.")
            break
        passes += 1
        passes_merge += 1

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_clusters = centers.shape[0]

        sizes = np.bincount(labels, minlength=n_clusters)
        small_clusters = np.where(sizes < min_size)[0]
        if small_clusters.size == 0:
            break  # nothing to merge

        if verbose:
            print(f"Merge pass {passes_merge} (n_clusters={n_clusters}): small={list(small_clusters)}")

        progressed = False

        # Try to remove exactly one small cluster this pass (simplifies bookkeeping)
        # Try smallest first
        for cid in small_clusters[np.argsort(sizes[small_clusters])]:
            # Propose removing this center (no in-place pops = no index shifts)
            proposal = [c for i, c in enumerate(centers) if i != cid]
            n_prop = len(proposal)

            # Refit with proposed centers; deterministic reuse
            k_prop = fit_kmeans_with_inits(n_prop, np.array(proposal))
            new_sizes = np.bincount(k_prop.labels_, minlength=n_prop)

            # Accept only if removing doesn't create a whale elsewhere
            if np.any(new_sizes > max_size):
                # reject this candidate; try another small cluster
                if verbose:
                    mx = int(new_sizes.max())
                    print(f"Reject removal of {cid}: would create size {mx} (> {max_size}).")
                continue

            # Accept: commit the change, then restart outer merge loop
            kmeans = k_prop
            progressed = True
            if verbose:
                print(f"Accepted removal of {cid}; n_clusters -> {n_prop}.")
            break

        if not progressed:
            if verbose:
                print("No valid merges without violating max_size; stopping merge loop.")
            break

    # Final clean refit (optional; already fitted)
    return kmeans

def get_n_largest_values_in_df(df, n, row_based=True):
    if not row_based:
        df = df.T

    top_k_values = np.sort(df.to_numpy(), axis=1)[:, -n:]
    top_values_df = pd.DataFrame(top_k_values[:, ::-1], index=df.index)
    return top_values_df


def get_n_largest_columns_in_df(df, n ,row_based=True):
    if not row_based:
        df = df.T

    data = df.to_numpy()
    top_k_indices = np.argsort(data, axis=1)[:, -n:]  # Indices of top k values
    top_k_columns = np.array(df.columns)[top_k_indices]
    top_columns_df = pd.DataFrame(top_k_columns[:, ::-1], index=df.index)

    return top_columns_df

def get_gene_module_expression(anndata:ad.AnnData, genes:List[str], layer:str ="__x__", epsilon:float =1e-5, log_base:int=2):
    diff_genes = set(genes) - set(anndata.var_names)
    if len(diff_genes):
        print("Some genes don't appear in the metacells object, ignoring them: %s" %",".join(list(diff_genes)))
        genes = anndata.var_names.intersection(genes)
    
    fp = get_anndata_fp(anndata, layer=layer) 
    fp_genes = fp.loc[:,genes]
    
    if log_base == 1:        
        expression = pd.Series(scipy.stats.gmean(fp_genes + epsilon, axis=1), index=anndata.obs_names)

    else:
        # mean (log(fp + epsilon))
        expression = pd.Series(np.mean(np.log(fp_genes + epsilon) / np.log(log_base), axis=1), index=anndata.obs_names)
        
    return expression




def ggplot(anndata, g1=None, g2=None, m1=None, m2=None, metadata_to_color=None, 
           epsilon=1e-5, mark_subgroup = [],
           title="", add_distribution=False, point_size=30, log_base = 2, 
           vline_value = None, hline_value = None, xlim=None, ylim=None, g1_label="g1 score", g2_label = "g2 score", color_dict=None,
           output_folder:str=None, legend=True, fontsize=18, figsize=6,file_prefix="", label1 = None, label2 = None, xaxis_formatter=None, yaxis_formatter=None):
    if metadata_to_color:
        data = anndata.obs[metadata_to_color]

    elif genes_to_color:
        if isinstance(genes_to_color,str):
            genes_to_color = [genes_to_color]
        data = get_gene_module_expression(anndata,genes_to_color, epsilon=epsilon, log_base=log_base)

    else:
        data=None


    if isinstance(g1,str):
        if g1 not in anndata.var_names:
            print("Gene %s not in the anndata object" %g1)
            return
        
        g1 = [g1.strip()]

    if isinstance(g2,str):
        if g2 not in anndata.var_names:
            print("Gene %s not in the anndata object" %g2)
            return
        
        g2 = [g2.strip()]


    if g1:
        g1_score = get_gene_module_expression(anndata, g1, log_base=log_base)
        g1_label = g1[0] if len(g1) == 1 else g1_label

    elif m1:
        g1_score = anndata.obs[m1]
        g1_label = m1
    
    if g2:
        g2_score = get_gene_module_expression(anndata, g2, log_base=log_base)
        g2_label = g2[0] if len(g2) == 1 else g2_label

    elif m2:
        g2_score = anndata.obs[m2]
        g2_label = m2
    
    
    plot_func = sb.scatterplot if not add_distribution else sb.jointplot

    plt.figure(figsize=(figsize,figsize))
    style = "seaborn-poster"
    if style not in plt.style.available:
        style = "seaborn-v0_8-poster"

    if style not in plt.style.available:
        style = "ggplot"

    

    with plt.style.context(style):
        if data is not None:
            if metadata_to_color:
                if color_dict:
                    s = plot_func(x=g1_score, y=g2_score, hue=data, palette=color_dict, s=point_size, legend=legend, edgecolor="black")
                else:
                    s = plot_func(x=g1_score, y=g2_score, hue=data, s=point_size,legend=legend, edgecolor="black")
                
            else:
                s = plot_func(x=g1_score, y=g2_score, hue=data, palette="Reds", s=point_size,legend=legend, edgecolor="black")

            if legend:
                sb.move_legend(s, loc=(1.01,0.2))
        else:
            s = plot_func(x=g1_score, y=g2_score, s=point_size)
            if len(mark_subgroup):
                plot_func(x=g1_score[mark_subgroup], y=g2_score[mark_subgroup], s=point_size,legend=legend, edgecolor="black")

        if vline_value:
            plt.axvline(x=vline_value, color='crimson', linestyle='--')

        if hline_value:
            plt.axhline(y=hline_value, color='crimson', linestyle='--')
            
        if label1:
            plt.xlabel(label1, fontsize=fontsize)
        else:
            plt.xlabel(g1_label, fontsize=fontsize)

        if label2:
            plt.ylabel(label2, fontsize=fontsize)
        else:
            plt.ylabel(g2_label, fontsize=fontsize)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.grid()
        plt.title(title)
        
        if xaxis_formatter:
            s.xaxis.set_major_formatter(mticker.FuncFormatter(xaxis_formatter))
        if yaxis_formatter:
            s.yaxis.set_major_formatter(mticker.FuncFormatter(yaxis_formatter))
            
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
    
        if output_folder:
            file_name = title if title else ""
            
            if label1:
                    g1_label = label1.replace(" ","_")
                
            if label2:
                g2_label = label2.replace(" ","_")
                
            
            file_name += "_%s_%s" %(g1_label, g2_label)
            file_name = "ggplot_" + file_name
            
            if file_prefix !="":   
                file_name = file_prefix + "_" + file_name
            
            
            plt.savefig(os.path.join(output_folder, file_name + ".pdf"), dpi=1200, format="pdf", bbox_inches="tight")
            
        plt.show()
        
def calc_diff_expression_of_two_metacells_groups(cells_anndata1:ad.AnnData, cells_anndata2:ad.AnnData, obs1:pd.Index, obs2:pd.Index, metacells_umis_df1:pd.DataFrame, metacells_umis_df2:pd.DataFrame, genes_to_test=[],diff_thresh:float = 1, pval_thresh:float = 0.05, epsilon=1e-5, calculate_p_val=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    calculate rlgc between two groups 

    take only those above diff_thresh  as optional genes
    calc how much umis per metacells 
    for each genes have a 2x2
        gene_in_group1 | gene in group 2
        total group 1  | tot group 2
        
    chiseq test 

    take only those above specific threshold 
    """
    group1_lgc = calc_group_expression_metacells(cells_anndata1, obs1, epsilon=epsilon, metacells_umis_df=metacells_umis_df1)
    group2_lgc = calc_group_expression_metacells(cells_anndata2, obs2, epsilon=epsilon, metacells_umis_df=metacells_umis_df2)

    if len(genes_to_test) == 0:
        genes_to_test = group1_lgc.index.intersection(group2_lgc.index)

    total_number_of_umis1 = metacells_umis_df1.sum(axis=1)
    total_number_of_umis2 = metacells_umis_df2.sum(axis=1)
    total_group1 = total_number_of_umis1.loc[obs1].sum()
    total_group2 = total_number_of_umis2.loc[obs2].sum()

    rlgc = pd.Series((group1_lgc.loc[genes_to_test].values - group2_lgc.loc[genes_to_test].values).reshape(-1), index=genes_to_test)
    
    # high diff between them
    genes_with_high_diff = rlgc.index[abs(rlgc) >= diff_thresh]

    # for each gene calculate the chi square test
    chi_square_results = pd.Series(0, index=genes_to_test)
    if calculate_p_val:
        mat1 = metacells_umis_df1.loc[obs1]
        mat2 = metacells_umis_df2.loc[obs2]
        genes_1_sum = mat1.sum(axis=0).loc[genes_to_test]
        genes_2_sum = mat2.sum(axis=0).loc[genes_to_test]
        for gene_i in range(len(genes_to_test)):
            g1_sum = genes_1_sum.iloc[gene_i]
            g2_sum = genes_2_sum.iloc[gene_i]

            if g1_sum == 0 and g2_sum.sum() == 0:
                chi_square_results.iloc[gene_i] = None
                continue

            obs = np.array([[g1_sum, g2_sum], [total_group1 - g1_sum, total_group2 - g2_sum]])
            _, p, _, _ = scipy.stats.chi2_contingency(obs)
            chi_square_results.iloc[gene_i] = p 

    
    # combine the chi_square_results series with the rlgc series
    diff_expression_genes = pd.concat([rlgc, chi_square_results.loc[rlgc.index], group1_lgc.loc[rlgc.index], group2_lgc.loc[rlgc.index]], axis=1)
    diff_expression_genes.columns = ["rlgc", "pval", "exp_obs1", "exp_obs2"]
    diff_expression_genes = diff_expression_genes.sort_values("rlgc")
    
    # now return the genes which are above the threshold and have a pval below the threshold, and also return the entire diff_expression_genes
    return diff_expression_genes[diff_expression_genes.index.isin(genes_with_high_diff) & (diff_expression_genes.pval < pval_thresh)], diff_expression_genes



def calc_group_expression_metacells(cells_anndata:ad.AnnData, obs1:pd.Index, epsilon:float = 1e-5, log_base=2, metacells_umis_df= None) -> pd.DataFrame:
    if metacells_umis_df is None:
        metacells_umis_df = get_metacells_umis_df(cells_anndata)

    group1_umis = metacells_umis_df.loc[obs1].sum()     
    group1_fp = group1_umis / group1_umis.sum() 
    if log_base ==1 :
        return group1_fp
    else:
        return np.log(group1_fp + epsilon) / np.log(log_base)



def get_marker_genes(addata:ad.AnnData, minimal_delta = 5, percentile=5, ignore_lateral=True, ignore_noisy=True, epsilon=1e-5, log_base=2) ->pd.Series:
    lgc = get_lgc(addata, epsilon=epsilon, log_base=log_base)
    lgc_delta = pd.Series(np.percentile(lgc, 100 - percentile,axis=0) - np.percentile(lgc, percentile,axis=0), index=addata.var_names)
    
    if ignore_lateral and "lateral_gene" in addata.var.columns:
        lgc_delta = lgc_delta[~addata.var.lateral_gene]
    if ignore_noisy and "noisy_gene" in addata.var.columns:
        lgc_delta = lgc_delta[~addata.var.noisy_gene]
    
    markers_series = lgc_delta[lgc_delta >= minimal_delta]
    return markers_series

def get_correaltion_between_metacells_atlases(q_addata:ad.AnnData, a_anndata:ad.AnnData, marker_genes=[], epsilon=1e-5, log_base=2):
    q_lgc = get_lgc(q_addata, epsilon=epsilon, log_base=log_base)
    a_lgc = get_lgc(a_anndata, epsilon=epsilon, log_base=log_base)
    correlation_df = get_correlation_between_two_df(q_lgc.loc[:,marker_genes], a_lgc.loc[:,marker_genes])
    return correlation_df



def add_exp_and_sample_info_to_cell_barcode(cell_barcode: np.ndarray, sample_name:Union[int, str, pd.Series], exp_id:str):
    if isinstance(sample_name, pd.Series):
        return (cell_barcode + "_" + exp_id + "_" + sample_name.astype(str)).values
    
    return f"{cell_barcode}_{exp_id}_{sample_name}_" 



def add_metadata_fractions_to_metacells_ad(cells_adata: ad.AnnData, metacell_adata:ad.AnnData, metadata_fields_to_add:list):
    for metadata_field in metadata_fields_to_add:
        property_of_obs = mc.ut.get_o_numpy(cells_adata, metadata_field).copy().astype(str)
        if np.any([isinstance(i, type(np.nan)) for i in property_of_obs]):
            property_of_obs[[isinstance(i, type(np.nan)) for i in property_of_obs]] = "unknown"

        
        unique_values = sorted(np.unique(property_of_obs))
        for value in unique_values:
            mc.tl.convey_obs_to_group(
                adata=cells_adata,
                gdata=metacell_adata,
                group="metacell",
                property_name=metadata_field,
                to_property_name=f"{metadata_field}_{value}",
                method=mc.ut.fraction_of_grouped(value),
            )


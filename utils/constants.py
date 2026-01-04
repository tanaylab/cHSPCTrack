import os

f_dir = os.path.dirname(os.path.abspath("__file__"))  
BASE_PATH = os.path.abspath(os.path.join(f_dir, "..", ".."))

CELLS_PATH = os.path.join(BASE_PATH, "data/cells_with_clone_info.h5ad")
METACELLS_PATH = os.path.join(BASE_PATH, "data/metacells.h5ad")
CELLS_TYPE_COLORS = os.path.join(BASE_PATH, "data/cell_type_colors.csv")
GENE_MODULES_DICT_PATH = os.path.join(BASE_PATH, "data/gene_modules_dict.pkl")

# TRAJECTORY_KMEAN_MAPPER
TRAJECTORY_PC_MAPPER = os.path.join(BASE_PATH, "data/trajectory_pc_mapper.pkl")

TRAJECTORIES_TO_TOP_LEVEL_CELL_TYPE_DICT = {
    "Ery":["Ery","MEP","MEBEMP"],
    "Basophils":["Basophils","BEP","BEMP", "MEBEMP"],
    "Mast":["BEMP", "Mast", "MEBEMP"],
    "Eosinophil":["Eosinophil","BEMP", "BEP", "MEBEMP"],
    "MK":["MK","MEP", "MEBEMP"],
    "Neutrophils":["Neutrophils","GMP",],
    "Monocyte": ["Monocyte","MonoP","GMP"],
}


# Those are cells after CD34+ sorting with lentivirus but before any culture
CD34_POSITIVE_CELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/pre_media_cd34_positive_cells.h5ad")

HUMAN_BM_CELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/human_bm_cells.h5ad")
HUMAN_BM_METACELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/human_bm_metacells.h5ad")
HUMAN_CD34_POSITIVE_METACELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/human_cd34_positive_bm_metacells.h5ad")  
CD34_POSITIVE_PB_METACELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/cd34_positive_pb_metacells.h5ad")
YL_FL_METACELLS_PATH = os.path.join(BASE_PATH, "data/other_datasets/yl_fl_metacells.h5ad")


IN_VIVO_ATLASES_FOR_COMPARISON = {
    "Human BM": HUMAN_BM_METACELLS_PATH,
    "Human CD34+ BM": HUMAN_CD34_POSITIVE_METACELLS_PATH,
    "CD34+ PB": CD34_POSITIVE_PB_METACELLS_PATH,
    "YS + FL": YL_FL_METACELLS_PATH,
}


# Clone calling constants
MIN_NUMBER_OF_READS_FOR_COMMON_TAG = 2**3  # common tags can get other tags for hamming distance to increase the number of reads

MIN_READS_FOR_VALID_TAG = 2**4  # tags with less than this number of reads are not considered valid and are ignored

MIN_NUMBER_OF_TAGS_IN_WELL = MIN_NUMBER_OF_TAGS_IN_CLONE = 2 # wells with less than this number of tags are not considered valid and are ignored

MIN_TAGS_IN_CELL = 2

MIN_READS_FOR_VALID_TAG_UMIS = 2

MIN_NUMBER_OF_UMIS_FOR_VALID_TAG = 2

MIN_SIZE_OF_VALID_CLONE = 2

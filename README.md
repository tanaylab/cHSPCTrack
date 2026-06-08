cHSPCTrack - clonal Hematopoietic Stem and Progenitor cells tracker
=================================
This repository contains the code used to produce the results shown in our joint Shlush & Tanay lab manuscript "Human HSPCs clones balance stochastic diversification with cytokine-induced differentiation" (Elia Colin,\*, Dror Brook,\*, Jonathan Izraeli, Oren Milman, Aviezer Lifshitz,  Tal Bachrach, Noa Chapal-Ilani, Liran Shlush,\*\*, Amos Tanay,\*\*).

Data availability
======================
All data required to reproduce the figures are available at:

```
s3://chspctrack/cHSPCTrack/
```

After cloning the code repository, the data should be downloaded into the data/ directory at the root of the repository using:

```bash
mkdir -p data
aws s3 sync s3://chspctrack/cHSPCTrack ./data --no-sign-request
```

Expected structure:
```
data/
  clonal_memory_data/
  differentiation_rate_data/
  other_datasets/
  tags_data/
  cell_type_colors.csv
  cells_with_clone_info.h5ad
  gene_modules_dict.pkl
  metacells.h5ad
  trajectory_pc_mapper.pkl
```

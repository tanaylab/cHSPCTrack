import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sb

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
princurve = importr("princurve", on_conflict="warn")
from scipy.special import logit



class PrincipleCurve:
    def __init__(self, n_components=2, reverse_order=False, use_fractions=True):
        """
        Args:
            n_components (int): PCA dimensions to reduce data.
            reverse_order (bool): Whether to flip pseudotime order (1 - pseudotime).
            use_fractions (bool): If True, input data is transformed by safe_logit before fitting or predicting.
        """
        self.n_components = n_components
        self.reverse_order = reverse_order
        self.use_fractions = use_fractions
        
        self.pca = PCA(n_components=n_components)
        self.curve_s_sorted = None
        self.arc_lengths = None
        self.tree = None
        self.fitted = False

    def fit(self, data, show_plots=False, hue=None, palette=None):
        """
        Fit principal curve to data.
        
        Args:
            data (pd.DataFrame): Input data matrix (cells x genes/features).
            show_plots (bool): If True, plot PCA scatter and principal curve.

        Returns:
            pd.Series: pseudotime indexed by data.index, sorted by pseudotime.
        """
        # Optional preprocessing
        X = safe_logit(data) if self.use_fractions else data.copy()

        # PCA reduction
        X_pca = self.pca.fit_transform(X)

        # princurve fit
        fit_result = princurve.principal_curve(X_pca)
        curve_s = np.array(fit_result.rx2("s"))
        data_order = np.array(fit_result.rx2("ord")) - 1
        self.curve_s_sorted = curve_s[data_order]

        diffs = np.diff(self.curve_s_sorted, axis=0)
        self.arc_lengths = np.insert(np.cumsum(np.linalg.norm(diffs, axis=1)), 0, 0)
        self.tree = cKDTree(self.curve_s_sorted)

        _, nearest_idx = self.tree.query(X_pca, k=1)
        pseudotime = self.arc_lengths[nearest_idx]
        pseudotime_norm = (pseudotime - self.arc_lengths.min()) / (self.arc_lengths.max() - self.arc_lengths.min())
        if self.reverse_order:
            pseudotime_norm = 1 - pseudotime_norm
        self.fitted = True

        pseudotime_series = pd.Series(pseudotime_norm, index=data.index).sort_values()

        if show_plots:
            plt.figure(figsize=(10, 6))
            if hue is not None and palette is not None:
                sb.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hue.loc[data.index], palette=palette, s=30, legend='full')
            else:
                sb.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=30, legend=False)
            plt.plot(self.curve_s_sorted[:, 0], self.curve_s_sorted[:, 1], color='black', linewidth=2, label='Principal Curve')
            plt.title("PCA + Principal Curve")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            sb.scatterplot(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                hue=pseudotime_series.loc[data.index].values,
                palette="viridis",
                s=20
            )
            plt.title("Cells colored by principal curve pseudotime")
            plt.tight_layout()
            plt.show()

        return pseudotime_series

    def predict(self, new_data, show_plots=False):
        """
        Project new data onto existing principal curve and compute pseudotime.
        
        Args:
            new_data (pd.DataFrame): New cells to project.
            show_plots (bool): If True, plot PCA scatter colored by pseudotime.

        Returns:
            pd.Series: pseudotime indexed by new_data.index.
        """
        if not self.fitted:
            raise RuntimeError("Must fit the principal curve before predicting.")

        X = safe_logit(new_data) if self.use_fractions else new_data.copy()
        X_pca = self.pca.transform(X)

        _, nearest_idx = self.tree.query(X_pca, k=1)
        pseudotime = self.arc_lengths[nearest_idx]
        pseudotime_norm = (pseudotime - self.arc_lengths.min()) / (self.arc_lengths.max() - self.arc_lengths.min())

        if self.reverse_order:
            pseudotime_norm = 1 - pseudotime_norm

        pseudotime_series = pd.Series(pseudotime_norm, index=new_data.index)

        if show_plots:
            plt.figure(figsize=(10, 6))
            sb.scatterplot(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                hue=pseudotime_series,
                palette="viridis",
                s=20
            )
            plt.title("Projection of new data on principal curve")
            plt.tight_layout()
            plt.show()

        return pseudotime_series.sort_values()

def safe_logit(df):
    return df.clip(1e-6, 1 - 1e-6).apply(logit)

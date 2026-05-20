import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import stats
from scipy.stats import pearsonr, truncnorm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def dis_to_label(dis):
    """Convert diagnosis labels to integer class labels.

    Parameters
    ----------
    dis : array-like
        Array of diagnosis labels containing `"CN"`, `"MCI"`, and `"AD"`.

    Returns
    -------
    list[int]
        Integer labels where CN=0, MCI=1, and AD=2.
    """
    label = []

    for i in range(dis.shape[0]):
        if dis[i] == "CN":
            label.append(0)
        if dis[i] == "MCI":
            label.append(1)
        if dis[i] == "AD":
            label.append(2)

    return label


def standard_scaler(x):
    """Standardize an array using population mean and standard deviation.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    np.ndarray
        Standardized array.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std(ddof=0)

    return (x - mean) / std


def TTsplit(brain_data, rs=34):
    """Create stratified group train-test folds.

    Parameters
    ----------
    brain_data : np.ndarray
        Data matrix whose first column contains subject IDs and second column
        contains diagnosis labels.
    rs : int, default=34
        Random seed for fold generation.

    Returns
    -------
    list[list[np.ndarray]]
        List of `[train_index, test_index]` pairs.
    """
    folds = []
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=rs, shuffle=True)

    for _, (train_index, test_index) in enumerate(
        sgkf.split(brain_data[:, 0], dis_to_label(brain_data[:, 1]), brain_data[:, 0])
    ):
        folds.append([train_index, test_index])

    return folds


def boostrap_dataloader(X, Y, Z, M, seed=None):
    """Generate a bootstrap resample of the given arrays.

    Parameters
    ----------
    X : np.ndarray
        Exposure matrix.
    Y : np.ndarray
        Outcome matrix.
    Z : np.ndarray
        Covariate matrix.
    M : np.ndarray
        Mediator matrix.
    seed : int, optional
        Random seed for bootstrap sampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Bootstrap-resampled `(X, Y, Z, M)`.
    """
    if seed is not None:
        np.random.seed(seed)

    N = X.shape[0]
    t = np.random.choice(N, size=N, replace=True)

    X_resampled = X[t]
    Y_resampled = Y[t]
    Z_resampled = Z[t]
    M_resampled = M[t]

    return X_resampled, Y_resampled, Z_resampled, M_resampled


def set_seed(seed: int = 34):
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default=34
        Random seed used by Python, NumPy, and environment-level hash seeding.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)


set_seed(34)


def get_columns_after(df, col_name, k):
    """Extract `k` columns starting from a given column name.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_name : str
        Starting column name.
    k : int
        Number of columns to extract.

    Returns
    -------
    np.ndarray
        Array containing the selected columns as rows.

    Raises
    ------
    ValueError
        If `col_name` is not found in the dataframe.
    """
    cols = df.columns.tolist()

    if col_name not in cols:
        raise ValueError(f"Column '{col_name}' not found.")

    start_idx = cols.index(col_name)
    end_idx = start_idx + k
    name = cols[start_idx:end_idx]

    return np.array([df[i].to_list() for i in name])


def get_data():
    """Load and preprocess covariate, brain, gene, and cognitive data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple `(Cov, Brain, Gene, Cog)` containing covariates, brain features,
        genetic features, and cognitive outcomes.
    """
    df = pd.read_csv("Data\Datal.csv")

    Gender = np.array([0 if g == "Female" else 1 for g in df["PTGENDER"].tolist()]).reshape(-1, 1)
    Educat = np.array(df["PTEDUCAT"].tolist()).reshape(-1, 1)

    temp = []
    for g in df["PTETHCAT"].tolist():
        if g == "Hisp/Latino":
            temp.append(0)
        elif g == "Not Hisp/Latino":
            temp.append(1)
        else:
            temp.append(2)
    Ethcat = np.array(temp).reshape(-1, 1)

    temp = []
    for g in df["PTRACCAT"].tolist():
        if g == "Asian":
            temp.append(0)
        elif g == "Black":
            temp.append(1)
        elif g == "More than one":
            temp.append(2)
        elif g == "Unknown":
            temp.append(3)
        elif g == "White":
            temp.append(4)
        else:
            temp.append(5)
    Ethrac = np.array(temp).reshape(-1, 1)

    temp = []
    for g in df["PTMARRY"].tolist():
        if g == "Divorced":
            temp.append(0)
        elif g == "Married":
            temp.append(1)
        elif g == "Never married":
            temp.append(2)
        elif g == "Unknown":
            temp.append(3)
        else:
            temp.append(4)
    Ethmar = np.array(temp).reshape(-1, 1)

    Age = np.array(df["AGE"].tolist()).reshape(-1, 1)
    Brain = get_columns_after(df, "s2018_LH_Background+FreeSurfer_Defined_Medial_Wall", 202).T
    Gene = get_columns_after(df, "ge_rs4420638", 688).T

    name = [
        "SubjectID",
        "DX",
        "ADAS11",
        "ADAS13",
        "ADASQ4",
        "CDRSB",
        "MMSE",
        "RAVLT_immediate",
        "RAVLT_perc_forgetting",
        "RAVLT_learning",
        "FAQ",
        "TRABSCOR",
    ]

    df_ = df
    for c in name[2:]:
        df_[c] = pd.to_numeric(df[c], errors="coerce")

    Cog = np.array([df_[i].to_list() for i in name], dtype=object).T
    Cov = np.concatenate([Age, Gender, Educat, Ethcat, Ethrac, Ethmar], axis=1)

    return Cov, Brain, Gene, Cog


def make_cmp(R, G, B, N=256):
    """Create a listed colormap from RGB channel ranges.

    Parameters
    ----------
    R : array-like
        Two-element red-channel range.
    G : array-like
        Two-element green-channel range.
    B : array-like
        Two-element blue-channel range.
    N : int, default=256
        Number of colors.

    Returns
    -------
    ListedColormap
        Generated colormap.
    """
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(R[0] / 256, R[1] / 256, N)
    vals[:, 1] = np.linspace(G[0] / 256, G[1] / 256, N)
    vals[:, 2] = np.linspace(B[0] / 256, B[1] / 256, N)

    return ListedColormap(vals)


def make_grad_cmap(R, G, B, N=256, name="custom_grad"):
    """Create a smooth gradient colormap between two custom colors.

    Parameters
    ----------
    R : array-like
        Two-element red-channel range with values in `[0, 255]`.
    G : array-like
        Two-element green-channel range with values in `[0, 255]`.
    B : array-like
        Two-element blue-channel range with values in `[0, 255]`.
    N : int, default=256
        Number of colors.
    name : str, default="custom_grad"
        Name of the colormap.

    Returns
    -------
    LinearSegmentedColormap
        Generated gradient colormap.
    """
    R = np.asarray(R) / 255.0
    G = np.asarray(G) / 255.0
    B = np.asarray(B) / 255.0

    start = (R[0], G[0], B[0])
    end = (R[1], G[1], B[1])

    return LinearSegmentedColormap.from_list(name, [start, end], N)


def make_double_cmp(R1, G1, B1, R2, G2, B2, N=256):
    """Create a two-sided listed colormap from two RGB gradients.

    Parameters
    ----------
    R1, G1, B1 : array-like
        RGB channel ranges for the first half of the colormap.
    R2, G2, B2 : array-like
        RGB channel ranges for the second half of the colormap.
    N : int, default=256
        Number of colors.

    Returns
    -------
    ListedColormap
        Combined colormap.
    """
    top = make_cmp(R1, G1, B1, N=N)
    bottom = make_cmp(R2, G2, B2, N=N)

    new_colors = np.vstack(
        [
            top(np.linspace(0, 1, int(N / 2))),
            bottom(np.linspace(1, 0, int(N / 2))),
        ]
    )

    return ListedColormap(new_colors)


def gen_data(
    s=2,
    q=20,
    t=10,
    p=20,
    e_noise=0.05,
    x_noise=0.05,
    seed=34,
    N=10000,
    iid=False,
):
    """Generate synthetic mediation data.

    Parameters
    ----------
    s : int, default=2
        Number of covariates.
    q : int, default=20
        Number of exposures.
    t : int, default=10
        Number of outcomes.
    p : int, default=20
        Number of mediators.
    e_noise : float, default=0.05
        Noise scale for mediator generation.
    x_noise : float, default=0.05
        Noise scale for outcome generation.
    seed : int, default=34
        Random seed.
    N : int, default=10000
        Number of samples.
    iid : bool, default=False
        If `True`, use independent Gaussian noise instead of correlated noise.

    Returns
    -------
    tuple[np.ndarray, ...]
        Generated coefficient matrices and synthetic data:
        `(alpha_gen, beta_gen, gamma_gen, zeta_gen, eta_gen,
        X_gen, Y_gen, M_gen, Z_gen)`.
    """
    np.random.seed(seed)

    X = np.zeros((1, q))
    Y = np.zeros((1, t))
    Z = np.zeros((1, s))
    M = np.zeros((1, p))

    idx = np.random.choice(M.shape[1] - 4, size=X.shape[1], replace=True)
    alpha_gen = np.zeros((X.shape[1], M.shape[1]))

    for i in range(X.shape[1]):
        alpha_gen[i][idx[i]] = 6 * (1 + np.random.rand())
        alpha_gen[i][idx[i] + 1] = 6 * (1 + np.random.rand())
        alpha_gen[i][idx[i] + 2] = 6 * (1 + np.random.rand())
        alpha_gen[i][idx[i] + 3] = 6 * (1 + np.random.rand())

    idx = np.random.choice(M.shape[1] - 4, size=Z.shape[1], replace=True)
    zeta_gen = np.zeros((Z.shape[1], M.shape[1]))

    for i in range(Z.shape[1]):
        zeta_gen[i][idx[i]] = 6 * (1 + np.random.rand())
        zeta_gen[i][idx[i] + 1] = 6 * (1 + np.random.rand())
        zeta_gen[i][idx[i] + 2] = 6 * (1 + np.random.rand())
        zeta_gen[i][idx[i] + 3] = 6 * (1 + np.random.rand())

    idx = np.random.choice(M.shape[1] - 4, size=Y.shape[1], replace=True)
    beta_gen = np.zeros((Y.shape[1], M.shape[1]))

    for i in range(Y.shape[1]):
        beta_gen[i][idx[i]] = 6 * (1 + np.random.rand())
        beta_gen[i][idx[i] + 1] = 6 * (1 + np.random.rand())
        beta_gen[i][idx[i] + 2] = 6 * (1 + np.random.rand())
        beta_gen[i][idx[i] + 3] = 6 * (1 + np.random.rand())

    beta_gen = beta_gen.T

    gamma_gen = np.zeros([q, t])
    gamma_gen[:5, :2] = 6 * (1 + np.random.rand(5, 2))
    gamma_gen[5:11, 2:4] = 6 * (1 + np.random.rand(6, 2))
    gamma_gen[0, 4:7] = 6 * (1 + np.random.rand(1, 3))
    gamma_gen[2, 4:7] = 6 * (1 + np.random.rand(1, 3))
    gamma_gen[4, 4:7] = 6 * (1 + np.random.rand(1, 3))
    gamma_gen[11:14, 7:] = np.diag(6 * (np.random.rand(3) + 1))

    fil = np.random.binomial(
        n=1,
        p=0.2,
        size=Z.shape[1] * Y.shape[1],
    ).reshape(Z.shape[1], Y.shape[1])

    eta_gen = (np.random.randn(Z.shape[1], Y.shape[1]).round() / 2 + 1) * fil

    alpha_gen += np.random.rand(alpha_gen.shape[0], alpha_gen.shape[1]) * 0.1
    zeta_gen += np.random.rand(zeta_gen.shape[0], zeta_gen.shape[1]) * 0.1
    beta_gen += np.random.rand(beta_gen.shape[0], beta_gen.shape[1]) * 0.1
    gamma_gen += np.random.rand(gamma_gen.shape[0], gamma_gen.shape[1]) * 0.1
    eta_gen += np.random.rand(eta_gen.shape[0], eta_gen.shape[1]) * 0.1

    X = X.T
    Y = Y.T
    Z = Z.T
    M = M.T

    X_gen = X
    Z_gen = Z

    p = M.shape[0]
    modules = [range(0, 5), range(5, 10), range(10, 15), range(15, 20)]
    rhos = [0.6, 0.5, 0.4, 0.3]
    sigma_e = 1.0

    R = np.eye(p)
    for G, rho in zip(modules, rhos):
        G = list(G)
        for i in G:
            for j in G:
                if i != j:
                    R[i, j] = rho

    Eps_covar = sigma_e**2 * R

    Xi_covar = np.array(
        [
            [1.00, 0.75, 0.80, 0.60, -0.55, 0.65, -0.45, -0.50, 0.35, 0.30],
            [0.75, 1.00, 0.90, 0.70, -0.60, 0.70, -0.55, -0.55, 0.40, 0.35],
            [0.80, 0.90, 1.00, 0.75, -0.65, 0.75, -0.60, -0.60, 0.45, 0.40],
            [0.60, 0.70, 0.75, 1.00, -0.50, 0.55, -0.55, -0.50, 0.55, 0.50],
            [-0.55, -0.60, -0.65, -0.50, 1.00, -0.50, 0.60, 0.65, -0.40, -0.35],
            [0.65, 0.70, 0.75, 0.55, -0.50, 1.00, -0.45, -0.45, 0.50, 0.45],
            [-0.45, -0.55, -0.60, -0.55, 0.60, -0.45, 1.00, 0.85, -0.55, -0.60],
            [-0.50, -0.55, -0.60, -0.50, 0.65, -0.45, 0.85, 1.00, -0.60, -0.65],
            [0.35, 0.40, 0.45, 0.55, -0.40, 0.50, -0.55, -0.60, 1.00, 0.85],
            [0.30, 0.35, 0.40, 0.50, -0.35, 0.45, -0.60, -0.65, 0.85, 1.00],
        ]
    )

    N_sample = N

    X_gen = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]),
        cov=np.eye(X.shape[0]),
        size=N_sample,
    ).T

    mean = (91.4 + 55) / 2
    std = 6.972037673038702
    a = (55 - mean) / std
    b = (91.4 - mean) / std

    # The number 55, 91.4, and 6.972037673038702 is taken based on the age distribution in "Datal"

    Age = truncnorm.rvs(a, b, loc=mean, scale=std, size=N_sample).reshape(1, -1)
    Gender = np.random.binomial(n=1, p=0.5, size=N_sample).reshape(1, -1)
    Z_gen = np.concatenate([Age, Gender], axis=0)

    eps = e_noise * np.random.multivariate_normal(
        mean=np.zeros(M.shape[0]),
        cov=Eps_covar,
        size=N_sample,
    )

    xi = x_noise * np.random.multivariate_normal(
        mean=np.zeros(Y.shape[0]),
        cov=Xi_covar,
        size=N_sample,
    )

    if iid:
        eps = e_noise * np.random.randn(N_sample, M.shape[0])
        xi = x_noise * np.random.randn(N_sample, Y.shape[0])

    M_gen = X_gen.T @ alpha_gen + Z_gen.T @ zeta_gen + eps
    Y_gen = M_gen @ beta_gen + X_gen.T @ gamma_gen + Z_gen.T @ eta_gen + xi

    return alpha_gen, beta_gen, gamma_gen, zeta_gen, eta_gen, X_gen, Y_gen, M_gen, Z_gen


def standardize_to_unit(M):
    """Scale a matrix to the range `[-1, 1]` by maximum absolute value.

    Parameters
    ----------
    M : array-like
        Input matrix.

    Returns
    -------
    np.ndarray
        Matrix scaled by its maximum absolute value.
    """
    M = np.asarray(M)
    max_abs = np.max(np.abs(M))

    if max_abs == 0:
        return np.zeros_like(M)

    return M / max_abs


def wald_95_coverage(true, pred_runs, perc=95):
    """Compute Wald confidence-interval coverage.

    Parameters
    ----------
    true : array-like
        True matrix.
    pred_runs : array-like
        Bootstrap or repeated estimated matrices.
    perc : int, default=95
        Nominal confidence level.

    Returns
    -------
    float
        Coverage rate.
    """
    true = np.asarray(true, float)
    pred_runs = np.asarray(pred_runs, float)

    K = pred_runs.shape[0]

    true_flat = true.ravel()
    pred_flat = pred_runs.reshape(K, -1)

    mask = true_flat > 25
    if mask.sum() == 0:
        return np.nan

    true_sub = true_flat[mask]
    pred_sub = pred_flat

    mean_sub = pred_sub.mean(axis=0)
    sd_sub = pred_sub.std(axis=0, ddof=1)
    se_sub = sd_sub / np.sqrt(K)

    z = 1.96
    lower = mean_sub - z * se_sub
    upper = mean_sub + z * se_sub

    covered = (true_sub >= lower) & (true_sub <= upper)
    coverage = covered.mean()

    return coverage


def ci_coverage(alpha_true, alpha_est_list, ci=95, c=25):
    """Compute percentile confidence-interval coverage over selected entries.

    Parameters
    ----------
    alpha_true : array-like
        True coefficient matrix.
    alpha_est_list : array-like
        Estimated coefficient matrices with shape `(K, p, q)`.
    ci : int, default=95
        Nominal confidence level.
    c : float, default=25
        Threshold for selecting entries of `alpha_true`.

    Returns
    -------
    float
        Coverage rate over entries satisfying `alpha_true > c`.
    """
    alpha_true = np.asarray(alpha_true)
    alpha_est = np.asarray(alpha_est_list)

    alpha_true_flat = alpha_true.ravel()

    K = alpha_est.shape[0]
    alpha_est_flat = alpha_est.reshape(K, -1)

    mask = alpha_true_flat > c

    if mask.sum() == 0:
        return np.nan

    true_sub = alpha_true_flat[mask]
    est_sub = alpha_est_flat[:, mask]

    lower_p = (100 - ci) / 2
    upper_p = 100 - lower_p

    ci_lower = np.percentile(est_sub, lower_p, axis=0)
    ci_upper = np.percentile(est_sub, upper_p, axis=0)

    covered = (true_sub >= ci_lower) & (true_sub <= ci_upper)
    coverage_rate = covered.mean()

    return coverage_rate


def type1_error_rate(true_mat, pred_mat, alpha=0.05):
    """Compute the type-I error rate from confidence intervals.

    Parameters
    ----------
    true_mat : np.ndarray
        Matrix of true values.
    pred_mat : np.ndarray
        Matrix containing confidence intervals, mean-standard deviation pairs,
        or standard errors.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    float
        Proportion of entries whose true value lies outside the confidence
        interval.
    """
    if pred_mat.ndim == 3 and pred_mat.shape[2] == 2:
        lower_bounds = pred_mat[..., 0]
        upper_bounds = pred_mat[..., 1]
    elif pred_mat.ndim == 3:
        mean = pred_mat[..., 0]
        std = pred_mat[..., 1] if pred_mat.shape[2] > 1 else np.ones_like(mean)
        z_score = stats.norm.ppf(1 - alpha / 2)
        lower_bounds = mean - z_score * std
        upper_bounds = mean + z_score * std
    elif pred_mat.ndim == 2:
        z_score = stats.norm.ppf(1 - alpha / 2)
        lower_bounds = true_mat - z_score * pred_mat
        upper_bounds = true_mat + z_score * pred_mat
    else:
        raise ValueError(
            f"Unexpected pred_mat shape: {pred_mat.shape}. "
            "Expected (n, m, 2) or (n, m, k) with k>=2."
        )

    outside_ci = (true_mat < lower_bounds) | (true_mat > upper_bounds)
    type1_error = np.mean(outside_ci)

    return type1_error


def df_sort(df):
    """Normalize each dataframe row and keep its largest-magnitude entries.

    For each row, the values are scaled by their maximum absolute value. Only
    the smallest 10 and largest 10 scaled entries are retained; all others are
    set to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Row-normalized and sparsified dataframe.
    """
    df = df.copy()
    df_norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for idx in df.index:
        row = df.loc[idx].values.astype(float)

        max_abs = np.max(np.abs(row))
        if max_abs == 0:
            scaled = np.zeros_like(row)
        else:
            scaled = row / max_abs

        n = len(scaled)
        if n > 10:
            smallest_idx = np.argpartition(scaled, 10)[:10]
            largest_idx = np.argpartition(scaled, -10)[-10:]
        else:
            smallest_idx = np.arange(n)
            largest_idx = np.arange(n)

        keep_idx = np.unique(np.concatenate([smallest_idx, largest_idx]))

        mask = np.ones(n, dtype=bool)
        mask[keep_idx] = False
        scaled[mask] = 0

        df_norm.loc[idx] = scaled

    return df_norm


def nrmse(true, pred):
    """Compute normalized root mean squared error.

    The RMSE is normalized by the range of the true matrix.

    Parameters
    ----------
    true : array-like
        True matrix.
    pred : array-like
        Predicted matrix.

    Returns
    -------
    float
        Normalized RMSE.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if true.shape != pred.shape:
        raise ValueError("true and pred must have the same shape")

    rmse_val = np.sqrt(np.mean((true - pred) ** 2))
    data_range = true.max() - true.min()

    if data_range == 0:
        return np.nan

    return rmse_val / data_range


def corr(true, pred):
    """Compute Pearson correlation between two flattened matrices.

    Parameters
    ----------
    true : array-like
        True matrix.
    pred : array-like
        Predicted matrix.

    Returns
    -------
    float
        Pearson correlation between flattened inputs.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if true.shape != pred.shape:
        raise ValueError("true and pred must have the same shape")

    t = true.flatten()
    p = pred.flatten()

    if np.std(t) == 0 or np.std(p) == 0:
        return 0.0

    return np.corrcoef(t, p)[0, 1]


def compute_auc_scores(true_matrix, pred_matrix):
    """Compute AUROC and AUPRC for matrix-valued predictions.

    Parameters
    ----------
    true_matrix : array-like
        Ground-truth binary matrix.
    pred_matrix : array-like
        Predicted scores or probabilities.

    Returns
    -------
    tuple[float, float]
        AUROC and AUPRC scores.
    """
    y_true = np.asarray(true_matrix).astype(float).flatten()
    y_score = np.asarray(pred_matrix).astype(float).flatten()

    if len(np.unique(y_true)) < 2:
        raise ValueError("True matrix must contain at least one positive and one negative label.")

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    return auroc, auprc


def plot_matrix_hist(mat, bins=50):
    """Plot a histogram of matrix entries.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix.
    bins : int, default=50
        Number of histogram bins.
    """
    values = mat.flatten()

    plt.hist(values, bins=bins, edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Matrix Values")
    plt.show()


def mediation_support_stability(ab_list, threshold):
    """Compute support stability across bootstrap estimates.

    Stability is measured by the average pairwise Jaccard similarity of the
    supports of estimated `alpha * beta` matrices.

    Parameters
    ----------
    ab_list : list[np.ndarray]
        List of estimated `alpha * beta` matrices. All matrices must have the
        same shape.
    threshold : float
        Threshold applied to `|alpha * beta|` to define support.

    Returns
    -------
    tuple[float, np.ndarray]
        Average pairwise Jaccard similarity and the full pairwise Jaccard
        similarity matrix.

    Raises
    ------
    ValueError
        If fewer than two matrices are provided or if matrix shapes differ.
    """
    if len(ab_list) < 2:
        raise ValueError("Need at least two matrices to assess stability.")

    shape0 = ab_list[0].shape
    for i, mat in enumerate(ab_list):
        if mat.shape != shape0:
            raise ValueError(f"Matrix at index {i} has shape {mat.shape}, expected {shape0}.")

    B = len(ab_list)
    supports = [(np.abs(mat) > threshold) for mat in ab_list]
    supports_flat = [s.flatten() for s in supports]

    jaccard_matrix = np.zeros((B, B), dtype=float)

    for i in range(B):
        jaccard_matrix[i, i] = 1.0

        for j in range(i + 1, B):
            s_i = supports_flat[i]
            s_j = supports_flat[j]

            intersection = np.logical_and(s_i, s_j).sum()
            union = np.logical_or(s_i, s_j).sum()

            if union == 0:
                jaccard = 1.0
            else:
                jaccard = intersection / union

            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard

    if B > 1:
        triu_indices = np.triu_indices(B, k=1)
        stability = jaccard_matrix[triu_indices].mean()
    else:
        stability = 1.0

    return stability, jaccard_matrix
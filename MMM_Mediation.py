import os
import random

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import MultiTaskElasticNet


class MMM_Mediation_main:
    """Main class for many-to-many-to-many mediation modeling.

    This class fits elastic-net models for mediator prediction and outcome
    prediction in a multivariate mediation setting.

    Attributes
    ----------
    Y : np.ndarray
        Outcome matrix with shape `(T, N)`.
    M : np.ndarray
        Mediator matrix with shape `(p, N)`.
    X : np.ndarray
        Exposure matrix with shape `(q, N)`.
    Z : np.ndarray
        Covariate matrix with shape `(s, N)`.
    Y_test : np.ndarray
        Test outcome matrix with shape `(T, 1)`.
    M_test : np.ndarray
        Test mediator matrix with shape `(p, 1)`.
    X_test : np.ndarray
        Test exposure matrix with shape `(q, 1)`.
    Z_test : np.ndarray
        Test covariate matrix with shape `(s, 1)`.
    """

    def __init__(self):
        """Initialize data containers and model objects."""
        self.Y = None
        self.M = None
        self.X = None
        self.Z = None

        self.Y_test = None
        self.M_test = None
        self.X_test = None
        self.Z_test = None

        self.Y_ = None
        self.U = None
        self.M_ = None
        self.R = None

        self.Elas_Y = None
        self.Elas_M = None

    def var_change(self):
        """Transpose variables and construct design matrices for training.

        Constructs
        ----------
        Y_ : np.ndarray
            Transposed outcome matrix with shape `(N, T)`.
        U : np.ndarray
            Joint design matrix `[M, X, Z]` with shape `(N, p + q + s)`.
        M_ : np.ndarray
            Transposed mediator matrix with shape `(N, p)`.
        R : np.ndarray
            Exposure-covariate design matrix `[X, Z]` with shape `(N, q + s)`.
        """
        self.Y_ = self.Y.T
        self.U = np.concatenate([self.M.T, self.X.T, self.Z.T], axis=1)
        self.M_ = self.M.T
        self.R = np.concatenate([self.X.T, self.Z.T], axis=1)

    def Training(
        self,
        alpha_M=0.1,
        alpha_Y=0.1,
        l1_ratio_Y=0.5,
        l1_ratio_M=0.5,
        coef_check=False,
        max_iter=10000,
    ):
        """Train mediator and outcome with two elastic-net models, one for mediation, one for prediction.

        Parameters
        ----------
        alpha_M : float, default=0.1
            Regularization strength for the mediator model.
        alpha_Y : float, default=0.1
            Regularization strength for the outcome model.
        l1_ratio_Y : float, default=0.5
            Elastic-net mixing parameter for the outcome model.
        l1_ratio_M : float, default=0.5
            Elastic-net mixing parameter for the mediator model.
        coef_check : bool, default=False
            If `True`, return the estimated coefficient blocks.
        max_iter : int, default=10000
            Maximum number of optimization iterations.

        Returns
        -------
        tuple, optional
            If `coef_check=True`, returns
            `(alpha, zeta, beta, gamma, eta, Y_coef)`.
            Otherwise, returns `None`.
        """
        self.var_change()

        self.Elas_Y = MultiTaskElasticNet(
            alpha=alpha_Y,
            l1_ratio=l1_ratio_Y,
            max_iter=max_iter,
            tol=1e-6,
        )
        self.Elas_M = MultiTaskElasticNet(
            alpha=alpha_M,
            l1_ratio=l1_ratio_M,
            max_iter=max_iter,
            tol=1e-6,
        )

        self.Elas_M.fit(self.R, self.M_)
        M_coef = self.Elas_M.coef_.T

        alpha = M_coef[: self.X.shape[0], :]
        zeta = M_coef[self.X.shape[0] : self.X.shape[0] + self.Z.shape[0], :]

        self.Elas_Y.fit(self.U, self.Y_)
        Y_coef = self.Elas_Y.coef_.T

        beta = Y_coef[: self.M.shape[0], :]
        gamma = Y_coef[self.M.shape[0] : self.M.shape[0] + self.X.shape[0], :]
        eta = Y_coef[self.M.shape[0] + self.X.shape[0] :, :]

        if coef_check:
            return alpha, zeta, beta, gamma, eta, Y_coef

    def Training_noM(
        self,
        alpha_Y=0.1,
        l1_ratio_Y=0.5,
        max_iter=10000,
    ):
        """Train the outcome model using exposures and covariates only.

        Parameters
        ----------
        alpha_Y : float, default=0.1
            Regularization strength for the outcome model.
        l1_ratio_Y : float, default=0.5
            Elastic-net mixing parameter for the outcome model.
        l1_ratio_M : float, default=0.5
            Unused parameter kept for API compatibility.
        coef_check : bool, default=False
            Unused parameter kept for API compatibility.
        max_iter : int, default=10000
            Maximum number of optimization iterations.

        Returns
        -------
        np.ndarray
            Outcome coefficient matrix with shape `(q + s, T)`.
        """
        self.U = np.concatenate([self.X.T, self.Z.T], axis=1)
        self.Y_ = self.Y.T

        self.Elas_Y = MultiTaskElasticNet(
            alpha=alpha_Y,
            l1_ratio=l1_ratio_Y,
            max_iter=max_iter,
            tol=1e-6,
        )

        self.Elas_Y.fit(self.U, self.Y_)
        Y_coef = self.Elas_Y.coef_.T

        return Y_coef

    def Testing_noM(self):
        """Predict outcomes using exposures and covariates only.

        Returns
        -------
        np.ndarray
            Predicted outcome matrix with shape `(1, T)`.
        """
        U_test = np.concatenate([self.X_test.T, self.Z_test.T], axis=1)
        pred_Y = self.Elas_Y.predict(U_test)

        return pred_Y

    def Training_noX(
        self,
        alpha_Y=0.1,
        l1_ratio_Y=0.5,
        max_iter=10000,
    ):
        """Train the outcome model using mediators and covariates only.

        Parameters
        ----------
        alpha_Y : float, default=0.1
            Regularization strength for the outcome model.
        l1_ratio_Y : float, default=0.5
            Elastic-net mixing parameter for the outcome model.
        l1_ratio_M : float, default=0.5
            Unused parameter kept for API compatibility.
        coef_check : bool, default=False
            Unused parameter kept for API compatibility.
        max_iter : int, default=10000
            Maximum number of optimization iterations.
        """
        self.U = np.concatenate([self.M.T, self.Z.T], axis=1)
        self.Y_ = self.Y.T

        self.Elas_Y = MultiTaskElasticNet(
            alpha=alpha_Y,
            l1_ratio=l1_ratio_Y,
            max_iter=max_iter,
            tol=1e-6,
        )

        self.Elas_Y.fit(self.U, self.Y.T)

    def Testing_noX(self):
        """Predict outcomes using mediators and covariates only.

        Returns
        -------
        np.ndarray
            Predicted outcome matrix with shape `(1, T)`.
        """
        U_test = np.concatenate([self.M_test.T, self.Z_test.T], axis=1)

        return self.Elas_Y.predict(U_test)

    def Training_MandX(
        self,
        alpha_Y=0.1,
        l1_ratio_Y=0.5,
        l1_ratio_M=0.5,
        coef_check=False,
        max_iter=10000,
    ):
        """Train the outcome model using mediators, exposures, and covariates directly without mediation (i.e. only 1 ElasticNet for prediction).

        Parameters
        ----------
        alpha_Y : float, default=0.1
            Regularization strength for the outcome model.
        l1_ratio_Y : float, default=0.5
            Elastic-net mixing parameter for the outcome model.
        l1_ratio_M : float, default=0.5
            Unused parameter kept for API compatibility.
        coef_check : bool, default=False
            Unused parameter kept for API compatibility.
        max_iter : int, default=10000
            Maximum number of optimization iterations.

        Returns
        -------
        np.ndarray
            Outcome coefficient matrix.
        """
        self.U = np.concatenate([self.M.T, self.X.T, self.Z.T], axis=1)
        self.Y_ = self.Y.T

        self.Elas_Y = MultiTaskElasticNet(
            alpha=alpha_Y,
            l1_ratio=l1_ratio_Y,
            max_iter=max_iter,
            tol=1e-6,
        )

        self.Elas_Y.fit(self.U, self.Y_)
        Y_coef = self.Elas_Y.coef_.T

        return Y_coef

    def Testing_MandX(self):
        """Predict outcomes using mediators, exposures, and covariates directly without mediation (i.e. only 1 ElasticNet for prediction).

        Returns
        -------
        np.ndarray
            Predicted outcome matrix with shape `(1, T)`.
        """
        U_test = np.concatenate([self.M_test.T, self.X_test.T, self.Z_test.T], axis=1)
        pred_Y = self.Elas_Y.predict(U_test)

        return pred_Y

    def Testing(self):
        """Predict mediators and outcomes, then compute evaluation metrics.

        The mediator model first predicts `M_test` from `[X_test, Z_test]`.
        The outcome model then predicts `Y_test` from
        `[pred_M, X_test, Z_test]`.

        Returns
        -------
        tuple
            A tuple containing mean squared error, mean absolute error,
            eight Pearson correlations, eight Pearson p-values, predicted
            mediators, and predicted outcomes.
        """
        R_test = np.concatenate([self.X_test.T, self.Z_test.T], axis=1)
        pred_M = self.Elas_M.predict(R_test)

        U_test = np.concatenate([pred_M, self.X_test.T, self.Z_test.T], axis=1)
        pred_Y = self.Elas_Y.predict(U_test)

        mse = np.mean((pred_Y.T - self.Y_test) ** 2)
        mae = np.mean(np.abs(pred_Y.T - self.Y_test))

        corr_1, p_value_1 = pearsonr(pred_Y.T[0, :], self.Y_test[0, :])
        corr_2, p_value_2 = pearsonr(pred_Y.T[1, :], self.Y_test[1, :])
        corr_3, p_value_3 = pearsonr(pred_Y.T[2, :], self.Y_test[2, :])
        corr_4, p_value_4 = pearsonr(pred_Y.T[3, :], self.Y_test[3, :])
        corr_5, p_value_5 = pearsonr(pred_Y.T[4, :], self.Y_test[4, :])
        corr_6, p_value_6 = pearsonr(pred_Y.T[5, :], self.Y_test[5, :])
        corr_7, p_value_7 = pearsonr(pred_Y.T[6, :], self.Y_test[6, :])
        corr_8, p_value_8 = pearsonr(pred_Y.T[7, :], self.Y_test[7, :])

        return (
            mse,
            mae,
            corr_1,
            corr_2,
            corr_3,
            corr_4,
            corr_5,
            corr_6,
            corr_7,
            corr_8,
            p_value_1,
            p_value_2,
            p_value_3,
            p_value_4,
            p_value_5,
            p_value_6,
            p_value_7,
            p_value_8,
            pred_M,
            pred_Y,
        )

    def set_seed(self, seed: int = 34):
        """Set random seeds for reproducibility.

        Parameters
        ----------
        seed : int, default=34
            Random seed used by Python, NumPy, and environment-level hash
            seeding.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.random.seed(seed)
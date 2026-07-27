from abc import ABC, abstractmethod
import numpy as np
from typing import List, Callable


class ConcaveComponent(ABC):
    """
    Abstract base class for a single concave function l_k(z).
    Since x is fixed, it is treated as an internal state of the component.
    """
    
    @abstractmethod
    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """Evaluates the function value at z."""
        pass

    @abstractmethod
    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Evaluates the gradient (or subgradient) with respect to z."""
        pass


class AffineLossComponent(ConcaveComponent):
    """
    Affine loss function: l(z) = a^T z + b
    Admits a closed-form solution in the inner oracle using the dual norm.
    """
    def __init__(self, a: np.ndarray, b: float):
        self.a = a
        self.b = b
        
    def evaluate(self, z: np.ndarray) -> float:
        return np.dot(self.a, z) + self.b
        
    def gradient(self, z: np.ndarray) -> np.ndarray:
        return self.a


class RadialLossComponent(ConcaveComponent):
    """
    Generalized radial loss function: l(z) = c_k + phi(||A z + b_vec||_2)
    where phi is concave and non-increasing.
    Admits a 1D section method solution in the inner oracle.
    """
    def __init__(self, A: np.ndarray, b_vec: np.ndarray, phi: Callable[[float], float], intercept: float = 0.0):
        self.A = A
        self.b_vec = b_vec
        self.phi = phi
        self.intercept = intercept

        # Precompute static matrices and eigendecomposition to avoid O(m^3) in the inner loop
        self.U, self.D_1d, self.VT = np.linalg.svd(A, full_matrices=False)
        self.D2 = self.D_1d ** 2

        # Precompute the constant bias term (D U^T b)
        self.y_bias = self.D_1d * (self.U.T @ self.b_vec)
        
        # Precompute the y values for all empirical data points
        self.Y_precomputed = None
    
    def setup_dataset(self, Z: np.ndarray):
        """
        Precomputes Y = D^2 V^T Z^T + y_bias for the entire dataset simultaneously.
        Z is expected to be of shape (t, d).
        """
        # (self.VT @ Z.T) has shape (k, t)
        # Multiply by D2 (shape k) and add bias (shape k) via broadcasting
        Y_T = self.D2[:, None] * (self.VT @ Z.T) + self.y_bias[:, None]
        
        # Transpose back to (t, k) so row i corresponds to data point i
        self.Y_precomputed = Y_T.T  
        
    def get_y(self, idx: int) -> np.ndarray:
        """Fetches the precomputed y vector using array indexing."""
        return self.Y_precomputed[idx]
        
    def evaluate(self, z: np.ndarray) -> float:
        norm_val = np.linalg.norm(self.A @ z + self.b_vec, ord=2)
        return self.intercept + self.phi(norm_val)
        
    def gradient(self, z: np.ndarray) -> np.ndarray:
        # Note: If you only use the exact solver, the PGD gradient is technically 
        # not needed, but good practice to implement for fallback/testing.
        # if using the fallback PGD solver for p != 2.0.
        diff = self.A @ z + self.b_vec
        norm_val = np.linalg.norm(diff, ord=2)
        if norm_val < 1e-8:
            return np.zeros_like(z)
        
        # Requires derivative of phi; omitted here for brevity as the exact solver 
        # only requires evaluate() and the A/b matrices.
        raise NotImplementedError("Gradient not required for the exact radial solver.")


class PointwiseMaxLoss:
    """
    Represents the pointwise maximum of K concave functions:
    l(z) = max_{k in [K]} l_k(z)
    """
    def __init__(self, components: List[ConcaveComponent]):
        if not components:
            raise ValueError("Must provide at least one concave component.")
        self.components = components
        self.K = len(components)

    def setup_dataset(self, Z: np.ndarray):
        """Propagates the dataset down to any components that require precomputation."""
        for comp in self.components:
            if hasattr(comp, 'setup_dataset'):
                comp.setup_dataset(Z)

    def evaluate(self, z: np.ndarray) -> float:
        """Evaluates the overall max loss at z."""
        vals = [comp.evaluate(z) for comp in self.components]
        return np.max(vals, axis=0)
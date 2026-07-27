import numpy as np
from typing import List, Tuple, Optional
from numba import njit, prange
from inner.losses import PointwiseMaxLoss


# =====================================================================
# Data Preparation
# =====================================================================
class InnerMaxResult:
    """Data class to hold the results of the inner maximization problem."""
    def __init__(self, 
                 worst_case_loss: float, 
                 optimal_lambda: float, 
                 optimal_budgets: np.ndarray,
                 worst_case_distribution: Tuple[np.ndarray, np.ndarray],
                 active_components: Optional[List[List[int]]] = None):
        self.worst_case_loss = worst_case_loss
        self.optimal_lambda = optimal_lambda
        self.optimal_budgets = optimal_budgets

        # Tuple containing: (weights_array, support_points_array)
        # Weights sum to 1.0. Support points shape is (num_measures, d)
        self.worst_case_distribution = worst_case_distribution
        self.active_components = active_components


def prepare_numba_arrays(z_empirical: np.ndarray, loss: 'PointwiseMaxLoss'):
    """
    Extracts all component properties into homogeneous NumPy arrays.
    Returns: Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked
    """
    t, d = z_empirical.shape
    K = loss.K

    # Get the rank of the SVD (number of singular values)
    r = loss.components[0].D_1d.shape[0] 
    
    # Pre-allocate stacked arrays
    Y_stacked = np.zeros((t, K, r))
    D2_stacked = np.zeros((K, r))
    VT_stacked = np.zeros((K, r, d))
    
    # O(d) projection arrays
    c_proj = np.zeros((t, K, r))
    C_perp_sq = np.zeros((t, K))
    nominal_evals = np.zeros((t, K))
    intercepts = np.zeros(K)
    
    for k, comp in enumerate(loss.components):
        Y_stacked[:, k, :] = comp.Y_precomputed
        D2_stacked[k, :] = comp.D2
        VT_stacked[k, :, :] = comp.VT
        intercepts[k] = comp.intercept
        U_k = comp.U
        
        for i in range(t):
            # c = A * z_i + b
            c_val = comp.A @ z_empirical[i] + comp.b_vec
            nominal_evals[i, k] = intercepts[k] + comp.phi(np.linalg.norm(c_val))
            
            # 1. Project C onto U's orthonormal space
            c_proj_val = U_k.T @ c_val
            c_proj[i, k, :] = c_proj_val
            
            # 2. Calculate the orthogonal remainder norm squared
            c_reconstructed = U_k @ c_proj_val
            C_perp_sq[i, k] = np.sum((c_val - c_reconstructed)**2)
            
    return Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked


# =====================================================================
# NUMBA JIT MATH KERNELS
# =====================================================================
PHI = 0.6180339887498949  # (sqrt(5) - 1) / 2

@njit(fastmath=True, cache=True)
def numba_phi(x: float) -> float:
    """
    Placeholder for the specific phi function. 
    Here we use -x^2 for the quadratic case.
    """
    return -(x**2)


@njit(fastmath=True, cache=True)
def numba_inner_oracle(idx: int, k: int, alpha: float, beta: float, 
                       Y_stacked: np.ndarray, D2_stacked: np.ndarray, c_proj: np.ndarray, 
                       C_perp_sq: np.ndarray, nominal_evals: np.ndarray, intercepts: np.ndarray) -> float:
    """
    Solves max_{||q||_2 <= beta} alpha * phi(|| A(z - q/alpha) + b ||_2)
    """
    if alpha < 1e-8:
        return 0.0
    if beta < 1e-8:
        return alpha * nominal_evals[idx, k]
        
    y = Y_stacked[idx, k]
    D2 = D2_stacked[k]
    scaled_D2 = D2 / alpha
    r = len(y)

    # 1. Hebden-Reinsch Method for Secular Equation
    # Check unconstrained case first (rho = 0)
    norm0_sq = 0.0
    for j in range(r):
        if scaled_D2[j] > 1e-12:
            norm0_sq += (y[j] / scaled_D2[j])**2
        elif abs(y[j]) > 1e-12:
            norm0_sq = np.inf
            
    if norm0_sq <= beta**2:
        rho = 0.0
    else:
        rho = 0.0 
        for _ in range(50):
            norm_q_sq = 0.0
            sum_deriv = 0.0
            for j in range(r):
                denom = rho + scaled_D2[j]
                term_sq = (y[j] / denom)**2  
                
                norm_q_sq += term_sq
                sum_deriv += term_sq / denom
                
            norm_q = np.sqrt(norm_q_sq)
            
            # Check relative error
            if abs(norm_q - beta) < 1e-5 * beta + 1e-12:
                break
                
            # Trust-region stable Newton step (Hebden-Reinsch)
            delta_rho = (norm_q_sq / sum_deriv) * ((norm_q - beta) / beta)
            rho = max(1e-10, rho + delta_rho)  # Safeguard

    # 2. O(d) Residual Evaluation           
    residual_sq = C_perp_sq[idx, k]
    for j in range(r):
        # q_tilde_j = y_j / (rho + s_j)
        q_tilde_j = y[j] / (rho + scaled_D2[j])
        
        # A * q_opt_j = D_j * q_tilde_j
        x_j = (np.sqrt(D2[j]) * q_tilde_j) / alpha
        
        # Accumulate the projected distance
        residual_sq += (c_proj[idx, k, j] - x_j)**2
    
    return alpha * (intercepts[k] + numba_phi(np.sqrt(residual_sq)))


@njit(fastmath=True, cache=True)
def numba_solve_inner(idx: int, k1: int, k2: int, alpha1: float, alpha2: float, b: float, eta_in: float,
                      Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts) -> float:
    """Golden section search over budget split beta1 + beta2 = b"""
    L_beta = 0.0
    U_beta = b
    a_beta = U_beta - PHI * (U_beta - L_beta)
    b_beta = L_beta + PHI * (U_beta - L_beta)
    
    def eval_split(beta1):
        v1 = numba_inner_oracle(idx, k1, alpha1, beta1, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
        v2 = numba_inner_oracle(idx, k2, alpha2, b - beta1, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
        return v1 + v2

    V_a = eval_split(a_beta)
    V_b = eval_split(b_beta)
    
    while (U_beta - L_beta) > eta_in:
        if V_a > V_b:
            U_beta = b_beta
            b_beta = a_beta
            V_b = V_a
            a_beta = U_beta - PHI * (U_beta - L_beta)
            V_a = eval_split(a_beta)
        else:
            L_beta = a_beta
            a_beta = b_beta
            V_a = V_b
            b_beta = L_beta + PHI * (U_beta - L_beta)
            V_b = eval_split(b_beta)
    
    return max(V_a, V_b)


@njit(fastmath=True, cache=True)
def numba_evaluate_S_i(idx: int, b: float, K: int, eta_out: float, eta_in: float,
                       Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts) -> tuple:
    """Evaluates the decoupled max-utility function for sample z_i."""
    if b <= 0.0:
        max_val = -np.inf
        best_k = 0
        for k in range(K):
            if nominal_evals[idx, k] > max_val:
                max_val = nominal_evals[idx, k]
                best_k = k
        return max_val, best_k, -1
        
    if K == 1:
        val = numba_inner_oracle(idx, 0, 1.0, b, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
        return val, 0, -1
        
    max_S_i = -np.inf
    best_k1, best_k2 = 0, -1
    
    # Unrolled combinations loop for Numba
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            L_alpha = 0.0
            U_alpha = 1.0
            a_alpha = U_alpha - PHI * (U_alpha - L_alpha)
            b_alpha = L_alpha + PHI * (U_alpha - L_alpha)
            
            V_a = numba_solve_inner(idx, k1, k2, a_alpha, 1.0 - a_alpha, b, eta_in, 
                                    Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
            V_b = numba_solve_inner(idx, k1, k2, b_alpha, 1.0 - b_alpha, b, eta_in, 
                                    Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
                                    
            while (U_alpha - L_alpha) > eta_out:
                if V_a > V_b:
                    U_alpha = b_alpha
                    b_alpha = a_alpha
                    V_b = V_a
                    a_alpha = U_alpha - PHI * (U_alpha - L_alpha)
                    V_a = numba_solve_inner(idx, k1, k2, a_alpha, 1.0 - a_alpha, b, eta_in, 
                                            Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
                else:
                    L_alpha = a_alpha
                    a_alpha = b_alpha
                    V_a = V_b
                    b_alpha = L_alpha + PHI * (U_alpha - L_alpha)
                    V_b = numba_solve_inner(idx, k1, k2, b_alpha, 1.0 - b_alpha, b, eta_in, 
                                            Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)

            pair_max = max(V_a, V_b)

            if pair_max > max_S_i:
                max_S_i = pair_max
                best_k1, best_k2 = k1, k2
                
    return max_S_i, best_k1, best_k2


@njit(fastmath=True, parallel=True, cache=True)
def numba_solve_all_decoupled(t: int, lambda_val: float, 
                              b_lower_bounds: np.ndarray, b_upper_bounds: np.ndarray,
                              K: int, eta_b: float, eta_out: float, eta_in: float,
                              Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts):
    """
    Parallelized wrapper that solves the decoupled subproblems for all t data points simultaneously 
    using all available CPU cores.
    """
    b_hat = np.zeros(t)
    k1_hat = np.zeros(t, dtype=np.int32)
    k2_hat = np.zeros(t, dtype=np.int32)
    
    # prange tells Numba to distribute this loop across multiple CPU threads!
    for i in prange(t):
        res = numba_solve_decoupled(
            i, lambda_val, b_lower_bounds[i], b_upper_bounds[i], K, eta_b, eta_out, eta_in,
            Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts
        )
        b_hat[i], k1_hat[i], k2_hat[i] = res

    return b_hat, k1_hat, k2_hat


@njit(fastmath=True, cache=True)
def numba_solve_decoupled(idx: int, lambda_val: float, b_low_init: float, b_high_init: float,
                          K: int, eta_b: float, eta_out: float, eta_in: float,
                          Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts) -> tuple:
    """Golden section search to maximize S_i(b) - lambda * b."""
    b_low = b_low_init
    b_high = b_high_init

    z1 = b_high - PHI * (b_high - b_low)
    z2 = b_low + PHI * (b_high - b_low)

    val1, k1_1, k2_1 = numba_evaluate_S_i(idx, z1, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
    f_z1 = val1 - lambda_val * z1

    val2, k1_2, k2_2 = numba_evaluate_S_i(idx, z2, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
    f_z2 = val2 - lambda_val * z2
    
    while (b_high - b_low) > eta_b:
        if f_z1 < f_z2:
            b_low = z1
            z1 = z2
            f_z1 = f_z2
            k1_1, k2_1 = k1_2, k2_2

            z2 = b_low + PHI * (b_high - b_low)
            val2, k1_2, k2_2 = numba_evaluate_S_i(idx, z2, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
            f_z2 = val2 - lambda_val * z2
        else:
            b_high = z2
            z2 = z1
            f_z2 = f_z1
            k1_2, k2_2 = k1_1, k2_1

            z1 = b_high - PHI * (b_high - b_low)
            val1, k1_1, k2_1 = numba_evaluate_S_i(idx, z1, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
            f_z1 = val1 - lambda_val * z1

    if f_z1 > f_z2:
        best_b, best_f, best_k1, best_k2 = z1, f_z1, k1_1, k2_1
    else:
        best_b, best_f, best_k1, best_k2 = z2, f_z2, k1_2, k2_2

    # --- Boundary Check ---
    val_low, k1_low, k2_low = numba_evaluate_S_i(idx, b_low_init, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
    f_low = val_low - lambda_val * b_low_init
    
    if f_low >= best_f - 1e-12:
        return b_low_init, k1_low, k2_low
        
    val_high, k1_high, k2_high = numba_evaluate_S_i(idx, b_high_init, K, eta_out, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts)
    f_high = val_high - lambda_val * b_high_init
    
    if f_high >= best_f - 1e-12:
        return b_high_init, k1_high, k2_high

    return best_b, best_k1, best_k2


# =====================================================================
# NUMBA JIT EXTRACTION KERNELS
# =====================================================================
@njit(fastmath=True, cache=True)
def numba_inner_oracle_extract(idx, k, alpha, beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d):
    q_opt = np.zeros(d)
    if alpha < 1e-8:
        return 0.0, q_opt
    if beta < 1e-8:
        return alpha * nominal_evals[idx, k], q_opt

    y = Y_stacked[idx, k]
    scaled_D2 = D2_stacked[k] / alpha
    r = len(y)

    norm0_sq = 0.0
    for j in range(r):
        if scaled_D2[j] > 1e-12:
            norm0_sq += (y[j] / scaled_D2[j])**2
        elif abs(y[j]) > 1e-12:
            norm0_sq = np.inf
            
    if norm0_sq <= beta**2:
        rho = 0.0
    else:
        rho = 0.0
        for _ in range(50):
            norm_q_sq = 0.0
            sum_deriv = 0.0
            for j in range(r):
                denom = rho + scaled_D2[j]
                term_sq = (y[j] / denom)**2  
                norm_q_sq += term_sq
                sum_deriv += term_sq / denom
                
            norm_q = np.sqrt(norm_q_sq)
            if abs(norm_q - beta) < 1e-5 * beta + 1e-12:
                break
                
            delta_rho = (norm_q_sq / sum_deriv) * ((norm_q - beta) / beta)
            rho = max(1e-10, rho + delta_rho)  # Safeguard

    residual_sq = C_perp_sq[idx, k]
    for j in range(r):
        q_tilde_j = y[j] / (rho + scaled_D2[j])
        q_opt += VT_stacked[k, j, :] * q_tilde_j
        
        x_j = (np.sqrt(D2_stacked[k, j]) * q_tilde_j) / alpha
        residual_sq += (c_proj[idx, k, j] - x_j)**2
        
    val = alpha * (intercepts[k] + numba_phi(np.sqrt(residual_sq)))
    return val, q_opt


@njit(fastmath=True, cache=True)
def numba_solve_inner_extract(idx, k1, k2, alpha1, alpha2, b, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d):
    L_beta = 0.0
    U_beta = b
    a_beta = U_beta - PHI * (U_beta - L_beta)
    b_beta = L_beta + PHI * (U_beta - L_beta)
    
    v1_a, q1_a = numba_inner_oracle_extract(idx, k1, alpha1, a_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
    v2_a, q2_a = numba_inner_oracle_extract(idx, k2, alpha2, b - a_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
    V_a = v1_a + v2_a

    v1_b, q1_b = numba_inner_oracle_extract(idx, k1, alpha1, b_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
    v2_b, q2_b = numba_inner_oracle_extract(idx, k2, alpha2, b - b_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
    V_b = v1_b + v2_b

    while (U_beta - L_beta) > eta_in:
        if V_a > V_b:
            U_beta = b_beta
            b_beta = a_beta
            V_b = V_a
            q1_b, q2_b = q1_a, q2_a
            a_beta = U_beta - PHI * (U_beta - L_beta)
            v1_a, q1_a = numba_inner_oracle_extract(idx, k1, alpha1, a_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            v2_a, q2_a = numba_inner_oracle_extract(idx, k2, alpha2, b - a_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            V_a = v1_a + v2_a
        else:
            L_beta = a_beta
            a_beta = b_beta
            V_a = V_b
            q1_a, q2_a = q1_b, q2_b
            b_beta = L_beta + PHI * (U_beta - L_beta)
            v1_b, q1_b = numba_inner_oracle_extract(idx, k1, alpha1, b_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            v2_b, q2_b = numba_inner_oracle_extract(idx, k2, alpha2, b - b_beta, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            V_b = v1_b + v2_b
            
    if V_a > V_b:
        return V_a, q1_a, q2_a
    return V_b, q1_b, q2_b


@njit(fastmath=True, parallel=True, cache=True)
def numba_extract_final_states(t, b_hat, k1_hat, k2_hat, K, d, eta_out, eta_in,
                               Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, 
                               intercepts, VT_stacked):
    total_loss = np.zeros(t)
    alphas_out = np.zeros((t, K))
    qs_out = np.zeros((t, K, d))
    
    for i in prange(t):
        k1 = k1_hat[i]
        k2 = k2_hat[i]
        b = b_hat[i]
        
        # Edge Case: Zero Budget
        if b <= 0.0:
            alphas_out[i, k1] = 1.0
            total_loss[i] = nominal_evals[i, k1]
        
        # Edge Case: Only 1 component is active
        elif k2 == -1:
            val, q = numba_inner_oracle_extract(
                i, k1, 1.0, b, Y_stacked, D2_stacked, c_proj, C_perp_sq, 
                nominal_evals, intercepts, VT_stacked, d
            )
            alphas_out[i, k1] = 1.0
            qs_out[i, k1] = q
            total_loss[i] = val
            
        # Target Extraction: Re-optimize for the cached k1, k2
        else:
            L_alpha = 0.0
            U_alpha = 1.0
            a_alpha = U_alpha - PHI * (U_alpha - L_alpha)
            b_alpha = L_alpha + PHI * (U_alpha - L_alpha)
            
            V_a, q1_a, q2_a = numba_solve_inner_extract(i, k1, k2, a_alpha, 1.0 - a_alpha, b, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            V_b, q1_b, q2_b = numba_solve_inner_extract(i, k1, k2, b_alpha, 1.0 - b_alpha, b, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            
            while (U_alpha - L_alpha) > eta_out:
                if V_a > V_b:
                    U_alpha = b_alpha
                    b_alpha = a_alpha
                    V_b = V_a
                    q1_b, q2_b = q1_a, q2_a
                    a_alpha = U_alpha - PHI * (U_alpha - L_alpha)
                    V_a, q1_a, q2_a = numba_solve_inner_extract(i, k1, k2, a_alpha, 1.0 - a_alpha, b, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
                else:
                    L_alpha = a_alpha
                    a_alpha = b_alpha
                    V_a = V_b
                    q1_a, q2_a = q1_b, q2_b
                    b_alpha = L_alpha + PHI * (U_alpha - L_alpha)
                    V_b, q1_b, q2_b = numba_solve_inner_extract(i, k1, k2, b_alpha, 1.0 - b_alpha, b, eta_in, Y_stacked, D2_stacked, c_proj, C_perp_sq, nominal_evals, intercepts, VT_stacked, d)
            
            if V_a > V_b:
                alphas_out[i, k1] = a_alpha
                alphas_out[i, k2] = 1.0 - a_alpha
                qs_out[i, k1] = q1_a
                qs_out[i, k2] = q2_a
                total_loss[i] = V_a
            else:
                alphas_out[i, k1] = b_alpha
                alphas_out[i, k2] = 1.0 - b_alpha
                qs_out[i, k1] = q1_b
                qs_out[i, k2] = q2_b
                total_loss[i] = V_b

    return total_loss, alphas_out, qs_out


# =====================================================================
# THE MANAGER CLASS
# =====================================================================

class InnerSolverl2:
    """
    High-performance wrapper class for the W-DRO L2 Inner Maximization.
    Incorporates precomputations, Numba JIT kernels and an one-time extraction.
    """
    def __init__(self, 
                 epsilon: float,  
                 eta_lambda: float = 1e-3, 
                 eta_b: float = 1e-3, 
                 eta_out: float = 1e-3, 
                 eta_in: float = 1e-3):
        
        self.epsilon = epsilon
        self.eta_lambda = eta_lambda
        self.eta_b = eta_b
        self.eta_out = eta_out
        self.eta_in = eta_in

    def solve(self, z_empirical: np.ndarray, loss: PointwiseMaxLoss) -> InnerMaxResult:
        t, d = z_empirical.shape 
        rho_t = self.epsilon * t  
        K = loss.K
        
        # Trigger global precomputation on the loss object
        loss.setup_dataset(z_empirical)
        
        # Extract arrays for Numba
        arrays = prepare_numba_arrays(z_empirical, loss)

        # Initialize the global absolute bounds for b
        b_hat = np.zeros(t)
        b_lower = np.zeros(t)
        b_upper = np.full(t, rho_t)
        
        # ---------------------------------------------------------
        # Phase 1: Fast bracketing using Numba kernels
        # ---------------------------------------------------------
        best_res = None
        auxi_res = None

        lambda_high = 1.0
        while True:
            res = numba_solve_all_decoupled(t, lambda_high, b_lower, b_upper, K, self.eta_b, self.eta_out, self.eta_in, *arrays[:-1])
            b_hat = res[0]

            if np.sum(b_hat) <= rho_t:
                b_lower = b_hat.copy()
                best_res = res
                break 

            b_upper = b_hat.copy()
            auxi_res = res
            lambda_high *= 2.0

        lambda_low = lambda_high / 2.0 if lambda_high > 1.99 else 0.0
        
        # ---------------------------------------------------------
        # Phase 2: Fast bisection/golden-section using Numba kernels
        # ---------------------------------------------------------
        while (lambda_high - lambda_low) > self.eta_lambda:
            lambda_mid = (lambda_low + lambda_high) / 2.0
            
            res = numba_solve_all_decoupled(t, lambda_mid, b_lower, b_upper, K, self.eta_b, self.eta_out, self.eta_in, *arrays[:-1])
            b_hat = res[0]
                
            if np.sum(b_hat) > rho_t:
                lambda_low = lambda_mid 
                b_upper = b_hat.copy()
                auxi_res = res
            else:
                lambda_high = lambda_mid 
                b_lower = b_hat.copy()
                best_res = res
                
        # Record lambda_opt
        lambda_opt = lambda_high

        # Compute auxiliary solution if not already computed
        if auxi_res is None:
            auxi_res = numba_solve_all_decoupled(t, lambda_low, b_lower, b_upper, K, self.eta_b, self.eta_out, self.eta_in, *arrays[:-1])

        # ---------------------------------------------------------
        # Final Pass: Extract worst-case distribution
        # ---------------------------------------------------------
        b_hat, k1_hat, k2_hat = best_res
        b_hat_aux, k1_hat_aux, k2_hat_aux = auxi_res
        s_hat = np.sum(b_hat)
        s_hat_aux = np.sum(b_hat_aux)

        # Convex combination of the two solutions if necessary
        if s_hat < rho_t and s_hat_aux > rho_t:
            theta = (rho_t - s_hat) / (s_hat_aux - s_hat)
            b_final = (1.0 - theta) * b_hat + theta * b_hat_aux
            k1_final = np.where(theta < 0.5, k1_hat, k1_hat_aux)
            k2_final = np.where(theta < 0.5, k2_hat, k2_hat_aux)
        elif s_hat_aux <= rho_t:
            b_final = b_hat_aux
            k1_final = k1_hat_aux
            k2_final = k2_hat_aux
        else:
            b_final = b_hat
            k1_final = k1_hat
            k2_final = k2_hat

        total_loss_arr, alphas_arr, qs_arr = numba_extract_final_states(
            t, b_final, k1_final, k2_final, K, d, self.eta_out, self.eta_in, 
            *arrays
        )

        total_loss = np.sum(total_loss_arr) / t

        weights_list = []
        support_points_list = []
        active_comps_list = []
        
        for i in range(t):
            active_k_for_i = []
            
            for k in range(K):
                alpha_ik = alphas_arr[i, k]
                if alpha_ik > 1e-8:
                    weight = alpha_ik / t
                    # Reconstruct z_hat from the q shift
                    point = z_empirical[i] - (qs_arr[i, k] / alpha_ik)
                    
                    weights_list.append(weight)
                    support_points_list.append(point)
                    active_k_for_i.append(k)
            
            active_comps_list.append(active_k_for_i)

        return InnerMaxResult(
            worst_case_loss=total_loss, 
            optimal_lambda=lambda_opt, 
            optimal_budgets=b_hat,
            worst_case_distribution=(np.array(weights_list), np.array(support_points_list)),
            active_components=active_comps_list
        )
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from wdro_inner.solver import InnerMaxResult


def prepare_tangent_compression_data(
    x_bar: np.ndarray, 
    z_empirical: np.ndarray, 
    S_history: list, 
    A_list: list, 
    B_list: list, 
    C_list: list
):
    """
    Vectorizes the computation of losses, gradients, and costs for the quadratic 
    min-max problem to bypass all Python loops.
    """
    N, n = z_empirical.shape
    K = len(A_list)
    
    # 1. Flatten all historical atoms into a single massive array
    M_i_counts = [len(S) for S in S_history]
    M_total = sum(M_i_counts)
    Z_flat = np.vstack(S_history)  # Shape: (M_total, n)
    
    # 2. Map each atom to its originating empirical sample i
    emp_indices = np.repeat(np.arange(N), M_i_counts)
    
    # 3. Vectorized Cost Computation: c(z, z_hat) = ||z - z_hat||_2
    Z_hat_expanded = z_empirical[emp_indices]
    cost_vec = np.linalg.norm(Z_flat - Z_hat_expanded, ord=2, axis=1)  # (M_total,)
    C_vec = np.repeat(cost_vec, K)                                     # (M_total * K,)
    
    # 4. Preallocate dense tensors for Loss and Gradient
    loss_mat = np.zeros((M_total, K))
    grad_tensor = np.zeros((M_total, K, n))
    
    # 5. Compute analytics using NumPy Broadcasting
    for k in range(K):
        # Loss Terms
        c_k_x = x_bar.T @ C_list[k] @ x_bar
        b_k_x = B_list[k] @ x_bar
        
        # Efficient diagonal of (Z @ A_k @ Z^T)
        z_A_z = np.sum(Z_flat @ A_list[k] * Z_flat, axis=1)
        
        loss_mat[:, k] = c_k_x + Z_flat @ b_k_x - z_A_z
        
        # Gradient Terms: 2 * C_k * x + B_k^T * z
        # Note: Z_flat @ B_k computes z^T B_k, which is (B_k^T z)^T
        grad_tensor[:, k, :] = 2 * (C_list[k] @ x_bar) + Z_flat @ B_list[k]
        
    # Flatten loss matrix for the objective vector
    L_vec = loss_mat.flatten()  # (M_total * K,)
    
    # 6. Build the Gradient Constraint Matrix (G_mat)
    # Transpose grad_tensor to (n, M_total, K) and flatten to map to alpha vector
    G_mat_dense = grad_tensor.transpose(2, 0, 1).reshape(n, M_total * K)
    G_mat = sp.csr_matrix(G_mat_dense)
    
    # 7. Build the Simplex Constraint Matrix (A_eq)
    # Assigns 1.0 to the correct empirical sample row for every flattened variable
    row_idx = np.repeat(emp_indices, K)
    col_idx = np.arange(M_total * K)
    val = np.ones(M_total * K)
    A_eq = sp.csr_matrix((val, (row_idx, col_idx)), shape=(N, M_total * K))
    
    return L_vec, C_vec, G_mat, A_eq, Z_flat, M_total, K


def compress_distribution_tangent_quad(
    x_bar: np.ndarray, 
    N: int, 
    K: int,
    rho: float, 
    R: float,
    L_vec: np.ndarray, 
    C_vec: np.ndarray, 
    G_mat: sp.csr_matrix, 
    A_eq: sp.csr_matrix, 
    Z_flat: np.ndarray, 
    M_total: int
):
    """
    Solves the tangent compression program instantaneously using precomputed 
    matrices and Gurobi's native Second-Order Cone interface.
    Enforces an L1-ball trust region X = {x : ||x||_1 <= R}.
    """
    n = len(x_bar)
    num_vars = M_total * K
    
    # Initialize silent Gurobi environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    m = gp.Model("Compression_Tangent_Quad", env=env)
    
    # Variables
    alpha = m.addMVar(shape=num_vars, lb=0.0, name="alpha")
    g = m.addMVar(shape=n, lb=-GRB.INFINITY, name="g")
    nu = m.addVar(lb=0.0, name="nu")

    # --- Constraints ---
    # 1. Gradient Definition
    expected_grad = (1.0 / N) * (alpha @ G_mat.T)
    m.addConstr(g == expected_grad, name="grad_eq")
    
    # 2. L1 Trust Region: -nu <= g_i <= nu  => ||g||_inf <= nu
    m.addConstr(g <= nu, name="linf_upper")
    m.addConstr(g >= -nu, name="linf_lower")
    
    # 3. Wasserstein Budget
    m.addConstr((1.0 / N) * C_vec @ alpha <= rho, name="budget")
    
    # 4. Simplex Marginals
    m.addConstr(alpha @ A_eq.T == np.ones(N), name="simplex")
    
    # --- Objective ---
    # Tangent exact lower bound over L1 ball: E[L] - g^T x_bar - R * ||g||_inf
    tangent_loss = (1.0 / N) * L_vec @ alpha
    shift_penalty = (x_bar @ g).sum()
    trust_penalty = R * nu
    m.setObjective(tangent_loss - shift_penalty - trust_penalty, GRB.MAXIMIZE)
        
    # --- Solve ---
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed. Status code: {m.Status}")
        
    # --- Extract and Filter Distribution ---
    alpha_reshaped = alpha.X.reshape((M_total, K))
    
    # Marginalize over K components and scale by empirical weight
    marginal_atom_weights = np.sum(alpha_reshaped, axis=1) / N
    
    # Extract strictly positive weights (Guaranteed <= N + n + 1)
    tolerance = 1e-8
    active_indices = np.where(marginal_atom_weights > tolerance)[0]
    
    compressed_weights = marginal_atom_weights[active_indices]
    compressed_atoms = Z_flat[active_indices]

    optimal_budgets = A_eq @ (alpha.X * C_vec)
    
    return InnerMaxResult(
        worst_case_loss=m.ObjVal,
        optimal_lambda=[],                  # Omitted for flattened tangent program
        optimal_budgets=optimal_budgets,
        worst_case_distribution=(compressed_weights, compressed_atoms),
        active_components=[]                # Omitted for flattened tangent program
    )
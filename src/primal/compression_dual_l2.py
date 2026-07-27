import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import mosek.fusion as mf
from wdro_inner.solver import InnerMaxResult


def prepare_dual_compression_data(
    z_empirical: np.ndarray, 
    S_history: list, 
    A_list: list, 
    B_list: list, 
    C_list: list
):
    """
    Precomputes matrices and constants for the exact Dual Compression RSOCP.
    """
    N, d = z_empirical.shape
    K = len(A_list)
    
    # 1. Flatten all historical atoms
    M_i_counts = [len(S) for S in S_history]
    M_total = sum(M_i_counts)
    Z_flat = np.vstack(S_history)  # (M_total, d)
    
    # 2. Cost vector: c(z, z_hat) = ||z - z_hat||_2
    emp_indices = np.repeat(np.arange(N), M_i_counts)
    Z_hat_expanded = z_empirical[emp_indices]
    cost_vec = np.linalg.norm(Z_flat - Z_hat_expanded, ord=2, axis=1)
    C_vec = np.repeat(cost_vec, K)  # (M_total * K,)
    
    # 3. Precompute Conjugate Constants
    z_A_z_flat = np.zeros(M_total * K)
    u_mat_flat = np.zeros((M_total * K, d))
    H_list = []
    
    for k in range(K):
        # (1) Inverse square root of C_k for the RSOC formulation
        eigvals, eigvecs = np.linalg.eigh(C_list[k])
        eigvals_clipped = np.clip(eigvals, 1e-10, None)
        # H_k = 1/2 * C_k^{-1/2}
        H_k = 0.5 * eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T
        H_list.append(H_k)
        
        # (2) z^T A_k z
        z_A_z = np.sum(Z_flat @ A_list[k] * Z_flat, axis=1)
        
        # (3) u = B_k^T z (computed as Z_flat @ B_k)
        u_k = Z_flat @ B_list[k]
        
        # Interleave into flattened arrays (atom 0 comp 0, atom 0 comp 1...)
        z_A_z_flat[k::K] = z_A_z
        u_mat_flat[k::K, :] = u_k
        
    # 4. Simplex Matrix
    row_idx = np.repeat(emp_indices, K)
    col_idx = np.arange(M_total * K)
    A_eq = sp.csr_matrix((np.ones(M_total * K), (row_idx, col_idx)), shape=(N, M_total * K))
    
    return C_vec, z_A_z_flat, u_mat_flat, H_list, A_eq, Z_flat, M_total, K


def compress_distribution_dual_quad(
    x_bar: np.ndarray, 
    N: int, 
    K: int,
    rho: float, 
    R: float,
    C_vec: np.ndarray,
    z_A_z_flat: np.ndarray,
    u_mat_flat: np.ndarray,
    H_list: list,
    A_eq: sp.csr_matrix, 
    Z_flat: np.ndarray, 
    M_total: int
):
    """
    Solves the exact Dual Compression Program natively as an SOCP 
    using the MOSEK interior point solver.
    Uses LP purification via SOC linearization to guarantee minimal support.
    """
    d = len(x_bar)
    J = M_total * K
    
    # ==========================================================
    # Step 1: Solve the exact SOCP (L1 trust region)
    # ==========================================================
    with mf.Model("Dual_Compression") as M:
        
        # --- Variables ---
        alpha = M.variable("alpha", J, mf.Domain.greaterThan(0.0))
        lam = M.variable("lam", [J, d], mf.Domain.unbounded())
        t = M.variable("t", J, mf.Domain.greaterThan(0.0))
        s = M.variable("s", 1, mf.Domain.greaterThan(0.0)) # Slack for infinity norm
        
        # 1. Wasserstein Budget: sum(C_vec * alpha) <= rho * N
        M.constraint("budget", mf.Expr.dot(C_vec.tolist(), alpha), mf.Domain.lessThan(rho * N))
        
        # 2. Simplex Marginals: A_eq * alpha == 1
        A_eq_coo = A_eq.tocoo()
        A_eq_mf = mf.Matrix.sparse(
            A_eq_coo.shape[0], A_eq_coo.shape[1], 
            A_eq_coo.row.tolist(), A_eq_coo.col.tolist(), A_eq_coo.data.tolist()
        )
        M.constraint("simplex", mf.Expr.mul(A_eq_mf, alpha), mf.Domain.equalsTo(np.ones(N).tolist()))
        
        # 3. Trust Penalty (Infinity Norm): -s <= (1/N)*sum(lam) <= s
        sum_lam = mf.Expr.mul(1.0 / N, mf.Expr.sum(lam, 0))
        s_vec = mf.Expr.repeat(s, d, 0)
        M.constraint("inf_norm_upper", mf.Expr.sub(sum_lam, s_vec), mf.Domain.lessThan(0.0))
        M.constraint("inf_norm_lower", mf.Expr.add(sum_lam, s_vec), mf.Domain.greaterThan(0.0))
        
        # 4. Native Rotated Second-Order Cone (RSOC) Constraints
        for k in range(K):
            H_k = H_list[k]
            u_k_mat = u_mat_flat[k::K, :]         
            H_u_k = u_k_mat @ H_k.T               
            
            # Extract variables specifically for component k
            idx = list(range(k, J, K))
            alpha_k = alpha.pick(idx)
            t_k = t.pick(idx)

            lam_coords = [[r, c] for r in idx for c in range(d)]
            lam_k_flat = lam.pick(lam_coords)
            lam_k = mf.Expr.reshape(lam_k_flat, M_total, d)
            
            # Build matrices for expressions
            H_k_mf = mf.Matrix.dense(H_k.T)
            H_u_k_mf = mf.Matrix.dense(H_u_k)
            
            # Calculate V_k = lam_k * H_k^T - diag(alpha_k) * H_u_k
            term1 = mf.Expr.mul(lam_k, H_k_mf)

            alpha_k_col = mf.Expr.reshape(alpha_k, M_total, 1)
            alpha_k_mat = mf.Expr.repeat(alpha_k_col, d, 1)
            term2 = mf.Expr.mulElm(alpha_k_mat, H_u_k_mf)

            V_k = mf.Expr.sub(term1, term2)
            
            # Map into MOSEK's Rotated Cone Domain: 2 * t * alpha >= ||sqrt(2) * V_k||^2
            sqrt2_V_k = mf.Expr.mul(np.sqrt(2.0), V_k)
            
            # alpha_k_col = mf.Expr.reshape(alpha_k, M_total, 1)
            t_k_col = mf.Expr.reshape(t_k, M_total, 1)
            
            stack = mf.Expr.hstack([t_k_col, alpha_k_col, sqrt2_V_k])
            M.constraint(f"rsoc_{k}", stack, mf.Domain.inRotatedQCone().axis(1))
            
        # --- Objective ---
        z_A_z_dot_alpha = mf.Expr.dot(z_A_z_flat.tolist(), alpha)
        sum_t = mf.Expr.sum(t)
        conj_loss = mf.Expr.mul(-1.0 / N, mf.Expr.add(sum_t, z_A_z_dot_alpha))
        obj = mf.Expr.sub(conj_loss, mf.Expr.mul(R, s))
        
        M.objective(mf.ObjectiveSense.Maximize, obj)
        
        # --- Solve ---
        M.setSolverParam("intpntCoTolRelGap", 1e-6)
        M.setSolverParam("numThreads", 0)
        M.solve()
        
        status = M.getProblemStatus()
        if status not in [mf.ProblemStatus.PrimalAndDualFeasible, mf.ProblemStatus.PrimalFeasible]:
            raise RuntimeError(f"MOSEK Fusion solver failed. Status: {status}")
            
        # Extract values
        alpha_opt = np.array(alpha.level())
        lam_opt = np.array(lam.level()).reshape((J, d))
        t_opt = np.array(t.level())
        prob_value = M.primalObjValue()

    # ==========================================================
    # Step 2: LP Purification (Extracting the BFS via Gurobi)
    # ==========================================================
    
    # 1. Filter the stable, active variables from the SOCP
    threshold = 1e-8
    active_idx = np.where(alpha_opt > threshold)[0]
    
    if len(active_idx) > 0:
        alpha_act = alpha_opt[active_idx]
        lam_act = lam_opt[active_idx]
        t_act = t_opt[active_idx]
        
        # 2. Compute exact local gradients and costs for the active subset
        y_act = lam_act / alpha_act[:, None]                                
        c_act = -(t_act / alpha_act) - z_A_z_flat[active_idx]               
        
        # Calculate target aggregate gradient to perfectly lock the trust penalty
        theta_target = (1.0 / N) * np.sum(lam_act, axis=0)                  
        
        # Extract subset matrices
        C_act = C_vec[active_idx]
        A_eq_act = A_eq[:, active_idx]
        
        # 3. Build the microscopic LP natively in Gurobi
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("FeasibilityTol", 1e-6) 
        env.setParam("OptimalityTol", 1e-6)
        env.setParam("NumericFocus", 3)      
        env.start()
        
        m = gp.Model("LP_Purification", env=env)
        
        alpha_lp = m.addMVar(shape=len(active_idx), lb=0.0, name="alpha_lp")
        
        m.addConstr(A_eq_act @ alpha_lp == np.ones(N), name="simplex")

        m.addConstr((1.0 / N) * C_act @ alpha_lp <= rho, name="budget")

        tol = 1e-8
        expected_grad = (1.0 / N) * (y_act.T @ alpha_lp)
        m.addConstr(expected_grad <= theta_target + tol, name="eq_upper")
        m.addConstr(expected_grad >= theta_target - tol, name="eq_lower")
        
        m.setObjective((1.0 / N) * c_act @ alpha_lp, GRB.MAXIMIZE)
        
        m.setParam("Method", 1) 
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            final_alpha = np.zeros(J)
            final_alpha[active_idx] = alpha_lp.X

            # --- Calculate the refined optimal value ---
            # LP Objective represents the Conjugate Loss
            lp_conj_loss = m.ObjVal
            
            # Recalculate the trust penalty using the purified gradient
            theta_new = (1.0 / N) * (y_act.T @ alpha_lp.X)
            lp_trust_penalty = -R * np.linalg.norm(theta_new, ord=np.inf)
            
            # Override the SOCP prob_value with the exact purified value
            prob_value = lp_conj_loss + lp_trust_penalty
        else:
            print(f"[Warning] Gurobi LP Purification failed (status {m.Status}). Using SOCP interior point.")
            final_alpha = alpha_opt
    else:
        print("[Warning] No active atoms found in SOCP solution. Using SOCP interior point.")
        final_alpha = alpha_opt
        
    # --- Extract Distribution ---
    alpha_reshaped = final_alpha.reshape((M_total, K))
    marginal_atom_weights = np.sum(alpha_reshaped, axis=1) / N
    
    tolerance = 1e-8
    active_indices = np.where(marginal_atom_weights > tolerance)[0]
    
    compressed_weights = marginal_atom_weights[active_indices]
    compressed_atoms = Z_flat[active_indices]

    optimal_budgets = A_eq @ (final_alpha * C_vec)
    
    return InnerMaxResult(
        worst_case_loss=prob_value,
        optimal_lambda=[],                  # Omitted for the dual formulation
        optimal_budgets=optimal_budgets,
        worst_case_distribution=(compressed_weights, compressed_atoms),
        active_components=[]                # Omitted for the dual formulation
    )
import numpy as np
import mosek.fusion as mf
from wdro_inner.solver import InnerMaxResult


def solve_full_dual_dro_quad(
    z_empirical: np.ndarray,
    A_list: list,
    B_list: list,
    C_list: list,
    rho: float,
    R: float
):
    """
    Solves the Exact Full Dual DRO under quadratic loss and unsquared L2 cost.
    Enforces an L1-ball trust region X = {x : ||x||_1 <= R}.
    """
    N, d = z_empirical.shape
    K = len(A_list)

    with mf.Model("Full_Dual_DRO") as M:
        
        # --- Variables ---
        alpha = M.variable("alpha", [N, K], mf.Domain.greaterThan(0.0))
        Y = M.variable("Y", [N, K, d], mf.Domain.unbounded())
        Q = M.variable("Q", [N, K, d], mf.Domain.unbounded())
        r = M.variable("r", [N, K], mf.Domain.greaterThan(0.0))
        t1 = M.variable("t1", [N, K], mf.Domain.greaterThan(0.0))
        t2 = M.variable("t2", [N, K], mf.Domain.greaterThan(0.0))
        s = M.variable("s", 1, mf.Domain.greaterThan(0.0)) # Slack for infinity norm

        # 1. Simplex Constraint: sum_k alpha_{ik} = 1 for all i
        M.constraint("simplex", mf.Expr.sum(alpha, 1), mf.Domain.equalsTo(1.0))

        # 2. Wasserstein Budget Constraint: (1/N) * sum(r) <= rho
        M.constraint("budget", mf.Expr.mul(1.0 / N, mf.Expr.sum(r)), mf.Domain.lessThan(rho))
        
        # 3. Cost Norms: ||Q_{ik}||_2 <= r_{ik} via Quadratic Cones
        # MOSEK accepts quadratic cones: r_ik >= ||Q_ik||_2
        for i in range(N):
            for k in range(K):
                r_ik = r.index(i, k)
                Q_ik = Q.slice([i, k, 0], [i + 1, k + 1, d]).reshape(d)
                M.constraint(f"norm_Q_{i}_{k}", mf.Expr.vstack(r_ik, Q_ik), mf.Domain.inQCone())

        # 4. Trust Penalty (Infinity Norm): -s <= (1/N)*sum(Y) <= s
        sum_Y = mf.Expr.mul(1.0 / N, mf.Expr.sum(Y, [0, 1]))
        s_vec = mf.Expr.repeat(s, d, 0)
        M.constraint("inf_norm_upper", mf.Expr.sub(sum_Y, s_vec), mf.Domain.lessThan(0.0))
        M.constraint("inf_norm_lower", mf.Expr.add(sum_Y, s_vec), mf.Domain.greaterThan(0.0))

        # 5. Perspective Reformulation (Native Rotated Second-Order Cones)
        for k in range(K):
            # Precompute H_k = 0.5 * C_k^{-1/2}
            eigvals_C, eigvecs_C = np.linalg.eigh(C_list[k])
            eigvals_C_clipped = np.clip(eigvals_C, 1e-10, None)
            H_k = 0.5 * eigvecs_C @ np.diag(1.0 / np.sqrt(eigvals_C_clipped)) @ eigvecs_C.T
            
            # Precompute A_k^{1/2}
            eigvals_A, eigvecs_A = np.linalg.eigh(A_list[k])
            eigvals_A_clipped = np.clip(eigvals_A, 1e-10, None)
            A_half = eigvecs_A @ np.diag(np.sqrt(eigvals_A_clipped)) @ eigvecs_A.T

            H_k_mf = mf.Matrix.dense(H_k.T)
            A_half_mf = mf.Matrix.dense(A_half.T)
            z_emp_mf = mf.Matrix.dense(z_empirical)

            for i in range(N):
                alpha_ik = alpha.index(i, k)
                Q_ik = Q.slice([i, k, 0], [i + 1, k + 1, d]).reshape(d)
                Y_ik = Y.slice([i, k, 0], [i + 1, k + 1, d]).reshape(d)
                
                t1_ik = t1.index(i, k)
                t2_ik = t2.index(i, k)
                
                # Z_ik = alpha_ik * z_hat_i + Q_ik
                z_hat_i_list = z_empirical[i].tolist()
                z_hat_i_row = mf.Expr.mul(alpha_ik, z_hat_i_list)
                Z_ik = mf.Expr.add(z_hat_i_row, Q_ik)
                
                # V_k = (Y_ik - Z_ik @ B_k) @ H_k^T
                B_k_mf = mf.Matrix.dense(B_list[k])
                Z_B = mf.Expr.mul(Z_ik, B_k_mf)
                V_k = mf.Expr.mul(mf.Expr.sub(Y_ik, Z_B), H_k_mf)
                
                # W_k = Z_ik @ A_half^T
                W_k = mf.Expr.mul(Z_ik, A_half_mf)
                
                # --- RSOC 1: 2 * t1_ik * alpha_ik >= ||sqrt(2) * V_k||^2 ---
                sqrt2_V_k = mf.Expr.mul(np.sqrt(2.0), V_k)
                stack1 = mf.Expr.vstack([mf.Expr.reshape(t1_ik, 1), mf.Expr.reshape(alpha_ik, 1), sqrt2_V_k])
                M.constraint(f"rsoc1_{i}_{k}", stack1, mf.Domain.inRotatedQCone())
                
                # --- RSOC 2: 2 * t2_ik * alpha_ik >= ||sqrt(2) * W_k||^2 ---
                sqrt2_W_k = mf.Expr.mul(np.sqrt(2.0), W_k)
                stack2 = mf.Expr.vstack([mf.Expr.reshape(t2_ik, 1), mf.Expr.reshape(alpha_ik, 1), sqrt2_W_k])
                M.constraint(f"rsoc2_{i}_{k}", stack2, mf.Domain.inRotatedQCone())

        # --- Objective ---
        # Maximize conjugate loss - R * s
        sum_t1_t2 = mf.Expr.add(mf.Expr.sum(t1), mf.Expr.sum(t2))
        conjugate_loss = mf.Expr.mul(-1.0 / N, sum_t1_t2)
        trust_penalty = mf.Expr.mul(R, s)
        objective = mf.Expr.sub(conjugate_loss, trust_penalty)
        
        M.objective(mf.ObjectiveSense.Maximize, objective)
        
        # --- Solve ---
        M.setSolverParam("intpntCoTolRelGap", 1e-6)
        M.setSolverParam("numThreads", 0)
        M.solve()
        
        status = M.getProblemStatus()
        if status not in [mf.ProblemStatus.PrimalAndDualFeasible, mf.ProblemStatus.PrimalFeasible]:
            print(f"[Warning] MOSEK finished with status: {status}")

        # --- Extract Results ---
        alpha_val = np.array(alpha.level()).reshape((N, K))
        Q_val = np.array(Q.level()).reshape((N, K, d))
        opt_val = M.primalObjValue()

    # --- Extract Worst-Case Distribution ---
    weights = []
    supports = []
    optimal_budgets = np.zeros(N)
    
    tol = 1e-8
    for i in range(N):
        budget_i = 0.0
        for k in range(K):
            a_ik = alpha_val[i, k]
            if a_ik > tol:
                w = a_ik / N
                z_ik = z_empirical[i] + (Q_val[i, k, :] / a_ik)
                
                weights.append(w)
                supports.append(z_ik)
                
                # The exact physical budget utilized by this component
                budget_i += np.linalg.norm(Q_val[i, k, :])
                
        optimal_budgets[i] = budget_i
                
    weights = np.array(weights)
    supports = np.array(supports)
        
    return InnerMaxResult(
        worst_case_loss=opt_val,
        optimal_lambda=[],                  # Omitted for full dual formulation
        optimal_budgets=optimal_budgets,
        worst_case_distribution=(weights, supports),
        active_components=[]                # Omitted for full dual formulation
    )
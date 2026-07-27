import numpy as np
import os
import pickle
from datetime import datetime
from primal.dbr_solver_l2 import SaddlePointProbleml2, distributional_best_response
from primal.full_dual_l2 import solve_full_dual_dro_quad


if __name__ == "__main__":
    np.random.seed(2026)
    N, d = 50, 50
    R = 100.0  # trust region radius
    K = 3
    epsilon = 0.1

    z_mean_shift = np.random.randn(1, d)
    z_emp = np.random.randn(N, d) + z_mean_shift  # Shifted empirical samples
    
    A_list = []
    B_list = []
    C_list = []
    
    # Generate K random matrices with proper structural constraints
    for _ in range(K):
        # A_k must be Positive Semi-Definite
        X_A = np.random.randn(d, d)
        A_list.append((X_A.T @ X_A) / d + 0.01 * np.eye(d)) # Add strong convexity to A for stability
        
        # B_k is a general coupling matrix
        component_bias = np.random.randn(d, d)
        B_list.append(np.random.randn(d, d) + component_bias)
        
        # C_k must be Positive Semi-Definite
        X_C = np.random.randn(d, d)
        C_list.append((X_C.T @ X_C) / d + 0.01 * np.eye(d)) # Add strong convexity to C for stability
    
    problem = SaddlePointProbleml2(A_list, B_list, C_list, d, R)
    
    x_init = np.zeros(d)
    lr = lambda t: 0.2 / np.sqrt(t)  # Decaying learning rate
    T = 50

    result_full_dual = solve_full_dual_dro_quad(z_emp, A_list, B_list, C_list, epsilon, R)
    dual_weights, dual_supports = result_full_dual.worst_case_distribution
    dual_opt_val = result_full_dual.worst_case_loss
    print(f"Dual optimal value: {dual_opt_val}")
    
    print("Starting Distributional Best-Response...")
    x_opt, (Q_weights, Q_supports), history = distributional_best_response(z_emp, problem, x_init, lr, T, epsilon, dual_optimal=dual_opt_val)
    print("\nOptimization Complete!")

    # Save the history dictionary to a file
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"dbr_history_{timestamp}.pkl")

    with open(filename, 'wb') as f:
        pickle.dump(history, f)
        
    print(f"Experimental history successfully saved to '{filename}'")
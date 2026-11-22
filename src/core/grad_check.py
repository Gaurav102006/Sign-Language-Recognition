"""
Unit II: Numerical Gradient Checking (Finite Differences).
Validates analytical backpropagation gradients against two-sided finite difference approximations.
Aligned to CSAI3027P Deep Learning - Project 13.
"""

from typing import Dict, Tuple
import numpy as np
from src.core.mlp import ScratchMLP


class GradientChecker:
    """
    Two-sided Finite Difference Numerical Gradient Checker:
    grad_num(theta_i) = [ J(theta + eps * e_i) - J(theta - eps * e_i) ] / (2 * eps)
    
    Relative error formula:
    rel_error = ||grad_num - grad_ana||_2 / ( ||grad_num||_2 + ||grad_ana||_2 + 1e-15 )
    
    Verification criteria:
    - rel_error < 1e-7: Excellent match (analytical gradient is mathematically correct).
    - 1e-7 <= rel_error < 1e-5: Acceptable (small floating point precision differences).
    - rel_error >= 1e-4: Potential bug in backpropagation equations.
    """

    @staticmethod
    def check_model_gradients(
        model: ScratchMLP,
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5,
        max_params_to_check: int = 40,
    ) -> Dict:
        """
        Runs gradient check on a subset or full set of model parameters.
        
        Args:
            model: ScratchMLP instance.
            x: Mini-batch input (e.g. shape (8, 126)).
            y: Mini-batch labels (e.g. shape (8,) or (8, n_classes)).
            epsilon: Perturbation step size.
            max_params_to_check: To keep verification fast, checks first N parameters.
            
        Returns:
            Dictionary with relative error, maximum absolute difference, and status.
        """
        # 1. Compute analytical gradients via forward + backward
        y_pred = model.forward(x)
        model.backward(y_pred, y)
        grad_analytical = model.get_grads_flat()

        # 2. Get flattened parameter vector
        theta = model.get_params_flat()
        n_params = len(theta)
        check_count = min(n_params, max_params_to_check)

        grad_numerical = np.zeros(check_count, dtype=np.float64)
        grad_ana_sub = grad_analytical[:check_count]

        # 3. Symmetric perturbation loop
        for i in range(check_count):
            theta_orig = theta[i]

            # J(theta + eps)
            theta[i] = theta_orig + epsilon
            model.set_params_flat(theta)
            pred_plus = model.forward(x)
            loss_plus = model.compute_loss(pred_plus, y)

            # J(theta - eps)
            theta[i] = theta_orig - epsilon
            model.set_params_flat(theta)
            pred_minus = model.forward(x)
            loss_minus = model.compute_loss(pred_minus, y)

            # Numerical gradient
            grad_numerical[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

            # Restore original parameter
            theta[i] = theta_orig

        model.set_params_flat(theta)  # Ensure fully restored

        # 4. Compute relative error norm
        diff_norm = np.linalg.norm(grad_numerical - grad_ana_sub)
        sum_norm = np.linalg.norm(grad_numerical) + np.linalg.norm(grad_ana_sub)
        relative_error = float(diff_norm / (sum_norm + 1e-15))
        max_abs_diff = float(np.max(np.abs(grad_numerical - grad_ana_sub)))

        passed = relative_error < 1e-5

        return {
            "total_parameters": n_params,
            "checked_parameters": check_count,
            "relative_error": relative_error,
            "max_abs_diff": max_abs_diff,
            "passed": passed,
            "status": "PASSED (Gradients Verified)" if passed else "FAILED (Gradient Mismatch)",
            "sample_analytical": grad_ana_sub[:5].tolist(),
            "sample_numerical": grad_numerical[:5].tolist(),
        }

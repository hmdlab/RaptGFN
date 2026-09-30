"""Internal metrics used by training and generation."""

import numpy as np

try:
    from pymoo.factory import get_performance_indicator

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

try:
    import pyhv as _pyhv

    PYHV_AVAILABLE = True
except Exception:
    PYHV_AVAILABLE = False

from raptgfn.algorithms.mogfn_utils import compute_hypervolume


def get_all_metrics(solutions, eval_metrics, **kwargs):
    """Compute the requested metrics for non-dominated objective scores."""
    metrics = {}

    if "hypervolume" in eval_metrics:
        if PYMOO_AVAILABLE and "hv_ref" in kwargs:
            try:
                hv_indicator = get_performance_indicator("hv", ref_point=kwargs["hv_ref"])
                metrics["hypervolume"] = hv_indicator.do(-solutions)
            except:
                # Try pyhv before the local fallback.
                if PYHV_AVAILABLE and "hv_ref" in kwargs:
                    try:
                        ref = kwargs["hv_ref"]
                        metrics["hypervolume"] = float(_pyhv.hypervolume(solutions, ref))
                    except:
                        metrics["hypervolume"] = compute_hypervolume(
                            solutions, kwargs.get("hv_ref")
                        )
                else:
                    metrics["hypervolume"] = compute_hypervolume(solutions, kwargs.get("hv_ref"))
        else:
            # Try pyhv before the local fallback.
            if PYHV_AVAILABLE and "hv_ref" in kwargs:
                try:
                    ref = kwargs["hv_ref"]
                    metrics["hypervolume"] = float(_pyhv.hypervolume(solutions, ref))
                except:
                    metrics["hypervolume"] = compute_hypervolume(solutions, kwargs.get("hv_ref"))
            else:
                metrics["hypervolume"] = compute_hypervolume(solutions, kwargs.get("hv_ref"))
        # Flag the possible higher-dimensional volume-sum approximation.
        if solutions.shape[1] > 3 and not PYMOO_AVAILABLE and not PYHV_AVAILABLE:
            metrics["hypervolume_is_approx"] = True

    if "r2" in eval_metrics and "num_obj" in kwargs:
        try:
            r2_prefs = kwargs.get("r2_prefs", None)
            if r2_prefs is None:
                # Generate preference directions on the simplex.
                from raptgfn.algorithms.mogfn_utils import generate_simplex

                r2_prefs = generate_simplex(kwargs["num_obj"], 10)
            metrics["r2"] = r2_indicator_set(r2_prefs, solutions, np.ones(kwargs["num_obj"]))
        except:
            metrics["r2"] = 0.0

    if "hsri" in eval_metrics and "num_obj" in kwargs:
        try:
            metrics["hsri"] = calculate_simple_hsr(solutions, kwargs["num_obj"])
        except:
            metrics["hsri"] = 0.0

    if "diversity" in eval_metrics:
        metrics["diversity"] = calculate_diversity(solutions)

    if "coverage" in eval_metrics:
        metrics["coverage"] = calculate_objective_space_coverage(solutions)

    return metrics


def r2_indicator_set(preferences, solutions, weights):
    """Compute the local R2 indicator approximation."""
    if len(solutions) == 0:
        return 0.0

    # Approximate the R2 indicator over preference directions.
    r2_values = []
    for pref in preferences:
        weighted_solutions = solutions * pref
        max_vals = np.max(weighted_solutions, axis=0)
        r2_values.append(np.sum(max_vals))

    return np.mean(r2_values)


def calculate_simple_hsr(solutions, num_obj):
    """Compute the simple hypervolume-based success-rate indicator."""
    if len(solutions) == 0:
        return 0.0

    # Compute the range of each objective.
    obj_ranges = np.max(solutions, axis=0) - np.min(solutions, axis=0)

    # Multiply the normalized objective ranges.
    normalized_ranges = obj_ranges / (np.max(obj_ranges) + 1e-8)
    hsr = np.prod(normalized_ranges)

    return float(hsr)


def calculate_diversity(solutions):
    """Compute diversity in objective space."""
    if len(solutions) <= 1:
        return 0.0

    # Average pairwise Euclidean distances.
    distances = []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            dist = np.linalg.norm(solutions[i] - solutions[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


def calculate_objective_space_coverage(solutions):
    """Compute objective-space coverage."""
    if len(solutions) == 0:
        return 0.0

    # Compute the range of each objective.
    ranges = np.max(solutions, axis=0) - np.min(solutions, axis=0)

    # Normalize the coverage measure.
    normalized_ranges = ranges / (1.0 + ranges)  # Map each nonnegative range into the unit interval.

    return np.mean(normalized_ranges)


def calculate_hypervolume_improvement(current_solutions, previous_solutions, reference_point=None):
    """Compute the change in the internal hypervolume metric."""
    if reference_point is None:
        all_solutions = (
            np.vstack([current_solutions, previous_solutions])
            if len(previous_solutions) > 0
            else current_solutions
        )
        reference_point = np.min(all_solutions, axis=0) - 0.1

    current_hv = compute_hypervolume(current_solutions, reference_point)
    previous_hv = (
        compute_hypervolume(previous_solutions, reference_point)
        if len(previous_solutions) > 0
        else 0.0
    )

    return current_hv - previous_hv

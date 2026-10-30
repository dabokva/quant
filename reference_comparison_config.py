CONFIG = {
    "data_root": "data",
    "studies": {
        # "r05": "experiments/sycamore_benchmark_r05",
        # "r09": "experiments/sycamore_benchmark_r09",
        "r25": "experiments/sycamore_benchmark_r25",
    },
    "output_root": "reference_comparisons",
    "references": {
        "correlated_matching": "obs_flips_predicted/dem_correlations/correlated_matching.b8",
        # "belief_matching": "obs_flips_predicted/dem_correlations/belief_matching.b8",
        # "harmony": "obs_flips_predicted/dem_correlations/harmony.b8",
    },
    "reference_format": "b8",
    "models": None,
    "thresholds": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
                   0.97, 0.98, 0.985, 0.99, 0.995, 1.0],
    "alpha": 0.05,
    "calibration_backend": "numpy",
    "verify_artifact_hashes": True,
    "plots": True,
}

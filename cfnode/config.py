"""
===============================================================
CF-NODE Configuration
This file contains all configurable hyperparameters used by the
CF-NODE algorithm.
These parameters control the behaviour of counterfactual node
editing, causal evidence estimation, adaptive prediction
refinement, and source prior correction.
Users typically only need to modify this file to experiment
with different CF-NODE settings.
For implementation details, see:
    cfnode/cfnode.py
===============================================================
"""
class CFNodeConfig:
    """
    Configuration for the CF-NODE algorithm.
    """
    # ==========================================================
    # TARGET DATASET
    # ==========================================================
    #
    # Used only when different parameter values are desired for
    # different target domains.
    #
    # This value can be overwritten dynamically in your own
    # evaluation script.
    #
    target_dataset = "messidor"
    target_name = "messidor"
    # ==========================================================
    # ENABLE / DISABLE COMPONENTS
    # ==========================================================
    use_cf_adaptation = True
    use_prior_correction = True
    use_aggressive_cf = False
    skip_cf_for_deepdrid = True
    # ==========================================================
    # SOURCE DOMAIN PRIOR
    # ==========================================================
    #
    # EyePACS empirical class distribution used for source
    # prior correction.
    #
    source_prior = [
        0.734783,
        0.069550,
        0.150658,
        0.024853,
        0.020156,
    ]
    # ==========================================================
    # SOURCE PRIOR CORRECTION
    # ==========================================================
    prior_correction_strength = 0.18
    # ==========================================================
    # BASE SOFTMAX TEMPERATURE
    # ==========================================================
    temp = 1.05
    # ==========================================================
    # CAUSAL GATE THRESHOLD
    # ==========================================================
    sensitivity_threshold = 0.12
    # ==========================================================
    # CONFIDENCE THRESHOLD
    # ==========================================================
    high_conf = 0.84
    # ==========================================================
    # COUNTERFACTUAL SUPPRESSION LEVELS
    # ==========================================================
    #
    # None uses the default suppression schedule implemented
    # inside CF-NODE.
    #
    cf_levels = None
    # ==========================================================
    # PER-DATASET CF STRENGTH
    # ==========================================================
    cf_strength = {
        "messidor": 1.0,
        "idrid": 1.0,
        "ddr": 1.0,
        "aptos": 1.0,
        "deepdrid": 1.0,
        "default": 1.0,
    }
    # ==========================================================
    # PREDICTION MODE
    # ==========================================================
    #
    # Available options:
    #
    # "argmax"
    # "expected"
    # "hybrid"
    #
    predict_mode = "hybrid"
    # ==========================================================
    # LOGIT ADJUSTMENT
    # ==========================================================
    pred_logit_adjust = 0.30
    # ==========================================================
    # LEGACY PARAMETERS
    # ==========================================================
    #
    # Retained for compatibility with the original implementation.
    #
    prior_strength = 0.03
    max_cf_alpha = 0.015
    tta_steps = 1
    tta_lr = 2e-5
    margin = 0.12
    lambda_tta = 0.35
    noise_weight = 0.15
    w_floor = 0.55
    aug_weight = 0.20
    entropy_weight = 0.003
    use_blur = False
    use_entropy = False
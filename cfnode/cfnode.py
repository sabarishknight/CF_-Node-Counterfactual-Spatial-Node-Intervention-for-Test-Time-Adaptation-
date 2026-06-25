"""
======================================================================
CF-NODE
Counterfactual Lesion Node Editing for Test-Time Adaptation

Official implementation accompanying the paper:

"Counterfactual Lesion Node Editing for Test-Time Adaptation
Under Domain Shift in Diabetic Retinopathy Grading"

----------------------------------------------------------------------

Purpose
-------
This module implements the complete CF-NODE algorithm for
source-free and label-free Test-Time Adaptation (TTA).

CF-NODE improves the robustness of diabetic retinopathy
grading models under domain shift by performing
counterfactual lesion node editing directly on intermediate
feature representations.

----------------------------------------------------------------------

Plug-and-Play Integration
-------------------------
CF-NODE is designed to be integrated into existing
PyTorch-based diabetic retinopathy grading models.

Your model should provide:

    • Classification logits
    • Lesion node features
    • Ordinal logits (optional but recommended)

A complete compatible implementation is provided in:

    model_example/model.py

----------------------------------------------------------------------

Main Functions
--------------

adapt_batch()
    Perform CF-NODE adaptation on a batch of images.

run_tta()
    Run CF-NODE over an entire test dataloader.

======================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np

# =========================================================
# DEFAULT SOURCE PRIOR
# EyePACS empirical prior — the working v1 configuration.
# =========================================================
DEFAULT_SOURCE_PRIOR = torch.tensor([
    0.73,
    0.07,
    0.15,
    0.025,
    0.025,
]).float()

# =========================================================
# COUNTERFACTUAL SUPPRESSION SCHEDULE
# =========================================================
DEFAULT_CF_LEVELS = (

    (0.02, 0.75),

    (0.05, 0.40),

    (0.10, 0.20),

    (0.15, 0.10),
)

# =========================================================
# AUGMENTATIONS
# =========================================================
AUGMENTATIONS = [

    lambda x: x,

    lambda x: torch.flip(x, dims=[3]),

    lambda x: torch.flip(x, dims=[2]),

    lambda x: torch.rot90(x, k=1, dims=[2, 3]),

    lambda x: torch.rot90(x, k=3, dims=[2, 3]),
]

# =========================================================
# CFG ACCESSOR
# =========================================================
def _cfg(cfg, key, default):

    return getattr(cfg, key, default)

# =========================================================
# RESOLVE CF STRENGTH
#
# Default 1.0 → v1 behaviour preserved.
# =========================================================
def _resolve_cf_strength(cfg):

    cf = _cfg(cfg, 'cf_strength', 1.0)

    if isinstance(cf, dict):

        name = (
            _cfg(cfg, 'target_name',    None)
            or _cfg(cfg, 'target_dataset', None)
            or _cfg(cfg, 'dataset',     None)
            or _cfg(cfg, 'target',      None)
            or _cfg(cfg, 'domain',      None)
        )

        if isinstance(name, str):

            name = name.lower()

        if name is not None and name in cf:

            cf = cf[name]

        elif 'default' in cf:

            cf = cf['default']

        else:

            cf = 1.0

    return float(cf)

# =========================================================
# RESOLVE SOURCE PRIOR
# =========================================================
def _resolve_source_prior(cfg, device):

    prior = _cfg(cfg, 'source_prior', None)

    if prior is None:

        prior = DEFAULT_SOURCE_PRIOR

    if not torch.is_tensor(prior):

        prior = torch.tensor(
            prior,
            dtype=torch.float32
        )

    prior = prior.float().to(device)

    return prior / (prior.sum() + 1e-8)

# =========================================================
# RESOLVE CF LEVELS
# =========================================================
def _resolve_cf_levels(cfg):

    levels = _cfg(cfg, 'cf_levels', None)

    if levels is None or len(levels) == 0:

        return DEFAULT_CF_LEVELS

    return levels

# =========================================================
# SAFE MODEL UNWRAP
# =========================================================
def get_base_model(model):

    return (
        model.module
        if isinstance(model, torch.nn.DataParallel)
        else model
    )

# =========================================================
# FORWARD WITH NODES
# =========================================================
def forward_with_nodes(model, x):

    out = model(x, return_nodes=True)

    if isinstance(out, tuple):

        if len(out) == 3:

            logits, nodes, ordinal_logits = out
            pooled = None

        elif len(out) == 4:

            logits, nodes, ordinal_logits, pooled = out

        else:

            logits = out[0]
            nodes = None
            ordinal_logits = None
            pooled = None

    else:

        logits = out
        nodes = None
        ordinal_logits = None
        pooled = None

    return logits, nodes, ordinal_logits, pooled

# =========================================================
# EXPECTED GRADE
# =========================================================
def expected_grade_from_probs(probs):

    grades = torch.arange(
        probs.size(-1),
        device=probs.device
    ).float()

    return (probs * grades).sum(dim=-1)

# =========================================================
# ORDINAL → CLASS PROBS
# =========================================================
def ordinal_logits_to_probs(ordinal_logits, num_classes=5):

    B      = ordinal_logits.size(0)
    device = ordinal_logits.device

    cum_probs = torch.sigmoid(ordinal_logits)

    ones  = torch.ones(B, 1, device=device)
    zeros = torch.zeros(B, 1, device=device)

    cum_full = torch.cat(
        [ones, cum_probs, zeros],
        dim=1
    )

    class_probs = (
        cum_full[:, :-1]
        - cum_full[:, 1:]
    )

    class_probs = torch.clamp(
        class_probs,
        min=1e-6
    )

    class_probs = class_probs / (
        class_probs.sum(dim=-1, keepdim=True)
        + 1e-8
    )

    return class_probs

# =========================================================
# ATTENTION SCORES
# =========================================================
def get_attention_scores(model, nodes):

    base = get_base_model(model)

    scores = base.attention_pool.attn(
        nodes
    ).squeeze(-1)

    return torch.softmax(scores, dim=1)

# =========================================================
# CLUSTER CONSISTENCY
# =========================================================
def compute_cluster_consistency(nodes):

    nodes_norm = F.normalize(nodes, dim=-1)

    sim = torch.bmm(
        nodes_norm,
        nodes_norm.transpose(1, 2)
    ).mean(dim=-1)

    sim_min = sim.min(
        dim=1,
        keepdim=True
    )[0]

    sim_max = sim.max(
        dim=1,
        keepdim=True
    )[0]

    return (
        (sim - sim_min)
        / (sim_max - sim_min + 1e-8)
    )

# =========================================================
#  ESTIMATE LESION NODE IMPORTANCE
#
# Purpose
# -------
# Computes the importance score for every lesion node by
# combining:
#
# • Attention score
# • Feature magnitude
# • Cluster consistency
#
# Higher scores indicate stronger influence on the current
# prediction.
# =========================================================
def compute_node_scores(attn_scores, nodes):

    strength = torch.norm(nodes, dim=-1)

    strength = strength / (
        strength.max(dim=1, keepdim=True)[0]
        + 1e-8
    )

    cluster = compute_cluster_consistency(nodes)

    return (
        0.50 * attn_scores
        + 0.25 * strength
        + 0.25 * cluster
    )

# =========================================================
# BUILD COUNTERFACTUAL LESION NODES
#
# Purpose
# -------
# Creates multiple counterfactual feature representations
# by progressively suppressing the most important lesion
# nodes identified in the current image.
#
# Input
# -----
# nodes         : (B, N, D)
# node_scores   : (B, N)
#
# Output
# ------
# cf_nodes_all  : Counterfactual node representations
# num_cf        : Number of counterfactual views
# =========================================================
def build_counterfactual_nodes(
    nodes,
    node_scores,
    cf_levels=DEFAULT_CF_LEVELS
):

    if cf_levels is None or len(cf_levels) == 0:

        cf_levels = DEFAULT_CF_LEVELS

    B, N, D = nodes.shape

    idx = torch.argsort(
        node_scores,
        dim=1,
        descending=True
    )

    cf_nodes_all = []

    for ratio, suppression in cf_levels:

        k = max(int(N * ratio), 1)

        top_idx = idx[:, :k]

        cf_nodes = nodes.clone()

        scaling = torch.ones_like(cf_nodes)

        scaling.scatter_(
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, D),
            suppression
        )

        cf_nodes_all.append(
            cf_nodes * scaling
        )

    return torch.cat(cf_nodes_all, dim=0), len(cf_levels)

# =========================================================
# CLASSIFY CF NODES
# =========================================================
def classify_cf_nodes(model, cf_nodes, global_feat):

    base = get_base_model(model)

    # =====================================================
    # NODE POOL
    # =====================================================
    pooled_nodes = base.attention_pool(cf_nodes)

    # =====================================================
    # REBUILD FUSED FEATURE
    # =====================================================
    fused = torch.cat(
        [
            pooled_nodes,
            global_feat
        ],
        dim=1
    )

    # =====================================================
    # FUSION BLOCK
    # =====================================================
    fused = base.fusion_block(fused)

    # =====================================================
    # DROPOUT
    # =====================================================
    fused = base.feature_dropout(fused)

    # =====================================================
    # CLASSIFIER
    # =====================================================
    logits = base.classifier(fused)

    return logits

# =========================================================
# STABLE CAUSAL DROP
# =========================================================
def compute_stable_causal_drop(exp_orig, exp_cf):

    drops = F.relu(
        exp_orig.unsqueeze(0) - exp_cf
    )

    mean_drop = drops.mean(dim=0)

    std_drop = drops.std(dim=0)

    return mean_drop / (std_drop + 0.15)

# =========================================================
# BLEND WEIGHT  (v1 formula — DO NOT CHANGE)
# =========================================================
def compute_blend_weight(stable_drop, confidence, cfg):

    sensitivity = _cfg(
        cfg,
        'sensitivity_threshold',
        0.08
    )

    cf_strength = _resolve_cf_strength(cfg)

    causal_gate = torch.sigmoid(
        5.0 * (stable_drop - sensitivity)
    )

    uncertainty = 1.0 - confidence

    alpha = cf_strength * (
        0.05 * causal_gate
        + 0.55 * causal_gate * uncertainty
    )

    return torch.clamp(alpha, 0.0, 0.65)

# =========================================================
# SOURCE PRIOR CORRECTION  (v1 formula — DO NOT CHANGE)
# =========================================================
def apply_source_prior_correction(probs, cfg):

    prior = _resolve_source_prior(
        cfg,
        probs.device
    )

    strength = _cfg(
        cfg,
        'prior_correction_strength',
        0.18
    )

    logits = torch.log(probs + 1e-8)

    logits = logits + (
        strength * torch.log(prior + 1e-8)
    )

    return F.softmax(logits, dim=-1)

# =========================================================
# SAFE NORMALIZE
# =========================================================
def safe_normalize(probs):

    probs = torch.nan_to_num(probs)

    probs = torch.clamp(
        probs,
        min=1e-6
    )

    probs = probs / (
        probs.sum(dim=-1, keepdim=True)
        + 1e-8
    )

    return probs

# =========================================================
# SINGLE VIEW
# =========================================================
def _cf_node_single_view(model, images, cfg):

    base_temp = _cfg(cfg, 'temp', 1.05)

    B = images.size(0)

    base = get_base_model(model)

    # =====================================================
    # ORIGINAL FORWARD
    # =====================================================
    logits_orig, nodes, ordinal_logits, _ = forward_with_nodes(
        model,
        images
    )

    probs_orig = F.softmax(
        logits_orig / base_temp,
        dim=-1
    )

    exp_orig = expected_grade_from_probs(
        probs_orig
    )

    confidence = probs_orig.max(
        dim=1
    ).values

    num_classes = probs_orig.size(-1)

    # =====================================================
    # FALLBACK
    # =====================================================
    if nodes is None:

        return probs_orig

    # =====================================================
    # RECOMPUTE GLOBAL FEATURES
    # =====================================================
    feat = base.features(images)

    global_feat = base.gem_pool(feat)

    global_feat = base.global_proj(global_feat)

    # =====================================================
    # NODE IMPORTANCE
    # =====================================================
    attn_scores = get_attention_scores(
        model,
        nodes
    )

    node_scores = compute_node_scores(
        attn_scores,
        nodes
    )

    # =====================================================
    # BUILD CF NODES
    # =====================================================
    cf_levels = _resolve_cf_levels(cfg)

    cf_nodes_all, num_cf = build_counterfactual_nodes(
        nodes,
        node_scores,
        cf_levels=cf_levels
    )

    # =====================================================
    # REPEAT GLOBAL FEATURES
    # =====================================================
    global_feat_repeat = global_feat.repeat(
        num_cf,
        1
    )

    # =====================================================
    # CF CLASSIFICATION
    # =====================================================
    logits_cf_all = classify_cf_nodes(
        model,
        cf_nodes_all,
        global_feat_repeat
    )

    probs_cf_all = F.softmax(
        logits_cf_all / base_temp,
        dim=-1
    )

    exp_cf = expected_grade_from_probs(
        probs_cf_all
    ).view(num_cf, B)

    # =====================================================
    # STABLE DROP
    # =====================================================
    stable_drop = compute_stable_causal_drop(
        exp_orig,
        exp_cf
    )

    # =====================================================
    # CAUSAL TEMPERATURE
    # =====================================================
    sample_temp = (
        base_temp
        - 0.22 * stable_drop
    )

    sample_temp = torch.clamp(
        sample_temp,
        min=0.72,
        max=1.08
    )

    # =====================================================
    # ORDINAL BLEND
    # =====================================================
    if ordinal_logits is not None:

        ordinal_probs = ordinal_logits_to_probs(
            ordinal_logits,
            num_classes=num_classes
        )

    else:

        ordinal_probs = probs_orig

    # =====================================================
    # BLEND
    # =====================================================
    alpha = compute_blend_weight(
        stable_drop,
        confidence,
        cfg
    ).unsqueeze(-1)

    blended_probs = (
        (1.0 - alpha) * probs_orig
        + alpha * ordinal_probs
    )

    # =====================================================
    # TEMPERATURE SHARPEN
    # =====================================================
    final_logits = (
        torch.log(blended_probs + 1e-8)
        / sample_temp.unsqueeze(-1)
    )

    return F.softmax(final_logits, dim=-1)

# =========================================================
# ADAPT BATCH  (argmax — v1 behaviour)
# =========================================================
@torch.no_grad()
def adapt_batch(model, images, cfg, return_probs=False):

    model.eval()

    view_probs_list = []

    for aug_fn in AUGMENTATIONS:

        view_probs = _cf_node_single_view(
            model,
            aug_fn(images),
            cfg
        )

        view_probs_list.append(view_probs)

    avg_probs = torch.stack(
        view_probs_list,
        dim=0
    ).mean(dim=0)

    # =====================================================
    # SOURCE PRIOR CORRECTION
    # =====================================================
    final_probs = apply_source_prior_correction(
        avg_probs,
        cfg
    )

    final_probs = safe_normalize(final_probs)

    # =====================================================
    # ARGMAX (v1)
    # =====================================================
    final_preds = torch.argmax(
        final_probs,
        dim=-1
    )

    # =====================================================
    # RANKING SCORE  (posterior expected grade)
    # =====================================================
    ranking_scores = expected_grade_from_probs(
        final_probs
    )

    preds_np = final_preds.cpu().numpy().astype(np.int64)

    probs_np = final_probs.cpu().numpy().astype(np.float32)

    ranking_np = ranking_scores.cpu().numpy().astype(np.float32)

    if return_probs:

        return (
            preds_np,
            probs_np,
            ranking_np
        )

    return preds_np

# =========================================================
# RUN TTA
# =========================================================
def run_tta(model, loader, cfg, return_probs=False):

    preds_all = []

    probs_all = []

    ranking_all = []

    labels_all = []

    device = next(model.parameters()).device

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        if return_probs:

            preds, probs, ranking = adapt_batch(
                model,
                images,
                cfg,
                return_probs=True
            )

            probs_all.append(probs)

            ranking_all.append(ranking)

        else:

            preds = adapt_batch(
                model,
                images,
                cfg
            )

        preds_all.extend(preds)

        labels_all.extend(labels.numpy())

    preds_all = np.array(
        preds_all,
        dtype=np.int64
    )

    labels_all = np.array(
        labels_all,
        dtype=np.int64
    )

    if return_probs:

        probs_all = np.concatenate(
            probs_all,
            axis=0
        ).astype(np.float32)

        ranking_all = np.concatenate(
            ranking_all,
            axis=0
        ).astype(np.float32)

        return (
            preds_all,
            labels_all,
            probs_all,
            ranking_all
        )

    return preds_all, labels_all



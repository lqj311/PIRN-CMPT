import math

import torch
import torch.nn.functional as F


def sample_tokens(tokens, max_tokens=200000):
    if tokens.shape[0] <= max_tokens:
        return tokens
    idx = torch.linspace(0, tokens.shape[0] - 1, steps=max_tokens, device=tokens.device).long()
    return tokens[idx]


def kmeans_prototypes(tokens, num_prototypes, iters=20, max_tokens=200000):
    tokens = sample_tokens(tokens.detach(), max_tokens=max_tokens)
    tokens = F.normalize(tokens, dim=-1)
    n = tokens.shape[0]
    if n == 0:
        raise ValueError('Cannot build prototypes from empty token set.')

    if n < num_prototypes:
        repeat = math.ceil(num_prototypes / n)
        centers = tokens.repeat(repeat, 1)[:num_prototypes].clone()
    else:
        idx = torch.linspace(0, n - 1, steps=num_prototypes, device=tokens.device).long()
        centers = tokens[idx].clone()

    for _ in range(iters):
        sim = tokens @ centers.t()
        assign = sim.argmax(dim=1)
        new_centers = []
        for k in range(num_prototypes):
            mask = assign == k
            if mask.any():
                new_centers.append(tokens[mask].mean(dim=0))
            else:
                new_centers.append(centers[k])
        centers = F.normalize(torch.stack(new_centers, dim=0), dim=-1)
    return centers


def balanced_sinkhorn_assignment(tokens, prototypes, temperature=0.05, iters=5):
    tokens = F.normalize(tokens, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    scores = tokens @ prototypes.t()
    logits = (scores / temperature).t()
    logits = logits - logits.max()
    q = torch.exp(logits)
    q = q / (q.sum() + 1e-8)

    k, n = q.shape
    r = torch.full((k,), 1.0 / k, device=q.device, dtype=q.dtype)
    c = torch.full((n,), 1.0 / n, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        q = q * (r / (q.sum(dim=1) + 1e-8)).unsqueeze(1)
        q = q * (c / (q.sum(dim=0) + 1e-8)).unsqueeze(0)

    assignment = q.t()
    assignment = assignment / (assignment.sum(dim=1, keepdim=True) + 1e-8)
    return assignment


def adaptive_prototype_refinement(tokens, prototypes, assignment, update_rate=0.15, confidence_threshold=0.25):
    proto_mass = assignment.sum(dim=0).unsqueeze(1)
    context = assignment.t() @ tokens / (proto_mass + 1e-8)
    context = F.normalize(context, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)

    confidence = (context * prototypes).sum(dim=-1, keepdim=True)
    gate = torch.sigmoid((confidence - confidence_threshold) * 10.0) * update_rate
    refined = F.normalize((1.0 - gate) * prototypes + gate * context, dim=-1)
    return refined


def adaptive_prototype_memory_update(
    tokens,
    prototypes,
    temperature=0.05,
    sinkhorn_iters=5,
    update_rate=0.15,
    confidence_threshold=0.25,
    update_iters=3,
):
    tokens = F.normalize(tokens, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    for _ in range(max(1, update_iters)):
        assignment = balanced_sinkhorn_assignment(
            tokens,
            prototypes,
            temperature=temperature,
            iters=sinkhorn_iters,
        )
        prototypes = adaptive_prototype_refinement(
            tokens,
            prototypes,
            assignment,
            update_rate=update_rate,
            confidence_threshold=confidence_threshold,
        )
    return prototypes


def structured_prototype_assignment(tokens, prototypes, temperature=0.05, sinkhorn_iters=5):
    assignment = balanced_sinkhorn_assignment(tokens, prototypes, temperature=temperature, iters=sinkhorn_iters)
    reconstruction = assignment @ prototypes
    return reconstruction, assignment


def gated_cross_modal_reconstruction(tokens, own_reconstruction, cross_prototypes, weight=0.5):
    tokens = F.normalize(tokens, dim=-1)
    cross_prototypes = F.normalize(cross_prototypes, dim=-1)
    attention = torch.softmax(tokens @ cross_prototypes.t() / math.sqrt(tokens.shape[-1]), dim=-1)
    cross_reconstruction = attention @ cross_prototypes
    confidence = attention.max(dim=-1, keepdim=True).values
    gate = torch.clamp(confidence * weight, 0.0, weight)
    return F.normalize((1.0 - gate) * own_reconstruction + gate * cross_reconstruction, dim=-1)


def _prototype_attention(query, prototypes, temperature=0.05):
    query = F.normalize(query, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    attention = torch.softmax((query @ prototypes.t()) / temperature, dim=-1)
    return attention @ prototypes, attention


def multi_stage_normality_communication(
    tokens,
    spa_reconstruction,
    own_prototypes,
    cross_prototypes,
    shared_prototypes=None,
    stage1_weight=0.5,
    stage2_weight=0.5,
    temperature=0.05,
):
    tokens = F.normalize(tokens, dim=-1)
    spa_reconstruction = F.normalize(spa_reconstruction, dim=-1)
    own_prototypes = F.normalize(own_prototypes, dim=-1)
    cross_prototypes = F.normalize(cross_prototypes, dim=-1)

    if shared_prototypes is not None:
        shared_prototypes = F.normalize(shared_prototypes, dim=-1)
        stage1_bank = torch.cat([cross_prototypes, shared_prototypes], dim=0)
    else:
        stage1_bank = cross_prototypes

    cross_context, cross_attention = _prototype_attention(spa_reconstruction, stage1_bank, temperature)
    cross_confidence = cross_attention.max(dim=-1, keepdim=True).values
    stage1_gate = torch.clamp(cross_confidence * stage1_weight, 0.0, stage1_weight)
    stage1 = F.normalize((1.0 - stage1_gate) * spa_reconstruction + stage1_gate * cross_context, dim=-1)

    own_context, own_attention = _prototype_attention(stage1, own_prototypes, temperature)
    own_confidence = own_attention.max(dim=-1, keepdim=True).values
    token_consistency = torch.sigmoid((tokens * stage1).sum(dim=-1, keepdim=True))
    stage2_gate = torch.clamp((own_confidence + token_consistency) * 0.5 * stage2_weight, 0.0, stage2_weight)
    stage2 = F.normalize((1.0 - stage2_gate) * stage1 + stage2_gate * own_context, dim=-1)
    return stage1, stage2


def reconstruction_error_map(tokens, reconstruction, out_size, feature_hw):
    err = torch.linalg.norm(F.normalize(tokens, dim=-1) - F.normalize(reconstruction, dim=-1), dim=-1)
    err = err.view(1, 1, *feature_hw)
    err = F.interpolate(err, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return err

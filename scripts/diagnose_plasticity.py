"""
Rigorous Representation Diagnostics for Project Linesight
Compares Foundational (3.18M) vs Late Saturated (10.64M) checkpoints:
1. Dormant Neurons (Sokar et al., ICML 2023) across multiple tau thresholds
2. Effective Rank (Roy & Vetterli, IEEE Trans. Sig. Proc. 2007)
3. Cumulative Threshold sRank (Kumar et al., ICML 2020)
4. Weight Frobenius and Spectral norms, Stable Rank
Evaluated on identical fixed deterministic state batches with a fixed tau quantile grid.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE_DIR))

from config_files import config_copy
from trackmania_rl.agents import iqn
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl import utilities

CHECKPOINTS = {
    "foundational_3.18M": BASE_DIR / "save" / "hocko_v1_foundational_checkpoint" / "weights1.torch",
    "saturated_10.64M": BASE_DIR / "save" / "hocko_v1" / "weights1.torch",
}


def compute_state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        h.update(state_dict[k].cpu().numpy().tobytes())
    return h.hexdigest()


def generate_fixed_state_batch(batch_size: int = 2048, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a deterministic, realistic batch of states covering track geometry,
    speeds, orientations, and lookahead vectors from ESL-Hockolicious.
    """
    rng = np.random.RandomState(seed)
    zone_centers = load_next_map_zone_centers("ESL-Hockolicious_0.5m_cl2.npy", BASE_DIR)
    total_vcps = len(zone_centers)
    finish_vcp = total_vcps - config_copy.n_zone_centers_extrapolate_after_end_of_map

    float_inputs = np.zeros((batch_size, config_copy.float_input_dim), dtype=np.float32)

    # Sample positions along track
    sample_indices = np.linspace(20, finish_vcp - 100, batch_size).astype(int)

    for i, idx in enumerate(sample_indices):
        pos = zone_centers[idx]
        forward_dir = zone_centers[idx + 1] - zone_centers[idx]
        forward_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)

        # 0: bias
        float_inputs[i, 0] = 0.0

        # 1-20: prev actions (one-hot / random active action)
        act = rng.randint(0, len(config_copy.inputs))
        for a_idx in range(5):
            for k_idx, k in enumerate(["accelerate", "brake", "left", "right"]):
                float_inputs[i, 1 + a_idx * 4 + k_idx] = float(config_copy.inputs[act][k])

        # 21-36: gear and wheels
        speed_mps = rng.uniform(20.0, 90.0)  # 72 to 324 km/h
        gear = int(min(5, max(1, speed_mps // 18)))
        rpm = float((speed_mps % 18) / 18.0 * 8000 + 2000)
        float_inputs[i, 21:25] = 0.0  # sliding
        float_inputs[i, 25:29] = 1.0  # ground contact
        float_inputs[i, 29:33] = 0.01 # damper absorb
        float_inputs[i, 33] = 1.0     # gearbox state
        float_inputs[i, 34] = float(gear)
        float_inputs[i, 35] = rpm
        float_inputs[i, 36] = 0.0     # gear counter

        # 37-40: angular velocity
        float_inputs[i, 37:40] = rng.normal(0.0, 0.2, size=3)

        # 40-43: linear velocity in car frame
        float_inputs[i, 40] = rng.normal(0.0, 1.0)
        float_inputs[i, 41] = 0.0
        float_inputs[i, 42] = speed_mps

        # 43-46: y map vector
        float_inputs[i, 43:46] = [0.0, 1.0, 0.0]

        # 46-166: 40 lookahead zone centers in car frame
        lookahead_step = config_copy.one_every_n_zone_centers_in_inputs
        for l_idx in range(config_copy.n_zone_centers_in_inputs):
            target_vcp = min(total_vcps - 1, idx + (l_idx + 1) * lookahead_step)
            rel_vec = zone_centers[target_vcp] - pos
            float_inputs[i, 46 + l_idx * 3 : 46 + (l_idx + 1) * 3] = rel_vec

        # 166: dist to finish
        float_inputs[i, 166] = min(float(config_copy.margin_to_announce_finish_meters), (finish_vcp - idx) * 0.5)

        # 167: freewheeling
        float_inputs[i, 167] = 0.0

    # Deterministic synthetic camera frames
    # Downsized dimensions: (1, 120, 160)
    img_inputs = np.zeros((batch_size, 1, 120, 160), dtype=np.uint8)
    for i in range(batch_size):
        # Road stripe in center
        img_inputs[i, 0, 40:80, 60:100] = 200
        # Road edge noise
        noise = rng.randint(50, 150, size=(120, 160), dtype=np.uint8)
        img_inputs[i, 0] = np.clip(img_inputs[i, 0].astype(np.int16) + (noise // 4), 0, 255).astype(np.uint8)

    img_tensor = (torch.from_numpy(img_inputs).float() - 128.0) / 128.0
    float_tensor = torch.from_numpy(float_inputs).float()
    return img_tensor, float_tensor


def compute_sokar_dormant_ratio(activations: torch.Tensor, tau: float) -> float:
    """
    Sokar et al. (ICML 2023) Dormant Neuron Metric:
    Score s_i = mean_batch(|h_i|) / mean_layer(mean_batch(|h_j|)).
    Dormant if s_i <= tau.
    """
    # For Conv layers: shape (B, C, H, W) -> average spatially to get per-channel activation
    if activations.dim() == 4:
        # Spatial mean of absolute activation per channel: (B, C)
        h = activations.abs().mean(dim=(2, 3))
    elif activations.dim() == 2:
        h = activations.abs()
    else:
        h = activations.abs().view(activations.shape[0], -1)

    # Batch mean per neuron: (D,)
    m = h.mean(dim=0)
    layer_mean = m.mean() + 1e-12

    # Normalized score
    s = m / layer_mean
    dormant_mask = (s <= tau)
    return float(dormant_mask.float().mean().item()) * 100.0


def compute_roy_vetterli_erank(h_matrix: torch.Tensor) -> float:
    """
    Roy & Vetterli (IEEE Trans. Sig. Proc. 2007) Effective Rank:
    SVD singular values -> probability distribution -> Shannon entropy -> exp(H).
    """
    H = h_matrix.float()
    if H.dim() > 2:
        H = H.view(H.shape[0], -1)

    # Center matrix
    H = H - H.mean(dim=0, keepdim=True)

    with torch.no_grad():
        try:
            S = torch.linalg.svdvals(H)
        except Exception:
            S = torch.svd(H)[1]

        S_sum = S.sum() + 1e-12
        p = S / S_sum

        # Entropy with 0*ln(0) = 0 handling
        non_zero = p > 1e-12
        p_nz = p[non_zero]
        entropy = -torch.sum(p_nz * torch.log(p_nz)).item()
        erank = math.exp(entropy)
        return float(erank)


def compute_kumar_srank(h_matrix: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Kumar et al. (ICML 2020) sRank:
    Minimum number of singular values whose sum explains `threshold` (e.g. 99%)
    of the total singular value sum.
    """
    H = h_matrix.float()
    if H.dim() > 2:
        H = H.view(H.shape[0], -1)

    H = H - H.mean(dim=0, keepdim=True)
    with torch.no_grad():
        try:
            S = torch.linalg.svdvals(H)
        except Exception:
            S = torch.svd(H)[1]

        cumsum = torch.cumsum(S, dim=0) / (S.sum() + 1e-12)
        idx = torch.where(cumsum >= threshold)[0]
        if len(idx) > 0:
            return int(idx[0].item()) + 1
        return len(S)


def compute_weight_metrics(weight_tensor: torch.Tensor) -> Dict[str, float]:
    """Computes Frobenius norm, Spectral norm, and Stable Rank of a 2D weight matrix."""
    W = weight_tensor.float()
    if W.dim() == 4:  # Conv filter
        W = W.view(W.shape[0], -1)
    elif W.dim() == 1:
        return {"frobenius": float(W.norm(2).item()), "spectral": float(W.abs().max().item()), "stable_rank": 1.0}

    with torch.no_grad():
        frob = float(W.norm("fro").item())
        try:
            sigma_max = float(torch.linalg.matrix_norm(W, ord=2).item())
        except Exception:
            sigma_max = float(torch.svd(W)[1][0].item())

        stable_rank = (frob ** 2) / (sigma_max ** 2 + 1e-12)
        return {
            "frobenius": round(frob, 4),
            "spectral": round(sigma_max, 4),
            "stable_rank": round(float(stable_rank), 4),
        }


def analyze_checkpoint(ckpt_key: str, ckpt_path: Path, img_tensor: torch.Tensor, float_tensor: torch.Tensor, device: str) -> Dict:
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_hash = compute_state_dict_sha256(state_dict)

    # Initialize uncompiled model on device for activation hook inspection
    model = iqn.IQN_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        iqn_embedding_dimension=config_copy.iqn_embedding_dimension,
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    ).to(device)
    model.eval()
    utilities.load_model_checkpoint(
        model,
        ckpt_path,
        map_location=device,
        allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
    )

    # Hook activations
    activations = {}

    def get_hook(name):
        def hook(mod, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    # Register hooks on LeakyReLU activation layers
    hooks = [
        model.img_head[1].register_forward_hook(get_hook("Conv1_Act")),
        model.img_head[3].register_forward_hook(get_hook("Conv2_Act")),
        model.img_head[5].register_forward_hook(get_hook("Conv3_Act")),
        model.img_head[7].register_forward_hook(get_hook("Conv4_Act")),
        model.float_feature_extractor[1].register_forward_hook(get_hook("Float_FC1_Act")),
        model.float_feature_extractor[3].register_forward_hook(get_hook("Float_FC2_Act")),
        model.A_head[1].register_forward_hook(get_hook("Advantage_FC1_Act")),
        model.V_head[1].register_forward_hook(get_hook("Value_FC1_Act")),
    ]

    # Deterministic fixed tau grid for IQN: K=32 quantiles evenly spaced
    num_quantiles = config_copy.iqn_k
    batch_size = img_tensor.shape[0]
    tau_fixed = torch.linspace(0.05, 0.95, num_quantiles, device=device).repeat_interleave(batch_size).unsqueeze(1)

    with torch.no_grad():
        img_dev = img_tensor.to(device)
        float_dev = float_tensor.to(device)
        Q, _ = model(img_dev, float_dev, num_quantiles=num_quantiles, tau=tau_fixed)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Also compute activations for the concatenated trunk
    with torch.no_grad():
        img_out = model.img_head(img_dev).detach().cpu()
        float_out = model.float_feature_extractor((float_dev - model.float_inputs_mean) / model.float_inputs_std).detach().cpu()
        concat_trunk = torch.cat((img_out, float_out), dim=1)
        activations["Joint_Trunk"] = concat_trunk

    # Compute dormant neuron percentages across tau grid
    tau_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    dormant_results = {}
    for layer_name, act in activations.items():
        dormant_results[layer_name] = {
            f"tau_{t}": round(compute_sokar_dormant_ratio(act, t), 2)
            for t in tau_grid
        }

    # Compute effective rank and sRank
    rank_results = {}
    for layer_name, act in activations.items():
        erank = compute_roy_vetterli_erank(act)
        srank_99 = compute_kumar_srank(act, threshold=0.99)
        dim = act.shape[1] if act.dim() == 2 else np.prod(act.shape[1:])
        rank_results[layer_name] = {
            "dim": int(dim),
            "roy_vetterli_erank": round(erank, 2),
            "kumar_srank_99": srank_99,
            "rank_utilization_pct": round((erank / dim) * 100.0, 2),
        }

    # Compute weight matrix properties
    weight_metrics = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            weight_metrics[name] = compute_weight_metrics(param.detach().cpu())

    return {
        "checkpoint": ckpt_key,
        "state_dict_sha256": state_hash,
        "dormant_neurons": dormant_results,
        "rank_metrics": rank_results,
        "weight_metrics": weight_metrics,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" PROJECT LINESIGHT — RIGOROUS REPRESENTATION DIAGNOSTICS")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Generate identical fixed state batch
    batch_size = 2048
    print(f"\n[+] Generating identical fixed state batch (B={batch_size}) from ESL-Hockolicious geometry...")
    img_tensor, float_tensor = generate_fixed_state_batch(batch_size=batch_size, seed=42)
    print(f"    Image Tensor: {img_tensor.shape}, dtype={img_tensor.dtype}")
    print(f"    Float Tensor: {float_tensor.shape}, dtype={float_tensor.dtype}")

    all_diagnostics = {}
    for ckpt_key, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"[!] Warning: Checkpoint path not found: {ckpt_path}")
            continue

        print(f"\n>>> Analyzing Checkpoint: [{ckpt_key}]")
        diag = analyze_checkpoint(ckpt_key, ckpt_path, img_tensor, float_tensor, device)
        all_diagnostics[ckpt_key] = diag
        print(f"    SHA256: {diag['state_dict_sha256']}")

    # Save raw JSON
    out_dir = BASE_DIR / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "representation_diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\n[+] Raw diagnostics saved to: {json_path}")

    # Print Comparative Tables
    print("\n" + "=" * 105)
    print(f"{'LAYER':<22} | {'METRIC':<18} | {'FOUNDATIONAL (3.18M)':<28} | {'SATURATED (10.64M)':<28}")
    print("=" * 105)

    f_diag = all_diagnostics.get("foundational_3.18M", {})
    s_diag = all_diagnostics.get("saturated_10.64M", {})

    # 1. Dormant Neurons Comparison (Sokar et al.)
    print(f"\n--- 1. DORMANT NEURON RATIO (%) [Sokar et al., tau=0.01 and tau=0.05] ---")
    print(f"{'Layer':<24} {'Dim':<8} {'Found (tau=0.01)':<18} {'Sat (tau=0.01)':<18} {'Found (tau=0.05)':<18} {'Sat (tau=0.05)':<18}")
    print("-" * 105)
    f_dorm = f_diag.get("dormant_neurons", {})
    s_dorm = s_diag.get("dormant_neurons", {})
    f_rank = f_diag.get("rank_metrics", {})
    for layer in f_dorm.keys():
        dim = f_rank.get(layer, {}).get("dim", "-")
        f_t01 = f_dorm[layer]["tau_0.01"]
        s_t01 = s_dorm[layer]["tau_0.01"]
        f_t05 = f_dorm[layer]["tau_0.05"]
        s_t05 = s_dorm[layer]["tau_0.05"]
        print(f"{layer:<24} {str(dim):<8} {f_t01:>6.1f}%{'':<11} {s_t01:>6.1f}%{'':<11} {f_t05:>6.1f}%{'':<11} {s_t05:>6.1f}%")

    # 2. Effective Rank Comparison (Roy & Vetterli vs Kumar et al. sRank)
    print(f"\n--- 2. REPRESENTATION DIMENSIONALITY [Roy & Vetterli Effective Rank vs Kumar sRank-99] ---")
    print(f"{'Layer':<24} {'Dim':<8} {'Found eRank':<14} {'Sat eRank':<14} {'Found sRank99':<16} {'Sat sRank99':<16}")
    print("-" * 105)
    s_rank = s_diag.get("rank_metrics", {})
    for layer in f_rank.keys():
        dim = f_rank[layer]["dim"]
        f_er = f_rank[layer]["roy_vetterli_erank"]
        s_er = s_rank[layer]["roy_vetterli_erank"]
        f_sr = f_rank[layer]["kumar_srank_99"]
        s_sr = s_rank[layer]["kumar_srank_99"]
        print(f"{layer:<24} {dim:<8} {f_er:>8.2f}{'':<6} {s_er:>8.2f}{'':<6} {f_sr:>6d}{'':<10} {s_sr:>6d}")

    # 3. Weight Matrix Stable Rank Comparison
    print(f"\n--- 3. WEIGHT MATRIX STABLE RANK (||W||_F^2 / sigma_max^2) ---")
    print(f"{'Weight Matrix':<36} {'Shape':<20} {'Found StableRank':<18} {'Sat StableRank':<18}")
    print("-" * 105)
    f_w = f_diag.get("weight_metrics", {})
    s_w = s_diag.get("weight_metrics", {})
    for name in f_w.keys():
        if "A_head" in name or "V_head" in name or "float_feature" in name or "img_head.0" in name:
            f_sr = f_w[name]["stable_rank"]
            s_sr = s_w[name]["stable_rank"]
            print(f"{name:<36} {'':<20} {f_sr:>8.2f}{'':<10} {s_sr:>8.2f}")

    print("\n" + "=" * 105)
    print(" SCIENTIFIC NOTE: These representation metrics are strictly correlational.")
    print(" Neither higher dormant unit ratios nor lower effective ranks prove loss of plasticity.")
    print(" Plasticity loss is a functional hypothesis verifiable ONLY via controlled fine-tuning.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

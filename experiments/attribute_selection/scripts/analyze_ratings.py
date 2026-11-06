import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import umap
import yaml


def varimax(loadings, max_iter=500, tol=1e-6):
    n, k = loadings.shape
    rotation = np.eye(k)
    for iteration in range(max_iter):
        old = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                cols = loadings @ rotation[:, [i, j]]
                u = cols[:, 0] ** 2 - cols[:, 1] ** 2
                v = 2 * cols[:, 0] * cols[:, 1]
                num = 2 * n * (u @ v) - 2 * u.sum() * v.sum()
                den = n * ((u ** 2 - v ** 2).sum()) - (u.sum() ** 2 - v.sum() ** 2)
                angle = 0.25 * np.arctan2(num, den)
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
                rotation[:, [i, j]] = rotation[:, [i, j]] @ rot
        if np.max(np.abs(rotation - old)) < tol:
            break
    if iteration == max_iter - 1:
        print(f"Warning: Varimax failed to converge within {max_iter} iterations.")
    return loadings @ rotation


def compute_entropy(col):
    col = col[~np.isnan(col)]
    _, counts = np.unique(col.round().astype(int), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.scores_path
    rng = np.random.default_rng(cfg.get("seed", 42))
    scores_matrix = np.load(npy_path)
    num_conversations, num_attributes = scores_matrix.shape
    attribute_names = list(cfg.candidate_attributes)

    if len(attribute_names) != num_attributes:
        raise ValueError(
            f"Number of attributes in config ({len(attribute_names)}) does not match "
            f"number of attributes in scores matrix ({num_attributes})"
        )

    # Handle NaN
    if np.isnan(scores_matrix).any():
        col_mean = np.nanmean(scores_matrix, axis=0)
        for j in range(num_attributes):
            mask = np.isnan(scores_matrix[:, j])
            if mask.any():
                scores_matrix[mask, j] = col_mean[j]
        print("Warning: Replaced NaN values with column means")

    # Per-attribute stats
    col_means = np.nanmean(scores_matrix, axis=0)
    col_stds = np.nanstd(scores_matrix, axis=0)
    col_entropies = np.array([compute_entropy(scores_matrix[:, j]) for j in range(num_attributes)])

    # Standardize
    col_std_safe = np.where(col_stds > 1e-10, col_stds, 1.0)
    scores_std = (scores_matrix - col_means) / col_std_safe

    zero_var = col_stds <= 1e-10
    if zero_var.any():
        zero_names = [attribute_names[i] for i in np.where(zero_var)[0]]
        print(f"Warning: {zero_var.sum()} zero-variance attribute(s): {zero_names[:5]}")
        scores_std[:, zero_var] = rng.standard_normal((num_conversations, zero_var.sum()))

    scores_std = np.asarray(scores_std, dtype=np.float64)
    corr_matrix = (scores_std.T @ scores_std) / (num_conversations - 1)
    np.fill_diagonal(corr_matrix, 1.0)
    if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
        raise ValueError("Correlation matrix contains NaN/Inf.")

    name_to_idx = {name: i for i, name in enumerate(attribute_names)}

    # ---- 1. Antonym sanity check ----
    antonym_pairs = [
        ('formal', 'casual'),
        ('concise', 'verbose'),
        ('optimistic', 'pessimistic'),
        ('emotionally neutral', 'appeals to emotion'),
        ('acknowledges uncertainty', 'speaks in absolutes'),
    ]
    print("Antonym correlations (negative = judge discriminates well):")
    for a, b in antonym_pairs:
        if a in name_to_idx and b in name_to_idx:
            c = corr_matrix[name_to_idx[a], name_to_idx[b]]
            col_a = scores_matrix[:, name_to_idx[a]]
            col_b = scores_matrix[:, name_to_idx[b]]
            pair_sum = col_a + col_b
            print(f"  {a} / {b}: corr={c:+.3f}")
            print(f"    {a}: mean={col_a.mean():.1f}  std={col_a.std():.1f}  range=[{col_a.min():.0f},{col_a.max():.0f}]")
            print(f"    {b}: mean={col_b.mean():.1f}  std={col_b.std():.1f}  range=[{col_b.min():.0f},{col_b.max():.0f}]")
            print(f"    sum: mean={pair_sum.mean():.1f}  std={pair_sum.std():.1f}  range=[{pair_sum.min():.0f},{pair_sum.max():.0f}]")
    print()

    # ---- 2. Score distributions grid ----
    score_min = int(np.nanmin(scores_matrix))
    score_max = int(np.nanmax(scores_matrix))
    bins = np.arange(score_min, score_max + 2) - 0.5

    ncols = 20
    nrows = (num_attributes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.4))
    axes = axes.flatten()

    for i, name in enumerate(attribute_names):
        ax = axes[i]
        counts, _ = np.histogram(scores_matrix[:, i], bins=bins)
        ax.bar(np.arange(score_min, score_max + 1), counts, width=0.8, color='steelblue', linewidth=0)
        ax.set_title(name, fontsize=4.5, pad=2)
        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='both', labelsize=3.5, length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(num_attributes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Score distributions per attribute', fontsize=12, y=1.002)
    plt.tight_layout(h_pad=1.2, w_pad=0.5)
    dist_path = npy_path.with_name(npy_path.stem + '_score_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {dist_path}")

    # ---- 2.5 Pre-filter: remove low-entropy attributes ----
    entropy_threshold = cfg.entropy_threshold
    keep_mask = col_entropies >= entropy_threshold
    removed = [attribute_names[i] for i in range(num_attributes) if not keep_mask[i]]
    print(f"Entropy filter (>= {entropy_threshold}): keeping {keep_mask.sum()}/{num_attributes} attributes")
    if removed:
        print(f"  Removed: {removed}")
    print()

    scores_std = scores_std[:, keep_mask]
    col_means = col_means[keep_mask]
    col_stds = col_stds[keep_mask]
    col_entropies = col_entropies[keep_mask]
    attribute_names = [attribute_names[i] for i in range(num_attributes) if keep_mask[i]]
    num_attributes_before_filtering = num_attributes
    num_attributes = keep_mask.sum()
    name_to_idx = {name: i for i, name in enumerate(attribute_names)}

    # Recompute correlation matrix on filtered set
    corr_matrix = (scores_std.T @ scores_std) / (num_conversations - 1)
    np.fill_diagonal(corr_matrix, 1.0)
    if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
        raise ValueError("Correlation matrix contains NaN/Inf after filtering.")

    # ---- 3. PCA + Varimax ----
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Parallel analysis — compute threshold at 95%
    n_iterations = 100
    random_eigenvalues = np.zeros((n_iterations, num_attributes))
    for i in range(n_iterations):
        random_data = rng.standard_normal((num_conversations, num_attributes))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigh(random_corr)[0])[::-1]
    threshold_95 = np.percentile(random_eigenvalues, 95, axis=0)
    k_95 = max(1, int(np.sum(eigenvalues > threshold_95)))

    # Determine k targets (2, 4, 6, 8... up to k_95)
    k_targets = list(range(2, k_95 + 1, 2))
    if k_95 not in k_targets:
        k_targets.append(k_95)

    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()

    lines = []
    lines.append(f"Parallel analysis: ceiling k_95={k_95}")
    lines.append(f"k_95 explains {cumulative_var[k_95-1]:.1%} of total variance")
    lines.append(f"Entropy filter (>= {entropy_threshold}): keeping {keep_mask.sum()}/{num_attributes_before_filtering} attributes")
    lines.append(f"Removed: {removed}")
    lines.append("")

    row_fmt = "  {:<45} {:>+8.3f}  {:>6.2f}  {:>6.2f}  {:>7.3f}"
    hdr = f"  {'Attribute':<45} {'Loading':>8}  {'Mean':>6}  {'Std':>6}  {'Entropy':>7}"

    for k in k_targets:
        lines.append(f"{'='*80}")
        lines.append(f"=== VARIMAX ROTATION WITH k={k} ===")
        lines.append(f"{'='*80}")
        lines.append("")

        loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        rotated_loadings = varimax(loadings)

        # Sort components by post-rotation sum of squared loadings (SSL).
        selected = []
        for comp_idx in range(k):
            ssl = float(np.sum(rotated_loadings[:, comp_idx] ** 2))
            best_idx = np.argmax(np.abs(rotated_loadings[:, comp_idx]))
            top5 = sorted(
                [(attribute_names[i], rotated_loadings[i, comp_idx]) for i in range(num_attributes)],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            selected.append({
                'component': comp_idx + 1,
                'ssl': ssl,
                'attribute': attribute_names[best_idx],
                'top5': top5,
            })

        selected.sort(key=lambda s: s['ssl'], reverse=True)

        for i, s in enumerate(selected):
            rank = i + 1
            lines.append(f"RC{rank} (SSL: {s['ssl']:.3f}, representative: {s['attribute']}):")
            lines.append(hdr)
            for name, loading in s['top5']:
                idx = name_to_idx[name]
                lines.append(row_fmt.format(name, loading, col_means[idx], col_stds[idx], col_entropies[idx]))
            lines.append("")

    output = "\n".join(lines)
    print(output)

    txt_path = npy_path.with_name(npy_path.stem + '_rcs.txt')
    txt_path.write_text(output)
    print(f"Saved {txt_path}")

    # ---- 3b. Write per-k attribute YAML files ----
    mode = cfg.get("mode", "response_attribute")
    subdir = "user_prompt_attributes" if mode == "prompt_attribute" else "response_attributes"
    yaml_dir = npy_path.parent / subdir
    yaml_dir.mkdir(parents=True, exist_ok=True)

    for k in k_targets:
        loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        rotated_loadings = varimax(loadings)
        # One representative attribute per component (highest absolute loading)
        attrs = []
        for comp_idx in range(k):
            best_idx = int(np.argmax(np.abs(rotated_loadings[:, comp_idx])))
            attrs.append(attribute_names[best_idx])
        yaml_path = yaml_dir / f"{subdir}_k{k}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"candidate_attributes": attrs}, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved {yaml_path}")

    # Scree plot
    n_show_scree = min(60, num_attributes)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_show_scree + 1), eigenvalues[:n_show_scree], 'bo-', label='Data')
    plt.plot(range(1, n_show_scree + 1), threshold_95[:n_show_scree], 'r--', label='95th percentile (random)')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'Scree Plot with Parallel Analysis (k_95={k_95})')
    plt.axvline(x=k_95, color='red', linestyle=':', alpha=0.5, label=f'k_95={k_95}')
    plt.legend()
    plt.tight_layout()
    scree_path = npy_path.with_name(npy_path.stem + '_scree.png')
    plt.savefig(scree_path, dpi=150)
    plt.close()
    print(f"Saved {scree_path}")

    # ---- 4. UMAP ----
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    embedding = umap.UMAP(metric='precomputed', random_state=42).fit_transform(distance_matrix)

    fig, ax = plt.subplots(figsize=(20, 16))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.6)
    for i, name in enumerate(attribute_names):
        ax.annotate(name, (embedding[i, 0], embedding[i, 1]), fontsize=10, alpha=0.8)
    ax.set_title('UMAP of attributes (distance = 1 - |correlation|)')
    ax.axis('off')
    plt.tight_layout()
    umap_path = npy_path.with_name(npy_path.stem + '_umap.png')
    plt.savefig(umap_path, dpi=200)
    plt.close()
    print(f"Saved {umap_path}")


if __name__ == "__main__":
    main()
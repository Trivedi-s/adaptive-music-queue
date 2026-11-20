"""Train the TD agents, evaluate them against the baselines, and plot learning curves.

Trains each learner on a uniform mix of the five listener profiles, `--runs` times with
different training seeds, evaluating on held-out listeners every `--eval-every` episodes
with exploration off. Baselines are evaluated once on the same held-out listeners.

Outputs:
  results/learning_curves.csv   agent, run, episodes, profile, mean return
  results/final.csv             final held-out return per agent, run and profile
  results/summary.md            mean ± 95% CI over runs, per profile and overall
  figures/learning_curves.png   mean held-out return vs training episodes

Usage:
  python scripts/train_agents.py                    # full run (5 seeds x 3 learners)
  python scripts/train_agents.py --episodes 4000 --runs 2 --eval-seeds 40   # quick check
  python scripts/train_agents.py --plot-only        # redraw the figure from results/*.csv
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from amq import PROFILES, Catalog  # noqa: E402
from amq.agents import TDAgent  # noqa: E402
from amq.evaluate import REFERENCE_AGENTS, evaluate, make_envs, train  # noqa: E402

LEARNERS = {
    "Q-learning": dict(method="q_learning"),
    "SARSA": dict(method="sarsa"),
    "Q-learning, no belief": dict(method="q_learning", use_belief=False),
}
# Validated categorical slots 1-3 (dataviz reference palette, light mode).
COLORS = {"Q-learning": "#2a78d6", "SARSA": "#eb6834", "Q-learning, no belief": "#1baf7a"}
STYLES = {"Q-learning": "-", "SARSA": "-", "Q-learning, no belief": "--"}
PLOTTED_REFERENCES = ["oracle", "belief greedy (no learning)", "skip-streak k=2",
                      "sliding-window bandit", "random"]
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def train_one(job):
    name, run, args = job
    catalog = Catalog.load()
    envs = make_envs(catalog, PROFILES)
    agent = TDAgent(catalog, alpha=args.alpha, gamma=args.gamma, seed=run, **LEARNERS[name])
    curve = train(agent, envs, args.episodes, eval_every=args.eval_every,
                  eval_seeds=args.eval_seeds, seed=run)
    final = evaluate(agent, envs, args.final_seeds)
    return name, run, curve, final


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.mean(), 0.0
    half = T95.get(len(values) - 1, 1.96) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half


def overall(scores):
    return float(np.mean([scores[p] for p in PROFILES]))


def spread_labels(ys, min_gap):
    """Nudge label positions apart (in data units) so they don't overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        out[cur] = max(out[cur], out[prev] + min_gap)
    return out


def plot(curves, refs, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ink, muted, grid = "#2b2b29", "#6b6b66", "#e6e5e0"
    for name in LEARNERS:
        runs = curves[name]
        x = np.array([ep for ep, _ in runs[0]])
        y = np.array([[overall(s) for _, s in run] for run in runs])
        m = y.mean(axis=0)
        half = np.array([mean_ci(y[:, j])[1] for j in range(y.shape[1])])
        ax.fill_between(x, m - half, m + half, color=COLORS[name], alpha=0.15, linewidth=0)
        ax.plot(x, m, STYLES[name], color=COLORS[name], linewidth=2, label=name)

    ref_y = [overall(refs[r]) for r in PLOTTED_REFERENCES]
    lo, hi = min(ref_y + [ax.get_ylim()[0]]), max(ref_y + [ax.get_ylim()[1]])
    label_y = spread_labels(ref_y, min_gap=0.045 * (hi - lo))
    for name, y, ly in zip(PLOTTED_REFERENCES, ref_y, label_y):
        ax.axhline(y, color=muted, linewidth=1, linestyle=":", zorder=1)
        ax.annotate(f"{name}  {y:.1f}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, (ly - y) / (hi - lo) * 5.5 * 72 * 0.8), textcoords="offset points",
                    va="center", fontsize=9, color=muted)

    ax.set_xlabel("training episodes (sessions)", color=ink)
    ax.set_ylabel("mean session return on held-out listeners\n(average over the 5 profiles)", color=ink)
    ax.set_title("Learning curves: tabular TD agents vs. baselines", color=ink, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), frameon=False, labelcolor=ink,
              title=f"shaded: 95% CI over {len(curves[next(iter(LEARNERS))])} training seeds",
              title_fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(path, dpi=150)


def load_results(results):
    """Rebuild (curves, refs) from the CSVs a previous run wrote."""
    curves = {name: {} for name in LEARNERS}
    with open(results / "learning_curves.csv") as f:
        for row in csv.DictReader(f):
            run = curves[row["agent"]].setdefault(int(row["run"]), {})
            run.setdefault(int(row["episodes"]), {})[row["profile"]] = float(row["mean_return"])
    curves = {name: [sorted(runs[r].items()) for r in sorted(runs)] for name, runs in curves.items()}
    refs = {}
    with open(results / "final.csv") as f:
        for row in csv.DictReader(f):
            if row["run"] == "":
                refs.setdefault(row["agent"], {})[row["profile"]] = float(row["mean_return"])
    return curves, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-seeds", type=int, default=100, help="held-out listeners per profile per checkpoint")
    parser.add_argument("--final-seeds", type=int, default=200, help="held-out listeners per profile at the end")
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel training processes (each uses one CPU core)")
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from results/*.csv")
    args = parser.parse_args()

    if args.plot_only:
        curves, refs = load_results(ROOT / "results")
        plot(curves, refs, ROOT / "figures" / "learning_curves.png")
        print("Saved figures/learning_curves.png")
        return

    t0 = time.time()
    jobs = [(name, run, args) for name in LEARNERS for run in range(args.runs)]
    curves = {name: [None] * args.runs for name in LEARNERS}
    finals = {name: [None] * args.runs for name in LEARNERS}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, run, curve, final in pool.map(train_one, jobs):
            curves[name][run], finals[name][run] = curve, final
            print(f"  {name} run {run}: final {overall(final):.2f}  ({time.time() - t0:.0f}s)", flush=True)

    catalog = Catalog.load()
    envs = make_envs(catalog, PROFILES)
    refs = {name: evaluate(cls(catalog), envs, args.final_seeds) for name, cls in REFERENCE_AGENTS.items()}

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with open(results / "learning_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "run", "episodes", "profile", "mean_return"])
        for name, runs in curves.items():
            for run, curve in enumerate(runs):
                for ep, scores in curve:
                    for p in PROFILES:
                        w.writerow([name, run, ep, p, f"{scores[p]:.4f}"])
    with open(results / "final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "run", "profile", "mean_return"])
        for name, runs in finals.items():
            for run, scores in enumerate(runs):
                for p in PROFILES:
                    w.writerow([name, run, p, f"{scores[p]:.4f}"])
        for name, scores in refs.items():
            for p in PROFILES:
                w.writerow([name, "", p, f"{scores[p]:.4f}"])

    header = "| Agent | " + " | ".join(PROFILES) + " | **mean** |"
    lines = [
        f"Mean session return (25 tracks) on {args.final_seeds} held-out listeners per profile. "
        f"Learners: {args.episodes} training episodes, α = {args.alpha}, γ = {args.gamma}, "
        f"ε 1.0 → 0.05; mean ± 95% CI over {args.runs} training seeds.",
        "",
        header,
        "|" + "---|" * (len(PROFILES) + 2),
    ]
    for name in LEARNERS:
        cells = []
        for p in list(PROFILES) + ["mean"]:
            vals = [overall(s) if p == "mean" else s[p] for s in finals[name]]
            m, h = mean_ci(vals)
            cells.append(f"{m:.2f} ± {h:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells[:-1]) + f" | **{cells[-1]}** |")
    for name, scores in refs.items():
        lines.append(f"| {name} | " + " | ".join(f"{scores[p]:.2f}" for p in PROFILES)
                     + f" | **{overall(scores):.2f}** |")
    (results / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    (ROOT / "figures").mkdir(exist_ok=True)
    plot(curves, refs, ROOT / "figures" / "learning_curves.png")
    print(f"\nSaved results/ and figures/learning_curves.png in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

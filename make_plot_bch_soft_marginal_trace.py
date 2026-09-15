"""Interactively plot marginal probabilities from a BCH-soft trace dump.

Only load pickle files that you trust: loading pickle files can execute
arbitrary code.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TRACE_FILE = (
    Path(__file__).resolve().parent.parent
    / "rawdump-bch-soft-v3"
    / "trace-bch-soft-marginal.pkl"
)


def _load_trace(filename):
    with open(filename, "rb") as trace_file:
        payload = pickle.load(trace_file)
    if payload.get("format") != "trace_bch_soft_marginal":
        raise ValueError(f"unsupported trace format in {filename}")
    if payload.get("version") != 1:
        raise ValueError(f"unsupported trace version in {filename}")
    return payload


def _nearest_alternative(puzzle, reference):
    matrix = np.asarray(puzzle["matrix"], dtype=np.uint8)
    rhs = np.asarray(puzzle["rhs"], dtype=np.uint8)
    standard = np.asarray(puzzle["correct_answer"], dtype=np.uint8)
    variable_count = matrix.shape[1]
    assignments = np.arange(1 << variable_count, dtype=np.uint64)[:, None]
    bits = ((assignments >> np.arange(variable_count, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    valid = bits[np.all((bits @ matrix.T) % 2 == rhs, axis=1)]
    alternatives = valid[np.any(valid != standard, axis=1)]
    if len(alternatives) == 0:
        return np.full(variable_count, 0.5, dtype=np.float64)
    squared_distances = np.sum(np.square(alternatives - reference), axis=1)
    return alternatives[np.argmin(squared_distances)].astype(np.float64)


def _plot_data(payload):
    puzzle = payload["puzzle"]
    variable_count = int(puzzle["variables"])
    priors = np.asarray(puzzle["percentages"], dtype=np.float64) / 100.0
    exact = np.asarray(puzzle["exact_marginals"], dtype=np.float64)
    map_assignment = np.asarray(puzzle["correct_answer"], dtype=np.float64)
    expected_shape = (variable_count,)
    if any(
        values.shape != expected_shape for values in (priors, exact, map_assignment)
    ):
        raise ValueError(
            "prior, exact marginals, or MAP assignment has an unexpected shape"
        )

    columns = [priors]
    labels = ["Prior"]
    for trace in payload["traces"]:
        reported = trace.get("reported_marginals")
        if reported is None:
            values = np.full(variable_count, np.nan)
        else:
            values = np.asarray(reported, dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(
                    f"trace {trace.get('request_index')} has an unexpected shape"
                )
        columns.append(values)
        labels.append(f"LLM ({trace['total_dots']} dots)")
    columns.append(exact)
    labels.append("Exact marginal")
    columns.append(map_assignment)
    labels.append("Standard solution")
    final_reported = payload["traces"][-1].get("reported_marginals")
    if final_reported is None:
        raise ValueError("last LLM response has no parsed marginal probabilities")
    final_reported = np.asarray(final_reported, dtype=np.float64)
    if final_reported.shape != expected_shape:
        raise ValueError("last LLM response has an unexpected shape")
    columns.append(_nearest_alternative(puzzle, final_reported))
    labels.append("Nearest alternative")
    return np.column_stack(columns), labels


def _marginal_metrics(data):
    """Return per-column RMS error and mean Bernoulli KL divergence."""
    exact = data[:, -3]
    rms_errors = np.full(data.shape[1], np.nan)
    kl_divergences = np.full(data.shape[1], np.nan)
    epsilon = np.finfo(np.float64).eps

    for column_index in range(1, data.shape[1] - 3):
        reported = data[:, column_index]
        if np.any(~np.isfinite(reported)):
            continue

        rms_errors[column_index] = np.sqrt(np.mean(np.square(reported - exact)))

        # Treat each marginal as a Bernoulli distribution and average the
        # divergences over variables. Clipping keeps exact 0/1 reports finite.
        clipped_reported = np.clip(reported, epsilon, 1.0 - epsilon)
        with np.errstate(divide="ignore", invalid="ignore"):
            true_terms = np.where(
                exact == 0.0,
                0.0,
                exact * (np.log(exact) - np.log(clipped_reported)),
            )
            false_exact = 1.0 - exact
            false_terms = np.where(
                false_exact == 0.0,
                0.0,
                false_exact * (np.log(false_exact) - np.log1p(-clipped_reported)),
            )
        kl_divergences[column_index] = np.mean(true_terms + false_terms)

    return rms_errors, kl_divergences


def main():
    parser = argparse.ArgumentParser(
        description="Interactively plot a BCH-soft marginal trace."
    )
    parser.add_argument(
        "trace_file",
        nargs="?",
        default=DEFAULT_TRACE_FILE,
        help=f"trace pickle (default: {DEFAULT_TRACE_FILE})",
    )
    args = parser.parse_args()

    payload = _load_trace(args.trace_file)
    data, column_labels = _plot_data(payload)
    rms_errors, kl_divergences = _marginal_metrics(data)
    variable_labels = [chr(ord("a") + index) for index in range(data.shape[0])]

    # figure_width = max(8.0, 0.75 * data.shape[1])
    # figure_height = max(4.5, 0.45 * data.shape[0])
    figure_width = 12
    figure_height = 7
    figure, (marginal_axis, metric_axis) = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    colormap = plt.get_cmap("RdBu").copy()
    colormap.set_bad("lightgray")
    marginal_axis.imshow(data, cmap=colormap, vmin=-0.1, vmax=1.1, aspect="auto")
    for axis in (marginal_axis, metric_axis):
        axis.axvline(0.5, color="k", linewidth=2)
        axis.axvline(data.shape[1] - 3.5, color="k", linewidth=2)

    marginal_axis.set_yticks(np.arange(data.shape[0]), labels=variable_labels)
    # axis.set_xlabel("Marginal source / LLM input padding")
    marginal_axis.set_ylabel("Variable")
    marginal_axis.set_title(
        "P(variable=True): prior, LLM trace, exact marginal, and solutions"
    )

    column_indices = np.arange(data.shape[1])
    metric_axis.plot(
        column_indices,
        rms_errors,
        marker="o",
        label="RMS error",
    )
    metric_axis.plot(
        column_indices,
        kl_divergences,
        marker="o",
        label="Mean KL divergence",
    )
    metric_axis.set_ylim(0, 1)
    metric_axis.set_ylabel("Value")
    metric_axis.grid(axis="y", alpha=0.3)
    metric_axis.legend()
    metric_axis.set_xticks(column_indices, labels=column_labels)
    metric_axis.tick_params(axis="x", labelrotation=45)
    for label in metric_axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # figure.colorbar(image, ax=axis, label="P(variable=True)", fraction=0.01)
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

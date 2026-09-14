"""Plot sum-product belief-propagation trajectories for BCH-soft dumps.

Only load pickle files that you trust: pickle loading can execute arbitrary code.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

OUTCOME_COLORS = {
    "correct": "tab:green",
    "alternative": "tab:orange",
    "wrong (first-second)": "tab:red",
    "wrong (other)": "tab:purple",
}

HISTOGRAM_OUTCOME_COLORS = {
    "correct": "tab:green",
    "alternative": "tab:orange",
    "wrong": "tab:red",
}

BIT_OUTCOME_COLORS = {
    "wrong": "tab:red",
    "correct": "tab:green",
}


def _validate_problem(problem):
    matrix = np.asarray(problem["matrix"], dtype=np.uint8)
    rhs = np.asarray(problem["rhs"], dtype=np.uint8)
    percentages = np.asarray(problem["percentages"], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("rhs must have one value per parity check")
    if percentages.shape != (matrix.shape[1],):
        raise ValueError("percentages must have one value per variable")
    if np.any((matrix != 0) & (matrix != 1)) or np.any((rhs != 0) & (rhs != 1)):
        raise ValueError("matrix and rhs must be binary")
    if np.any((percentages <= 0) | (percentages >= 100)):
        raise ValueError("percentages must be strictly between 0 and 100")
    return matrix, rhs, percentages / 100.0


def sum_product_trajectory(problem, iterations=50, temperature=1.0, clip=1e-15):
    """Return variable P(True) beliefs after each flooding BP iteration.

    Row factors enforce XOR(matrix[row] * variables) == rhs[row].  Messages
    are represented as log(P(False) / P(True)) likelihood ratios.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    matrix, rhs, probabilities = _validate_problem(problem)
    edge_rows, edge_columns = np.nonzero(matrix)
    edge_count = len(edge_rows)
    prior_llr = (np.log1p(-probabilities) - np.log(probabilities)) / temperature

    variable_to_check = prior_llr[edge_columns].copy()
    check_to_variable = np.zeros(edge_count, dtype=np.float64)
    beliefs = [1.0 / (1.0 + np.exp(np.clip(prior_llr, -745, 745)))]

    row_edges = [np.flatnonzero(edge_rows == row) for row in range(matrix.shape[0])]
    column_edges = [
        np.flatnonzero(edge_columns == column) for column in range(matrix.shape[1])
    ]

    for _ in range(iterations):
        next_check_to_variable = np.empty_like(check_to_variable)
        tanh_messages = np.tanh(variable_to_check / 2.0)
        for row, edges in enumerate(row_edges):
            parity_sign = -1.0 if rhs[row] else 1.0
            for edge in edges:
                other_edges = edges[edges != edge]
                product = parity_sign * np.prod(tanh_messages[other_edges])
                product = np.clip(product, -1.0 + clip, 1.0 - clip)
                next_check_to_variable[edge] = 2.0 * np.arctanh(product)
        check_to_variable = next_check_to_variable

        posterior_llr = prior_llr.copy()
        for column, edges in enumerate(column_edges):
            posterior_llr[column] += check_to_variable[edges].sum()
        beliefs.append(1.0 / (1.0 + np.exp(np.clip(posterior_llr, -745, 745))))

        next_variable_to_check = np.empty_like(variable_to_check)
        for column, edges in enumerate(column_edges):
            total = posterior_llr[column]
            next_variable_to_check[edges] = total - check_to_variable[edges]
        variable_to_check = next_variable_to_check

    return np.asarray(beliefs)


def _max_sum_check_message(incoming, parity):
    """Return the best incoming score for each requested XOR parity."""
    best = np.array([0.0, -np.inf])
    for message in incoming:
        best = np.array(
            [
                max(best[0] + message[0], best[1] + message[1]),
                max(best[0] + message[1], best[1] + message[0]),
            ]
        )
    return best[parity]


def _scores_to_true_probability(scores):
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights[..., 1] / weights.sum(axis=-1)


def min_sum_trajectory(problem, iterations=50, temperature=1.0):
    """Return max-product/min-sum P(True) pseudo-beliefs by iteration."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    matrix, rhs, probabilities = _validate_problem(problem)
    edge_rows, edge_columns = np.nonzero(matrix)
    unary_scores = (
        np.column_stack((np.log1p(-probabilities), np.log(probabilities))) / temperature
    )
    unary_scores -= unary_scores.max(axis=1, keepdims=True)

    variable_to_check = unary_scores[edge_columns].copy()
    check_to_variable = np.zeros_like(variable_to_check)
    beliefs = [_scores_to_true_probability(unary_scores)]
    row_edges = [np.flatnonzero(edge_rows == row) for row in range(matrix.shape[0])]
    column_edges = [
        np.flatnonzero(edge_columns == column) for column in range(matrix.shape[1])
    ]

    for _ in range(iterations):
        next_check_to_variable = np.empty_like(check_to_variable)
        for row, edges in enumerate(row_edges):
            for edge in edges:
                incoming = variable_to_check[edges[edges != edge]]
                for value in (0, 1):
                    required_parity = int(rhs[row]) ^ value
                    next_check_to_variable[edge, value] = _max_sum_check_message(
                        incoming, required_parity
                    )
                next_check_to_variable[edge] -= next_check_to_variable[edge].max()
        check_to_variable = next_check_to_variable

        posterior_scores = unary_scores.copy()
        for column, edges in enumerate(column_edges):
            posterior_scores[column] += check_to_variable[edges].sum(axis=0)
        beliefs.append(_scores_to_true_probability(posterior_scores))

        next_variable_to_check = np.empty_like(variable_to_check)
        for column, edges in enumerate(column_edges):
            next_variable_to_check[edges] = (
                posterior_scores[column] - check_to_variable[edges]
            )
            next_variable_to_check[edges] -= next_variable_to_check[edges].max(
                axis=1, keepdims=True
            )
        variable_to_check = next_variable_to_check

    return np.asarray(beliefs)


def _exact_solution_scores(problem, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    matrix, rhs, probabilities = _validate_problem(problem)
    variable_count = matrix.shape[1]
    assignments = np.arange(1 << variable_count, dtype=np.uint64)[:, None]
    bits = ((assignments >> np.arange(variable_count, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    satisfies = np.all((bits @ matrix.T) % 2 == rhs, axis=1)
    solutions = bits[satisfies]
    if len(solutions) == 0:
        raise ValueError("parity checks have no satisfying assignment")
    log_scores = (
        solutions @ (np.log(probabilities) - np.log1p(-probabilities))
        + np.log1p(-probabilities).sum()
    ) / temperature
    return solutions, log_scores


def exact_marginals(problem, temperature=1.0):
    """Enumerate the posterior and return exact P(True) variable marginals."""
    solutions, log_weights = _exact_solution_scores(problem, temperature)
    weights = np.exp(log_weights - log_weights.max())
    return (weights @ solutions) / weights.sum()


def exact_max_marginals(problem, temperature=1.0):
    """Enumerate exact max-marginal scores and map them to P(True) values."""
    solutions, log_scores = _exact_solution_scores(problem, temperature)
    max_scores = np.empty((solutions.shape[1], 2), dtype=np.float64)
    for column in range(solutions.shape[1]):
        for value in (0, 1):
            matching = solutions[:, column] == value
            max_scores[column, value] = (
                log_scores[matching].max() if np.any(matching) else -np.inf
            )
    return _scores_to_true_probability(max_scores)


def _parse_model_answer(problem, variable_count):
    content = problem.get("model_answer")
    if not isinstance(content, str):
        return None
    parts = content.strip().removeprefix("(").removesuffix(")").split(",")
    if len(parts) != variable_count:
        return None
    values = []
    for part in parts:
        value = part.strip()
        if value in ("0", "false", "False"):
            values.append(0)
        elif value in ("1", "true", "True"):
            values.append(1)
        else:
            return None
    return np.asarray(values, dtype=np.uint8)


def _second_best_solution(problem, correct_answer):
    solutions, log_scores = _exact_solution_scores(problem, temperature=1.0)
    is_correct = np.all(solutions == correct_answer, axis=1)
    if not np.any(is_correct):
        raise ValueError("correct_answer is not a parity-check solution")
    other_indices = np.flatnonzero(~is_correct)
    if len(other_indices) == 0:
        return None
    return solutions[other_indices[np.argmax(log_scores[other_indices])]]


def _outcome(problem):
    if problem["correct"]:
        return "correct"
    if problem["alternative"]:
        return "alternative"

    correct_answer = np.asarray(problem["correct_answer"], dtype=np.uint8)
    candidate = _parse_model_answer(problem, len(correct_answer))
    if candidate is None:
        return "wrong (other)"
    second_best = _second_best_solution(problem, correct_answer)
    if second_best is None:
        return "wrong (other)"
    first_second_bits = correct_answer != second_best
    wrong_bits = candidate != correct_answer
    if np.all(~wrong_bits | first_second_bits):
        return "wrong (first-second)"
    return "wrong (other)"


def load_problems(filename):
    with open(filename, "rb") as dump_file:
        payload = pickle.load(dump_file)
    if not isinstance(payload, dict) or payload.get("format") != "benchmark_bch_soft":
        raise ValueError("not a benchmark_bch_soft dump")
    if payload.get("version") != 1:
        raise ValueError(f"unsupported dump version: {payload.get('version')!r}")
    return payload["problems"]


def _probability_ratio(problem):
    """Return the stored ratio, or reconstruct it for older compatible dumps."""
    if "probability_ratio" in problem:
        return float(problem["probability_ratio"])
    _, log_scores = _exact_solution_scores(problem, temperature=1.0)
    if len(log_scores) == 1:
        return np.inf
    top_two = np.partition(log_scores, -2)[-2:]
    return float(np.exp(top_two.max() - top_two.min()))


def _derived_ratio_filename(filename):
    path = Path(filename)
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_ratios{suffix}")


def _derived_marginal_map_filename(filename):
    path = Path(filename)
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_marginal_map{suffix}")


def _derived_bit_marginal_map_filename(filename):
    path = Path(filename)
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_bit_marginal_map{suffix}")


def _histogram_outcome(outcome):
    """Collapse detailed trajectory outcomes into three histogram categories."""
    return outcome if outcome in ("correct", "alternative") else "wrong"


def main():
    parser = argparse.ArgumentParser(
        description="Plot iterative inference trajectories, colored by LLM outcome."
    )
    parser.add_argument(
        "pickle_files",
        nargs="+",
        help="one or more pickles produced by benchmark_bch_soft.py",
    )
    parser.add_argument("-o", "--output", help="write the plot to this file")
    parser.add_argument(
        "--ratio-output",
        help="write the probability-ratio histogram to this file",
    )
    parser.add_argument(
        "--ratio-bins",
        type=int,
        default=4,
        help="number of logarithmic probability-ratio bins (default: 4)",
    )
    parser.add_argument(
        "--marginal-map-output",
        help="write the exact-marginal/MAP-difference histogram to this file",
    )
    parser.add_argument(
        "--marginal-map-bins",
        type=int,
        default=20,
        help="number of exact-marginal/MAP-difference bins (default: 20)",
    )
    parser.add_argument(
        "--bit-marginal-output",
        help="write the per-bit marginal/MAP-difference histogram to this file",
    )
    parser.add_argument(
        "--bit-marginal-bins",
        type=int,
        default=20,
        help="number of per-bit marginal/MAP-difference bins (default: 20)",
    )
    parser.add_argument("-n", "--iterations", type=int, default=50)
    parser.add_argument(
        "--algorithm",
        choices=("bp", "min-sum"),
        default="bp",
        help="inference algorithm to run (default: bp, meaning sum-product)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="positive temperature for priors P(x) ** (1/T) (default: 1)",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "max"),
        default="mean",
        help="aggregation for absolute exact-target error (default: mean)",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="include invalid model responses, classified by their answer flags",
    )
    parser.add_argument("--title", default="Inference trajectories by LLM outcome")
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")
    if args.ratio_bins < 1:
        parser.error("--ratio-bins must be at least 1")
    if args.marginal_map_bins < 1:
        parser.error("--marginal-map-bins must be at least 1")
    if args.bit_marginal_bins < 1:
        parser.error("--bit-marginal-bins must be at least 1")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("plotting requires matplotlib (install it in the project .venv)")

    problems = [
        problem
        for pickle_file in args.pickle_files
        for problem in load_problems(pickle_file)
    ]
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    marginal_axis, map_axis = axes
    ratios_by_outcome = {outcome: [] for outcome in OUTCOME_COLORS}
    marginal_map_differences_by_metric = {
        metric: {outcome: [] for outcome in HISTOGRAM_OUTCOME_COLORS}
        for metric in ("Maximum absolute error", "Mean absolute error", "RMS error")
    }
    bit_marginal_map_differences_by_outcome = {
        outcome: [] for outcome in BIT_OUTCOME_COLORS
    }
    plotted = 0
    seen_outcomes = set()
    for problem in problems:
        if not problem["valid"] and not args.include_invalid:
            continue
        if args.algorithm == "bp":
            beliefs = sum_product_trajectory(problem, args.iterations, args.temperature)
            exact_marginal = exact_marginals(problem, args.temperature)
            target = exact_marginal
            target_name = "marginal"
        else:
            beliefs = min_sum_trajectory(problem, args.iterations, args.temperature)
            target = exact_max_marginals(problem, args.temperature)
            exact_marginal = exact_marginals(problem, args.temperature)
            target_name = "max-marginal"
        absolute_errors = np.abs(beliefs - target)
        errors = (
            absolute_errors.mean(axis=1)
            if args.metric == "mean"
            else absolute_errors.max(axis=1)
        )
        correct_answer = np.asarray(problem["correct_answer"], dtype=np.uint8)
        if correct_answer.shape != (beliefs.shape[1],):
            raise ValueError("correct_answer must have one value per variable")
        map_errors = np.where(correct_answer, 1.0 - beliefs, beliefs).mean(axis=1)
        outcome = _outcome(problem)
        ratios_by_outcome[outcome].append(_probability_ratio(problem))
        histogram_outcome = _histogram_outcome(outcome)
        marginal_map_difference = exact_marginal - correct_answer
        absolute_marginal_map_difference = np.abs(marginal_map_difference)
        marginal_map_differences_by_metric["Maximum absolute error"][
            histogram_outcome
        ].append(absolute_marginal_map_difference.max())
        marginal_map_differences_by_metric["Mean absolute error"][
            histogram_outcome
        ].append(absolute_marginal_map_difference.mean())
        marginal_map_differences_by_metric["RMS error"][histogram_outcome].append(
            np.sqrt(np.mean(np.square(marginal_map_difference)))
        )
        candidate = _parse_model_answer(problem, len(correct_answer))
        if candidate is not None:
            bit_is_correct = candidate == correct_answer
            bit_marginal_map_difference = np.abs(exact_marginal - correct_answer)
            bit_marginal_map_differences_by_outcome["correct"].extend(
                bit_marginal_map_difference[bit_is_correct]
            )
            bit_marginal_map_differences_by_outcome["wrong"].extend(
                bit_marginal_map_difference[~bit_is_correct]
            )
        label = outcome.capitalize() if outcome not in seen_outcomes else None
        seen_outcomes.add(outcome)
        plot_kwargs = {
            "color": OUTCOME_COLORS[outcome],
            "alpha": 0.45,
            "linewidth": 1.5,
        }
        marginal_axis.plot(
            np.arange(args.iterations + 1),
            errors,
            label=label,
            **plot_kwargs,
        )
        map_axis.plot(
            np.arange(args.iterations + 1),
            map_errors,
            **plot_kwargs,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("the dump contains no plottable problems")
    marginal_axis.set_ylabel(
        f"{args.metric.capitalize()} absolute\n{target_name} error"
    )
    marginal_axis.set_title(
        f"{args.title}\nAlgorithm: {args.algorithm}; temperature: {args.temperature:g}"
    )
    marginal_axis.grid(alpha=0.2)
    marginal_axis.legend()
    map_axis.set_xlabel("Inference iteration")
    map_axis.set_ylabel("Mean probability assigned\nto the wrong MAP bit")
    map_axis.grid(alpha=0.2)
    figure.tight_layout()

    finite_ratios = [
        ratio
        for ratios in ratios_by_outcome.values()
        for ratio in ratios
        if np.isfinite(ratio)
    ]
    if not finite_ratios:
        raise ValueError("the plottable problems have no finite probability ratios")
    if min(finite_ratios) <= 0:
        raise ValueError("probability ratios must be positive")
    minimum_ratio = min(finite_ratios)
    maximum_ratio = max(finite_ratios)
    if minimum_ratio == maximum_ratio:
        minimum_ratio /= 1.05
        maximum_ratio *= 1.05
    ratio_bins = np.geomspace(minimum_ratio, maximum_ratio, args.ratio_bins + 1)
    ratio_figure, ratio_axis = plt.subplots(figsize=(8, 5))
    histogram_outcomes = list(OUTCOME_COLORS)
    histogram_ratios = []
    for outcome in histogram_outcomes:
        ratios = np.asarray(ratios_by_outcome[outcome], dtype=np.float64)
        histogram_ratios.append(ratios[np.isfinite(ratios)])
    ratio_axis.hist(
        histogram_ratios,
        bins=ratio_bins,
        stacked=True,
        color=[OUTCOME_COLORS[outcome] for outcome in histogram_outcomes],
        label=[outcome.capitalize() for outcome in histogram_outcomes],
    )
    infinite_count = sum(
        not np.isfinite(ratio)
        for ratios in ratios_by_outcome.values()
        for ratio in ratios
    )
    ratio_axis.set_xscale("log")
    ratio_axis.set_xlabel("Best/second probability ratio")
    ratio_axis.set_ylabel("Problems")
    ratio_axis.set_title("Probability-ratio distribution by LLM outcome")
    ratio_axis.grid(alpha=0.2)
    ratio_axis.legend()
    if infinite_count:
        ratio_axis.text(
            0.99,
            0.95,
            f"{infinite_count} infinite ratio(s) omitted",
            ha="right",
            va="top",
            transform=ratio_axis.transAxes,
        )
    ratio_figure.tight_layout()

    marginal_map_figure, marginal_map_axes = plt.subplots(
        3, 1, sharex=True, figsize=(8, 12)
    )
    marginal_map_outcomes = list(HISTOGRAM_OUTCOME_COLORS)
    marginal_map_bins = np.linspace(0.0, 1.0, args.marginal_map_bins + 1)
    for axis, (metric_name, differences_by_outcome) in zip(
        marginal_map_axes, marginal_map_differences_by_metric.items()
    ):
        axis.hist(
            [differences_by_outcome[outcome] for outcome in marginal_map_outcomes],
            bins=marginal_map_bins,
            stacked=True,
            color=[
                HISTOGRAM_OUTCOME_COLORS[outcome] for outcome in marginal_map_outcomes
            ],
            label=[outcome.capitalize() for outcome in marginal_map_outcomes],
        )
        axis.set_ylabel("Problems")
        axis.set_title(metric_name)
        axis.grid(alpha=0.2)
        axis.legend()
    marginal_map_axes[-1].set_xlim(0.0, 1.0)
    marginal_map_axes[-1].set_xlabel(
        "Difference between exact marginals and MAP assignment"
    )
    marginal_map_figure.suptitle(
        "Exact-marginal/MAP difference distribution by LLM outcome"
    )
    marginal_map_figure.tight_layout()

    bit_marginal_figure, bit_marginal_axis = plt.subplots(figsize=(8, 5))
    bit_marginal_outcomes = list(BIT_OUTCOME_COLORS)
    bit_marginal_axis.hist(
        [
            bit_marginal_map_differences_by_outcome[outcome]
            for outcome in bit_marginal_outcomes
        ],
        bins=np.linspace(0.0, 1.0, args.bit_marginal_bins + 1),
        stacked=True,
        color=[BIT_OUTCOME_COLORS[outcome] for outcome in bit_marginal_outcomes],
        label=[outcome.capitalize() for outcome in bit_marginal_outcomes],
    )
    bit_marginal_axis.set_xlim(0.0, 1.0)
    bit_marginal_axis.set_xlabel(
        "Absolute difference between exact marginal and MAP bit"
    )
    bit_marginal_axis.set_ylabel("LLM answer bits")
    bit_marginal_axis.set_title("Per-bit LLM outcome by marginal/MAP difference")
    bit_marginal_axis.grid(alpha=0.2)
    bit_marginal_axis.legend()
    bit_marginal_figure.tight_layout()

    ratio_output = args.ratio_output
    if ratio_output is None and args.output:
        ratio_output = _derived_ratio_filename(args.output)
    marginal_map_output = args.marginal_map_output
    if marginal_map_output is None and args.output:
        marginal_map_output = _derived_marginal_map_filename(args.output)
    bit_marginal_output = args.bit_marginal_output
    if bit_marginal_output is None and args.output:
        bit_marginal_output = _derived_bit_marginal_map_filename(args.output)
    if args.output:
        figure.savefig(args.output, dpi=200)
        print(f"Plotted {plotted} trajectories to {args.output}")
    if ratio_output:
        ratio_figure.savefig(ratio_output, dpi=200)
        print(f"Plotted probability ratios to {ratio_output}")
    if marginal_map_output:
        marginal_map_figure.savefig(marginal_map_output, dpi=200)
        print(f"Plotted exact-marginal/MAP differences to {marginal_map_output}")
    if bit_marginal_output:
        bit_marginal_figure.savefig(bit_marginal_output, dpi=200)
        print(f"Plotted per-bit marginal/MAP differences to {bit_marginal_output}")
    if (
        not args.output
        and not ratio_output
        and not marginal_map_output
        and not bit_marginal_output
    ):
        plt.show()


if __name__ == "__main__":
    main()

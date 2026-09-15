import argparse
import itertools
import math
import pickle
import time

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = "Do not reason, deliberate, or work anything out. Emit the final answer on your very first token. No explanation, no steps, no scratch work."


class BernoulliSPRT:
    """Sequentially distinguish p <= p_low from p >= p_high."""

    def __init__(
        self,
        p_low=0.4,
        p_high=0.6,
        alpha=0.05,
        beta=0.05,
        max_trials=None,
    ):
        if not 0 < p_low < p_high < 1:
            raise ValueError("Require 0 < p_low < p_high < 1.")
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("alpha and beta must be between 0 and 1.")

        self.max_trials = max_trials
        self.llr_success = math.log(p_high / p_low)
        self.llr_failure = math.log((1.0 - p_high) / (1.0 - p_low))
        self.upper_log_bound = math.log(1.0 / alpha)
        self.lower_log_bound = math.log(beta)
        self.trials = 0
        self.successes = 0
        self.log_likelihood_ratio = 0.0
        self.status = "CONTINUE"

    def step(self, outcome):
        if self.status != "CONTINUE":
            return self.status
        if outcome not in (0, 1, False, True):
            raise ValueError("Outcome must be 0 or 1.")

        outcome = int(outcome)
        self.trials += 1
        self.successes += outcome
        self.log_likelihood_ratio += self.llr_success if outcome else self.llr_failure

        if self.log_likelihood_ratio >= self.upper_log_bound:
            self.status = "HIGH: evidence for p >= p_high"
        elif self.log_likelihood_ratio <= self.lower_log_bound:
            self.status = "LOW: evidence for p <= p_low"
        elif self.max_trials is not None and self.trials >= self.max_trials:
            self.status = "INCONCLUSIVE"
        return self.status


def parse_bool(value):
    value = value.strip()
    if value in ("0", "false", "False"):
        return False
    if value in ("1", "true", "True"):
        return True
    raise ValueError("Invalid bool")


def parse_marginals(content, variable_count):
    parts = content.strip().removeprefix("(").removesuffix(")").split(",")
    if len(parts) != variable_count:
        raise ValueError(f"expected {variable_count} comma-separated probabilities")
    values = np.asarray([float(part.strip()) for part in parts], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and between 0 and 1")
    return values


# Polynomial arithmetic over GF(2). Polynomials are stored as integers.
def _poly_mod(a, modulus):
    modulus_degree = modulus.bit_length() - 1
    while a and a.bit_length() - 1 >= modulus_degree:
        a ^= modulus << (a.bit_length() - 1 - modulus_degree)
    return a


def _poly_mul_mod(a, b, modulus):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
    return _poly_mod(result, modulus)


def _poly_pow_mod(a, exponent, modulus):
    result = 1
    while exponent:
        if exponent & 1:
            result = _poly_mul_mod(result, a, modulus)
        a = _poly_mul_mod(a, a, modulus)
        exponent >>= 1
    return result


def _poly_gcd(a, b):
    while b:
        a, b = b, _poly_mod(a, b)
    return a


def _prime_factors(n):
    factors = set()
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.add(divisor)
            n //= divisor
        divisor += 2
    if n > 1:
        factors.add(n)
    return factors


def _is_irreducible(poly, degree):
    x = 0b10
    if _poly_pow_mod(x, 1 << degree, poly) != x:
        return False
    for factor in _prime_factors(degree):
        test = _poly_pow_mod(x, 1 << (degree // factor), poly) ^ x
        if _poly_gcd(poly, test) != 1:
            return False
    return True


def _is_primitive(poly, degree):
    if not _is_irreducible(poly, degree):
        return False
    order = (1 << degree) - 1
    for factor in _prime_factors(order):
        if _poly_pow_mod(0b10, order // factor, poly) == 1:
            return False
    return True


def find_primitive_polynomial(degree):
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError("degree must be an integer")
    if not 3 <= degree <= 16:
        raise ValueError("degree must be between 3 and 16")
    for middle in range(1 << (degree - 1)):
        poly = (1 << degree) | (middle << 1) | 1
        if _is_primitive(poly, degree):
            return poly
    raise RuntimeError(f"could not find a primitive polynomial of degree {degree}")


def _random_row_mix(matrix, rng):
    matrix = matrix.copy()
    row_count = matrix.shape[0]
    for _ in range(8 * row_count * row_count):
        first, second = rng.choice(row_count, size=2, replace=False)
        if rng.integers(2):
            matrix[first] ^= matrix[second]
        else:
            matrix[[first, second]] = matrix[[second, first]]
    return matrix


def _balanced_random_row_mix(matrix, rng, candidates):
    if isinstance(candidates, bool) or not isinstance(candidates, (int, np.integer)):
        raise TypeError("candidates must be an integer")
    if candidates < 1:
        raise ValueError("candidates must be at least 1")

    upper = matrix.shape[1] // 2
    lower = min(3, upper)
    best = None
    best_score = None
    tied = 0
    for _ in range(candidates):
        mixed = _random_row_mix(matrix, rng)
        weights = mixed.sum(axis=1, dtype=np.int64)
        distance = weights - np.clip(weights, lower, upper)
        score = int(distance @ distance)
        if best_score is None or score < best_score:
            best, best_score, tied = mixed, score, 1
        elif score == best_score:
            tied += 1
            if rng.integers(tied) == 0:
                best = mixed
    return best


def random_bch_parity_check(
    variable_count, correctable_errors=2, rng=None, row_mix_candidates=64
):
    """Construct a randomized, shortened binary BCH parity-check matrix."""
    if isinstance(variable_count, bool) or not isinstance(
        variable_count, (int, np.integer)
    ):
        raise TypeError("variable_count must be an integer")
    if not 2 <= variable_count <= (1 << 16) - 1:
        raise ValueError("variable_count must be between 2 and 65535")
    if isinstance(correctable_errors, bool) or not isinstance(
        correctable_errors, (int, np.integer)
    ):
        raise TypeError("correctable_errors must be an integer")
    if correctable_errors < 1:
        raise ValueError("correctable_errors must be at least 1")

    degree = max(3, int(variable_count).bit_length())
    rng = np.random.default_rng(rng)
    primitive_poly = find_primitive_polynomial(degree)
    code_length = (1 << degree) - 1
    if 2 * correctable_errors >= code_length:
        raise ValueError(
            "correctable_errors must be less than half the full BCH code length "
            f"({code_length})"
        )
    powers = np.empty(code_length, dtype=np.uint64)
    powers[0] = 1
    for index in range(1, code_length):
        value = int(powers[index - 1]) << 1
        if value & (1 << degree):
            value ^= primitive_poly
        powers[index] = value & code_length

    matrix = np.empty((correctable_errors * degree, code_length), dtype=np.uint8)
    for index in range(code_length):
        for syndrome_index in range(correctable_errors):
            exponent = 2 * syndrome_index + 1
            value = int(powers[(exponent * index) % code_length])
            row_offset = syndrome_index * degree
            for bit in range(degree):
                matrix[row_offset + bit, index] = (value >> bit) & 1

    matrix = matrix[:, rng.permutation(code_length)[:variable_count]]
    return _balanced_random_row_mix(matrix, rng, row_mix_candidates)


def _gf2_matvec(matrix, values):
    return np.bitwise_xor.reduce(matrix & values[None, :], axis=1)


def _variable_name(index):
    letter = chr(ord("a") + index % 26)
    suffix = "" if index < 26 else str(index // 26)
    return letter + suffix


def all_parity_solutions(matrix, rhs):
    """Enumerate every binary solution to matrix @ x = rhs over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8)
    rhs = np.asarray(rhs, dtype=np.uint8)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    row_count, variable_count = matrix.shape
    if rhs.shape != (row_count,):
        raise ValueError("rhs must have one value per row")
    if np.any((matrix != 0) & (matrix != 1)) or np.any((rhs != 0) & (rhs != 1)):
        raise ValueError("matrix and rhs must be binary")

    augmented = np.column_stack((matrix.copy(), rhs.copy()))
    pivot_columns = []
    pivot_row = 0
    for column in range(variable_count):
        candidates = np.flatnonzero(augmented[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        augmented[[pivot_row, selected]] = augmented[[selected, pivot_row]]
        for row in range(row_count):
            if row != pivot_row and augmented[row, column]:
                augmented[row] ^= augmented[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    if np.any(
        np.all(augmented[:, :variable_count] == 0, axis=1)
        & (augmented[:, variable_count] == 1)
    ):
        return []

    free_columns = [
        column for column in range(variable_count) if column not in pivot_columns
    ]
    solutions = []
    for free_values in itertools.product((0, 1), repeat=len(free_columns)):
        solution = np.zeros(variable_count, dtype=np.uint8)
        solution[free_columns] = free_values
        for row, column in enumerate(pivot_columns):
            solution[column] = augmented[row, variable_count] ^ np.bitwise_xor.reduce(
                augmented[row, free_columns] & solution[free_columns], initial=0
            )
        solutions.append(solution)
    return solutions


def solution_log_probability(solution, percentages):
    probabilities = np.asarray(percentages, dtype=np.float64) / 100.0
    solution = np.asarray(solution, dtype=np.uint8)
    return float(
        np.where(solution, np.log(probabilities), np.log1p(-probabilities)).sum()
    )


def choose_probabilities(
    solutions,
    rng,
    ratio_min,
    ratio_max,
    max_attempts,
    marginal_min=0.0,
    marginal_max=1.0,
):
    """Sample priors until the ratio and marginal/MAP difference are in range."""
    if math.isnan(ratio_min) or ratio_min < 1:
        raise ValueError("ratio_min must be at least 1")
    if math.isnan(ratio_max) or ratio_max < ratio_min:
        raise ValueError("ratio_max must be at least ratio_min")
    if not 0.0 <= marginal_min <= 1.0:
        raise ValueError("marginal_min must be between 0 and 1")
    if not marginal_min <= marginal_max <= 1.0:
        raise ValueError("marginal_max must be between marginal_min and 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    solution_array = np.asarray(solutions, dtype=np.uint8)
    if len(solution_array) == 0:
        raise ValueError("at least one valid solution is required")

    inclusive_ratio_min = np.nextafter(ratio_min, -math.inf)
    inclusive_ratio_max = np.nextafter(ratio_max, math.inf)
    inclusive_marginal_min = np.nextafter(marginal_min, -math.inf)
    inclusive_marginal_max = np.nextafter(marginal_max, math.inf)

    for _ in range(max_attempts):
        percentages = rng.integers(10, 91, size=solution_array.shape[1])
        probabilities = percentages / 100.0
        log_false = np.log1p(-probabilities)
        log_odds = np.log(probabilities) - log_false
        log_scores = solution_array @ log_odds + log_false.sum()
        if len(log_scores) == 1:
            best_index, probability_ratio = 0, math.inf
        else:
            top_two = np.argpartition(log_scores, -2)[-2:]
            if log_scores[top_two[0]] > log_scores[top_two[1]]:
                best_index, second_index = map(int, top_two)
            else:
                second_index, best_index = map(int, top_two)
            log_ratio = float(log_scores[best_index] - log_scores[second_index])
            probability_ratio = math.exp(log_ratio)
        if not inclusive_ratio_min <= probability_ratio <= inclusive_ratio_max:
            continue

        weights = np.exp(log_scores - log_scores.max())
        exact_marginals = (weights @ solution_array) / weights.sum()
        maximum_marginal_map_difference = np.abs(
            exact_marginals - solution_array[best_index]
        ).max()
        if (
            inclusive_marginal_min
            <= maximum_marginal_map_difference
            <= inclusive_marginal_max
        ):
            return (
                percentages,
                solution_array[best_index],
                exact_marginals,
                probability_ratio,
                float(maximum_marginal_map_difference),
            )

    raise RuntimeError(
        f"could not find probabilities with a best/second ratio in the inclusive "
        f"range [{ratio_min}, {ratio_max}] and a maximum absolute marginal/MAP "
        f"difference in the inclusive range [{marginal_min}, {marginal_max}] "
        f"in {max_attempts} attempts"
    )


def bch_to_soft_logic(matrix, percentages, rhs, phrasing="original"):
    matrix = np.asarray(matrix, dtype=np.uint8)
    percentages = np.asarray(percentages)
    rhs = np.asarray(rhs, dtype=np.uint8)
    row_count, variable_count = matrix.shape
    if percentages.shape != (variable_count,):
        raise ValueError("percentages must have one value per variable")
    if rhs.shape != (row_count,):
        raise ValueError("rhs must have one value per row")

    names = [_variable_name(index) for index in range(variable_count)]
    priors = "\n".join(
        f"{name} = {int(percentage)}% chance of being True"
        for name, percentage in zip(names, percentages)
    )
    checks = []
    for row, expected in zip(matrix, rhs):
        variables = [names[index] for index, included in enumerate(row) if included]
        expression = " xor ".join(variables) if variables else "False"
        checks.append(f"{expression} = {bool(expected)}")

    logic = (
        "All variables are boolean.\n\n"
        + priors
        + "\n\nThe following expressions are all correct:\n"
        + "\n".join(checks)
    )
    if phrasing == "original":
        return logic + f"\n\nWhat is the most likely value of ({', '.join(names)})?"
    if phrasing == "marginal":
        return (
            logic
            + "\n\nReport the exact marginal probability P(variable=True), conditioned "
            "on all expressions above, for each variable in this order: "
            f"({', '.join(names)}). Return exactly one tuple of "
            f"{len(names)} decimal numbers between 0 and 1."
        )
    raise ValueError(f"unsupported phrasing: {phrasing!r}")


def run_test(llm, args, variables):
    rng = np.random.default_rng()
    matrix = random_bch_parity_check(
        variables,
        correctable_errors=args.correctable_errors,
        rng=rng,
        row_mix_candidates=args.row_mix_candidates,
    )
    seed_solution = rng.integers(0, 2, size=variables, dtype=np.uint8)
    rhs = _gf2_matvec(matrix, seed_solution)
    solutions = all_parity_solutions(matrix, rhs)
    (
        percentages,
        answer,
        exact_marginals,
        probability_ratio,
        maximum_marginal_map_difference,
    ) = choose_probabilities(
        solutions,
        rng,
        args.ratio_min,
        args.ratio_max,
        args.max_probability_attempts,
        args.marginal_min,
        args.marginal_max,
    )
    logic = bch_to_soft_logic(matrix, percentages, rhs, args.phrasing)

    lines = ["Answer immediately with one tuple and nothing else.", "", logic]
    if args.input_dots is not None:
        lines.append(". " * args.input_dots)
    prompt = "\n".join(lines)
    print(prompt)
    print()

    if args.dry_run:
        content = "(dry run)"
        reasoning_tokens = 0
    else:
        history = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
        while True:
            try:
                response = llm.invoke(history)
                break
            except Exception as error:
                print(error)
                print("Retrying in 10 seconds ...")
                time.sleep(10)
        print(response)
        print()

        if isinstance(response.content, str):
            content = response.content
        elif (
            isinstance(response.content, list)
            and len(response.content) == 1
            and "text" in response.content[0]
        ):
            content = response.content[0]["text"]
        else:
            content = "(unrecognized response)"

        details = response.usage_metadata.get("output_token_details", {})
        reasoning_tokens = details.get("reasoning", 0)
        reasoning_tokens += max(0, response.usage_metadata["output_tokens"] - 80)

    correct = False
    alternative = False
    reported_marginals = None
    candidate = None
    try:
        if args.phrasing == "marginal":
            reported_marginals = parse_marginals(content, variables)
            candidate = (reported_marginals >= 0.5).astype(np.uint8)
        else:
            parts = content.strip().removeprefix("(").removesuffix(")").split(",")
            if len(parts) == variables:
                candidate = np.array(
                    [parse_bool(part) for part in parts], dtype=np.uint8
                )
        if candidate is not None:
            correct = np.array_equal(candidate, answer)
            alternative = np.array_equal(_gf2_matvec(matrix, candidate), rhs)
    except ValueError:
        pass

    valid = reasoning_tokens == 0
    print(f"Model answer: {content}")
    if reported_marginals is not None:
        print(f"Thresholded answer: {tuple(bool(value) for value in candidate)}")
    print(f"Correct answer: {tuple(bool(value) for value in answer)}")
    print(f"Best/second probability ratio: {probability_ratio:.6g}")
    print(
        "Maximum absolute marginal/MAP difference: "
        f"{maximum_marginal_map_difference:.6g}"
    )
    print(f"Reasoning tokens: {reasoning_tokens}")
    result = "CORRECT" if correct else "ALTERNATIVE" if alternative else "WRONG"
    print(result if valid else "INVALID")
    print()

    if getattr(args, "dump", None) is not None:
        args.dump_records.append(
            {
                "variables": variables,
                "matrix": matrix.copy(),
                "rhs": rhs.copy(),
                "percentages": percentages.copy(),
                "correct_answer": answer.copy(),
                "exact_marginals": exact_marginals.copy(),
                "probability_ratio": probability_ratio,
                "maximum_marginal_map_difference": maximum_marginal_map_difference,
                "logic": logic,
                "prompt": prompt,
                "model_answer": content,
                "reported_marginals": (
                    None if reported_marginals is None else reported_marginals.copy()
                ),
                "thresholded_answer": (
                    None if candidate is None else candidate.copy()
                ),
                "reasoning_tokens": reasoning_tokens,
                "correct": bool(correct),
                "valid": bool(valid),
                "alternative": bool(alternative),
            }
        )
    return correct, valid, alternative


def run_sample(llm, args, variables):
    test = BernoulliSPRT(max_trials=args.repeats)
    score = 0
    score_alt = 0
    invalid = 0
    total = 0
    for _ in range(args.repeats):
        correct, valid, alternative = run_test(llm, args, variables)
        score += int(correct and valid)
        score_alt += int(alternative and not correct and valid)
        invalid += int(not valid)
        total += 1
        if args.horizon and test.step(correct and valid) != "CONTINUE":
            print(test.status)
            break

    print(f"Provider:          {args.provider}")
    print(f"Model:             {args.model}")
    print(f"Phrasing:          {args.phrasing}")
    print(f"Variables:         {variables}")
    print(f"Correctable errors: {args.correctable_errors}")
    print(f"Ratio range:       [{args.ratio_min}, {args.ratio_max}]")
    print(f"Marginal range:    [{args.marginal_min}, {args.marginal_max}]")
    print(f"Input dots:        {args.input_dots}")
    print(f"Total correct:     {score} / {total}  ({100 * score // total}%)")
    print(f"Total alternative: {score_alt} / {total}  ({100 * score_alt // total}%)")
    print(f"Total invalid:     {invalid} / {total}  ({100 * invalid // total}%)")
    return score, invalid, total


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference over XOR constraints with probabilistic priors."
    )
    parser.add_argument("-v", "--variables", type=int, default=10)
    parser.add_argument("-r", "--repeats", type=int, default=50)
    parser.add_argument("--ratio-min", type=float, default=4.0)
    parser.add_argument("--ratio-max", type=float, default=np.inf)
    parser.add_argument("--marginal-min", type=float, default=0.0)
    parser.add_argument("--marginal-max", type=float, default=1.0)
    parser.add_argument("--max-probability-attempts", type=int, default=10000)
    parser.add_argument(
        "--correctable-errors",
        type=int,
        default=1,
        help="number of errors the BCH code can correct (default: 1)",
    )
    parser.add_argument("--row-mix-candidates", type=int, default=64)
    parser.add_argument("-p", "--provider", type=str, default="openai")
    parser.add_argument("-m", "--model", type=str, default="gpt-5.6-luna")
    parser.add_argument(
        "--phrasing",
        choices=("original", "marginal"),
        default="original",
        help="prompt/output phrasing to use (default: original)",
    )
    parser.add_argument("--horizon", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-dots", type=int, default=None)
    parser.add_argument(
        "--dump",
        metavar="FILE",
        help="pickle every generated puzzle and its model outcome to FILE",
    )
    parser.add_argument("--lo", type=int, default=None)
    parser.add_argument("--hi", type=int, default=None)
    args = parser.parse_args()
    args.dump_records = []

    if not 2 <= args.variables <= 21:
        parser.error("--variables must be between 2 and 21")
    if math.isnan(args.ratio_min) or args.ratio_min < 1:
        parser.error("--ratio-min must be at least 1")
    if math.isnan(args.ratio_max) or args.ratio_max < args.ratio_min:
        parser.error("--ratio-max must be at least --ratio-min")
    if not 0.0 <= args.marginal_min <= 1.0:
        parser.error("--marginal-min must be between 0 and 1")
    if not args.marginal_min <= args.marginal_max <= 1.0:
        parser.error("--marginal-max must be between --marginal-min and 1")
    if args.max_probability_attempts < 1:
        parser.error("--max-probability-attempts must be at least 1")
    if args.correctable_errors < 1:
        parser.error("--correctable-errors must be at least 1")

    kwargs = {}
    if args.model in ("gpt-4", "gpt-4.1", "claude-sonnet-4-5", "claude-haiku-4-5"):
        pass
    elif args.model in ("gpt-5", "gpt-5-mini", "gpt-5-nano"):
        kwargs["reasoning_effort"] = "minimal"
    elif args.model in (
        "gpt-6-astra",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-opus-5",
    ):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["reasoning_effort"] = "none"

    llm = (
        None
        if args.dry_run
        else init_chat_model(
            model_provider=args.provider,
            model=args.model,
            max_tokens=100,
            **kwargs,
        )
    )

    if args.horizon:
        if args.lo is not None and args.hi is not None:
            lo, hi = args.lo, args.hi
        else:
            lo, hi = 1, 2
            while True:
                score, _, total = run_sample(llm, args, hi)
                if score * 2 < total:
                    break
                if hi >= 21:
                    lo = hi
                    break
                lo, hi = hi, min(2 * hi, 21)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            score, _, total = run_sample(llm, args, mid)
            if score * 2 < total:
                hi = mid
            else:
                lo = mid
        print()
        print(f"Success horizon: {lo} variables")
    else:
        run_sample(llm, args, args.variables)

    if args.dump is not None:
        payload = {
            "format": "benchmark_bch_soft",
            "version": 1,
            "benchmark": {
                "provider": args.provider,
                "model": args.model,
                "phrasing": args.phrasing,
                "ratio_min": args.ratio_min,
                "ratio_max": args.ratio_max,
                "marginal_min": args.marginal_min,
                "marginal_max": args.marginal_max,
                "correctable_errors": args.correctable_errors,
                "row_mix_candidates": args.row_mix_candidates,
                "input_dots": args.input_dots,
                "dry_run": args.dry_run,
            },
            "problems": args.dump_records,
        }
        with open(args.dump, "wb") as dump_file:
            pickle.dump(payload, dump_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Dumped {len(args.dump_records)} problems to {args.dump}")


if __name__ == "__main__":
    main()

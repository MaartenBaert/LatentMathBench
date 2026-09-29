import argparse
import itertools
import math
import numpy as np
import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = "Do not reason, deliberate, or work anything out. Emit the final answer on your very first token. No explanation, no steps, no scratch work."


class BernoulliSPRT:
    """
    Sequential test of:

        LOW:  p <= p_low
        HIGH: p >= p_high

    No error guarantee is made when p_low < p < p_high.
    """

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

        self.p_low = p_low
        self.p_high = p_high
        self.alpha = alpha
        self.beta = beta
        self.max_trials = max_trials

        # Per-observation log likelihood-ratio increments:
        # log[P(X | p_high) / P(X | p_low)]
        self.llr_success = math.log(p_high / p_low)
        self.llr_failure = math.log((1.0 - p_high) / (1.0 - p_low))

        # Conservative anytime-valid thresholds.
        #
        # Under p <= p_low:
        #   P(ever selecting HIGH) <= alpha
        #
        # Under p >= p_high:
        #   P(ever selecting LOW) <= beta
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

        if outcome == 1:
            self.log_likelihood_ratio += self.llr_success
        else:
            self.log_likelihood_ratio += self.llr_failure

        if self.log_likelihood_ratio >= self.upper_log_bound:
            self.status = "HIGH: evidence for p >= p_high"
        elif self.log_likelihood_ratio <= self.lower_log_bound:
            self.status = "LOW: evidence for p <= p_low"
        elif self.max_trials is not None and self.trials >= self.max_trials:
            # Do not force a decision without separately calibrating
            # the resulting error probabilities.
            self.status = "INCONCLUSIVE"

        return self.status

    @property
    def sample_proportion(self):
        if self.trials == 0:
            return None
        return self.successes / self.trials


def parse_bool(val):
    val = val.strip()
    if val in (0, "false", "False"):
        return False
    elif val in (1, "true", "True"):
        return True
    else:
        raise ValueError("Invalid bool")


import numpy as np


# ---------------------------------------------------------------------------
# Polynomial arithmetic over GF(2), used to find a primitive polynomial.
# Polynomials are represented as integers:
#     x^3 + x + 1  <->  0b1011
# ---------------------------------------------------------------------------


def _poly_mod(a, modulus):
    md = modulus.bit_length() - 1
    while a and a.bit_length() - 1 >= md:
        a ^= modulus << ((a.bit_length() - 1) - md)
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

    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.add(p)
            n //= p
        p += 2

    if n > 1:
        factors.add(n)

    return factors


def _is_irreducible(poly, r):
    """
    Rabin irreducibility test over GF(2).
    """
    x = 0b10

    # x^(2^r) must equal x modulo poly.
    if _poly_pow_mod(x, 1 << r, poly) != x:
        return False

    # For every prime divisor q of r:
    # gcd(poly, x^(2^(r/q)) - x) must be 1.
    for q in _prime_factors(r):
        test = _poly_pow_mod(x, 1 << (r // q), poly) ^ x
        if _poly_gcd(poly, test) != 1:
            return False

    return True


def _is_primitive(poly, r):
    if not _is_irreducible(poly, r):
        return False

    order = (1 << r) - 1
    x = 0b10

    # x must have multiplicative order 2^r - 1.
    for q in _prime_factors(order):
        if _poly_pow_mod(x, order // q, poly) == 1:
            return False

    return True


def find_primitive_polynomial(r):
    """
    Find a degree-r primitive polynomial over GF(2).

    Suitable for the relatively small values of r used for logic puzzles.
    """
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)):
        raise TypeError("r must be an integer")
    if not 3 <= r <= 16:
        raise ValueError("r must be between 3 and 16")

    # The leading and constant coefficients must both be 1.
    for middle in range(1 << (r - 1)):
        poly = (1 << r) | (middle << 1) | 1
        if _is_primitive(poly, r):
            return poly

    raise RuntimeError(f"could not find a primitive polynomial of degree {r}")


# ---------------------------------------------------------------------------
# Two-error-correcting binary BCH parity-check matrix
# ---------------------------------------------------------------------------


def _random_row_mix(H, rng):
    """
    Apply random invertible row operations over GF(2).

    This preserves the code but hides the algebraic structure of H.
    """
    H = H.copy()
    m = H.shape[0]

    for _ in range(8 * m * m):
        i, j = rng.choice(m, size=2, replace=False)

        if rng.integers(2):
            # Elementary row addition, invertible over GF(2).
            H[i] ^= H[j]
        else:
            # Row swap.
            H[[i, j]] = H[[j, i]]

    return H


def _balanced_random_row_mix(H, rng, candidates):
    """
    Return the best of several random invertible row mixes.

    Rows with between three and half of the variables have no penalty.
    Outside that interval, the squared distance to the nearest boundary is
    penalized. Equal-scoring candidates are selected uniformly at random.
    """
    if isinstance(candidates, bool) or not isinstance(
        candidates, (int, np.integer)
    ):
        raise TypeError("candidates must be an integer")
    if candidates < 1:
        raise ValueError("candidates must be at least 1")

    variable_count = H.shape[1]
    upper = variable_count // 2
    lower = min(3, upper)

    best = None
    best_score = None
    tied = 0

    for _ in range(candidates):
        mixed = _random_row_mix(H, rng)
        weights = mixed.sum(axis=1, dtype=np.int64)
        clipped = np.clip(weights, lower, upper)
        distance = weights - clipped
        score = int(distance @ distance)

        if best_score is None or score < best_score:
            best = mixed
            best_score = score
            tied = 1
        elif score == best_score:
            tied += 1
            if rng.integers(tied) == 0:
                best = mixed

    return best


def random_bch2_parity_check(variable_count, rng=None, row_mix_candidates=64):
    """
    Construct a randomized, shortened binary two-error-correcting BCH matrix.

    Parameters
    ----------
    variable_count : int
        Number of columns in the matrix. The field degree is the smallest r
        for which variable_count <= 2**r - 1 (and at least 3). The matrix has
        2*r rows.

        Practical puzzle sizes include:
            2--7 variables:   6 checks
            8--15 variables:  8 checks
            16--31 variables: 10 checks

    rng : None, int, or numpy.random.Generator
        Random-number source.

    row_mix_candidates : int
        Number of random invertible row mixes from which to select the most
        balanced one. Must be at least 1.

    Returns
    -------
    H : ndarray of shape (2*r, variable_count), dtype uint8
        A binary parity-check matrix capable of uniquely locating any
        pair of erroneous coordinates.
    """
    if isinstance(variable_count, bool) or not isinstance(
        variable_count, (int, np.integer)
    ):
        raise TypeError("variable_count must be an integer")
    if not 2 <= variable_count <= (1 << 16) - 1:
        raise ValueError("variable_count must be between 2 and 65535")

    r = max(3, int(variable_count).bit_length())

    rng = np.random.default_rng(rng)

    primitive_poly = find_primitive_polynomial(r)
    n = (1 << r) - 1
    mask = n

    # powers[i] represents alpha**i in the polynomial basis.
    powers = np.empty(n, dtype=np.uint64)
    powers[0] = 1

    for i in range(1, n):
        value = int(powers[i - 1]) << 1

        # Reduce x^r using the primitive polynomial.
        if value & (1 << r):
            value ^= primitive_poly

        powers[i] = value & mask

    H = np.empty((2 * r, n), dtype=np.uint8)

    for i in range(n):
        x = int(powers[i])
        x_cubed = int(powers[(3 * i) % n])

        # First syndrome component: x
        for bit in range(r):
            H[bit, i] = (x >> bit) & 1

        # Second syndrome component: x^3
        for bit in range(r):
            H[r + bit, i] = (x_cubed >> bit) & 1

    # Randomly choose the retained coordinates. Deleting columns this way
    # shortens the code; distinct two-error syndromes remain distinct.
    H = H[:, rng.permutation(n)[:variable_count]]

    # Randomize the parity equations themselves, preferring mixes without
    # very short or unusually long equations.
    H = _balanced_random_row_mix(H, rng, row_mix_candidates)

    return H


# ---------------------------------------------------------------------------
# Puzzle generation and rendering
# ---------------------------------------------------------------------------


def _gf2_matvec(H, values):
    """
    Compute H @ values over GF(2) without relying on integer overflow.
    """
    return np.bitwise_xor.reduce(H & values[None, :], axis=1)


def make_two_error_puzzle(H, rng=None):
    """
    Create a puzzle in which exactly two displayed assignments are wrong.
    """
    H = np.asarray(H, dtype=np.uint8)
    rng = np.random.default_rng(rng)

    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix")
    if np.any((H != 0) & (H != 1)):
        raise ValueError("H must be binary")

    m, n = H.shape
    if n < 2:
        raise ValueError("H must have at least two columns")

    correct = rng.integers(0, 2, size=n, dtype=np.uint8)
    rhs = _gf2_matvec(H, correct)

    error_indices = np.sort(rng.choice(n, size=2, replace=False))

    displayed = correct.copy()
    displayed[error_indices] ^= 1

    return correct, displayed, rhs, error_indices


def bch_to_logic(H, correct, displayed, rhs, number_wrong=2, phrasing='exactly'):
    H = np.asarray(H, dtype=np.uint8)
    displayed = np.asarray(displayed, dtype=np.uint8)
    rhs = np.asarray(rhs, dtype=np.uint8)

    if H.ndim != 2:
        raise ValueError("H must be two-dimensional")

    m, n = H.shape

    if correct.shape != (n,):
        raise ValueError("correct must have one value per column of H")
    if displayed.shape != (n,):
        raise ValueError("displayed must have one value per column of H")
    if rhs.shape != (m,):
        raise ValueError("rhs must have one value per row of H")

    def variable_name(i):
        letter = chr(ord("a") + i % 26)
        suffix = "" if i < 26 else str(i // 26)
        return letter + suffix

    names = [variable_name(i) for i in range(n)]

    assignments = ", ".join(
        f"{name} = {bool(value)}" for name, value in zip(names, (correct if phrasing == "alternative" else displayed))
    )

    checks = []
    for row, expected in zip(H, rhs):
        variables = [names[i] for i, included in enumerate(row) if included]
        expression = " xor ".join(variables) if variables else "False"
        checks.append(f"{expression} = {bool(expected)}")

    word = "expression" if number_wrong == 1 else "expressions"
    if phrasing == 'exactly':
        phrase = f"Exactly {number_wrong} of these {word} are incorrect:\n"
    elif phrasing == 'some':
        phrase = "Some of these expressions are incorrect:\n"
    elif phrasing == 'ignore':
        phrase = "These expressions may be incorrect and should be ignored:\n"
    elif phrasing == "boolean":
        phrase = "All variables are boolean.\n"
    elif phrasing == "not":
        phrase = "This is not a valid solution:\n"
    elif phrasing == "alternative":
        phrase = "This is a valid solution:\n"
    else:
        raise ValueError("invalid phrasing")

    problem = "The following expressions are all correct:\n" + "\n".join(checks)

    # if phrasing == "not":
    #     return (
    #         problem + "\n\n" +
    #         phrase + f"{assignments}\n\n" +
    #         f"What is the correct solution ({', '.join(names)})?"
    #     )
    if phrasing == "alternative":
        return (
            problem + "\n\n" +
            phrase + f"{assignments}\n\n" +
            f"What is the alternative solution ({', '.join(names)})?"
        )
    else:
        return (
            phrase + (f"{assignments}\n\n" if phrasing != "boolean" else "\n") +
            problem + "\n\n"
            f"What is the value of ({', '.join(names)})?"
        )


def brute_force_two_error_solutions(H, displayed, rhs):
    """
    Find every solution satisfying both:

    1. All parity equations H @ candidate = rhs over GF(2).
    2. candidate differs from displayed in exactly two positions.

    Returns
    -------
    solutions : list of (error_pair, candidate)
        error_pair is a tuple (i, j).
        candidate is the corrected binary assignment.
    """
    H = np.asarray(H, dtype=np.uint8)
    displayed = np.asarray(displayed, dtype=np.uint8)
    rhs = np.asarray(rhs, dtype=np.uint8)

    if H.ndim != 2:
        raise ValueError("H must be two-dimensional")

    m, n = H.shape

    if displayed.shape != (n,):
        raise ValueError("displayed must have one value per column of H")
    if rhs.shape != (m,):
        raise ValueError("rhs must have one value per row of H")

    solutions = []

    for i, j in itertools.combinations(range(n), 2):
        candidate = displayed.copy()
        candidate[i] ^= 1
        candidate[j] ^= 1

        if np.array_equal(_gf2_matvec(H, candidate), rhs):
            solutions.append(((i, j), candidate))

    return solutions


def run_test(llm, args, variables):

    rng = np.random.default_rng()

    H = random_bch2_parity_check(
        variables, rng, row_mix_candidates=args.row_mix_candidates
    )
    corig, cerr, rhs, error_indices = make_two_error_puzzle(H, rng)
    logic = bch_to_logic(H, corig, cerr, rhs, number_wrong=2, phrasing=args.phrasing)

    solutions = brute_force_two_error_solutions(H, cerr, rhs)
    assert len(solutions) == 1
    assert np.array_equal(solutions[0][1], corig)

    lines = []

    lines.append("Answer immediately with one tuple and nothing else.")
    lines.append("")
    lines.append(logic)

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
            except Exception as e:
                print(e)
                print("Retrying in 10 seconds ...")
                time.sleep(10)

        print(response)
        print()

        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            if len(response.content) != 1:
                content = "(incorrect length list)"
            elif "text" not in response.content[0]:
                content = "(non-text response)"
            else:
                content = response.content[0]["text"]
        else:
            content = "(unknown type)"

        if "output_token_details" in response.usage_metadata:
            reasoning_tokens = response.usage_metadata["output_token_details"][
                "reasoning"
            ]
        else:
            reasoning_tokens = 0
        reasoning_tokens += max(0, response.usage_metadata["output_tokens"] - 80)

    try:
        correct = False
        alternative = False
        tmp = content.strip().removeprefix("(").removesuffix(")")
        parts = tmp.split(",")
        if len(parts) == len(corig):
            if all(parse_bool(x) == bool(y) for x, y in zip(parts, corig)):
                correct = True
            candidate = np.array([parse_bool(x) for x in parts])
            alternative = np.array_equal(_gf2_matvec(H, candidate), rhs)
            if correct:
                assert alternative
    except ValueError:
        correct = False

    valid = reasoning_tokens == 0

    print(f"Model answer: {content}")
    print(f"Correct answer: {tuple(bool(y) for y in corig)}")
    print(f"Reasoning tokens: {reasoning_tokens}")
    print(("CORRECT" if correct else "ALTERNATIVE" if alternative else "WRONG") if valid else "INVALID")
    print()

    return (correct, valid, alternative)


def run_sample(llm, args, variables):

    test = BernoulliSPRT(max_trials=args.repeats)

    score = 0
    score_alt = 0
    invalid = 0
    total = 0
    for rep in range(args.repeats):
        (correct, valid, alternative) = run_test(llm=llm, args=args, variables=variables)
        score += int(correct and valid)
        score_alt += int(alternative and not correct and valid)
        invalid += int(not valid)
        total += 1
        if args.horizon:
            status = test.step(correct and valid)
            if status != "CONTINUE":
                print(test.status)
                break

    print(f"Provider:          {args.provider}")
    print(f"Model:             {args.model}")
    print(f"Variables:         {variables}")
    print(f"Input dots:        {args.input_dots}")
    print(f"Total correct:     {score} / {total}  ({100 * score // total}%)")
    print(f"Total alternative: {score_alt} / {total}  ({100 * score_alt // total}%)")
    print(f"Total invalid:     {invalid} / {total}  ({100 * invalid // total}%)")

    return (score, invalid, total)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a long Python calculation exercise."
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=int,
        default=4,
        help="number of variables, between 2 and 21",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=50,
        help="number of times the test is repeated",
    )
    parser.add_argument(
        "--row-mix-candidates",
        type=int,
        default=64,
        help="number of randomized parity-check mixes to score (default: 64)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openai",
        help="model provider (via LangChain)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5.6-luna",
        help="model name (via LangChain)",
    )
    parser.add_argument(
        "--phrasing",
        type=str,
        default="exactly",
        choices=["exactly", "some", "ignore", "boolean", "not", "alternative"],
        help="change question phrasing to 'some of these are incorrect'",
    )
    parser.add_argument(
        "--horizon",
        action="store_true",
        help="calculate success horizon using binary search over variables",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without doing LLM calls (response is always empty)",
    )
    parser.add_argument(
        "--input-dots",
        type=int,
        default=None,
        help="add input dots (filler tokens)",
    )
    parser.add_argument("--lo", type=int, default=None)
    parser.add_argument("--hi", type=int, default=None)

    args = parser.parse_args()

    kwargs = dict()
    if args.model in ("gpt-4", "gpt-4.1", "claude-sonnet-4-5", "claude-haiku-4-5"):
        pass  # non-reasoning model or no effort parameter
    elif args.model in ("gpt-5", "gpt-5-mini", "gpt-5-nano"):
        kwargs["reasoning_effort"] = "minimal"
    elif args.model in (
        "gpt-6-astra",
        "gpt-6.1-sol",
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

    if args.dry_run:
        llm = None
    else:
        llm = init_chat_model(
            model_provider=args.provider,
            model=args.model,
            max_tokens=100,
            **kwargs,
        )

    if args.horizon:
        if args.lo is not None and args.hi is not None:
            lo, hi = args.lo, args.hi
        else:
            # One variable cannot contain the two errors required by the puzzle,
            # so begin at the smallest supported variable count.
            lo = 1
            hi = 2
            while True:
                (score, invalid, total) = run_sample(
                    llm=llm, args=args, variables=hi
                )
                if score * 2 < total:
                    break
                if hi >= 256:
                    lo = hi
                    break
                lo = hi
                hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            (score, invalid, total) = run_sample(
                llm=llm, args=args, variables=mid
            )
            if score * 2 < total:
                hi = mid
            else:
                lo = mid
        print()
        print(f"Success horizon: {lo} variables")
    else:
        run_sample(llm=llm, args=args, variables=args.variables)


if __name__ == "__main__":
    main()

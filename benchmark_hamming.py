import argparse
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
        self.llr_failure = math.log(
            (1.0 - p_high) / (1.0 - p_low)
        )

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
        elif (
            self.max_trials is not None
            and self.trials >= self.max_trials
        ):
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


def random_hamming_parity_check(
    r: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random binary parity-check matrix for a Hamming code.

    The resulting code has parameters:
        n = 2**r - 1
        k = 2**r - r - 1

    Returns
    -------
    H : ndarray of shape (r, n), dtype uint8
        A parity-check matrix over GF(2). Codewords c satisfy
        (H @ c) % 2 == 0.

    Notes
    -----
    A binary Hamming parity-check matrix contains every nonzero r-bit
    column exactly once. Randomness is therefore introduced by choosing
    a uniformly random ordering of those columns.
    """
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)):
        raise TypeError("r must be an integer")
    if not 1 <= r <= 63:
        raise ValueError("r must be between 1 and 63")

    n = (1 << r) - 1

    # Every nonzero r-bit vector, represented as a column.
    values = np.arange(1, 1 << r, dtype=np.uint64)
    bit_positions = np.arange(r, dtype=np.uint64)[:, None]
    H = ((values[None, :] >> bit_positions) & 1).astype(np.uint8)

    # A random coordinate permutation gives a random non-systematic matrix.
    return H[:, rng.permutation(n)]


def make_random_puzzle(H, rng=None):
    H = np.asarray(H, dtype=np.uint8)
    rng = np.random.default_rng(rng)

    m, n = H.shape

    # Start with a random hidden assignment.
    correct = rng.integers(0, 2, size=n, dtype=np.uint8)

    # The corresponding right-hand sides for H @ correct = rhs (mod 2).
    rhs = (H @ correct) % 2

    # Flip exactly one assignment to produce the displayed erroneous values.
    error_index = rng.integers(n)
    cerr = correct.copy()
    cerr[error_index] ^= 1

    return correct, cerr, rhs, error_index


def hamming_to_logic(H, cerr, rhs):
    H = np.asarray(H, dtype=np.uint8)
    cerr = np.asarray(cerr, dtype=np.uint8)
    rhs = np.asarray(rhs, dtype=np.uint8)

    m, n = H.shape

    if cerr.shape != (n,):
        raise ValueError("cerr must have one value per column of H")
    if rhs.shape != (m,):
        raise ValueError("rhs must have one value per row of H")

    def variable_name(i):
        letter = chr(ord("a") + i % 26)
        suffix = "" if i < 26 else str(i // 26)
        return letter + suffix

    names = [variable_name(i) for i in range(n)]

    assignments = ", ".join(
        f"{name} = {bool(value)}"
        for name, value in zip(names, cerr)
    )

    checks = []
    for row, expected in zip(H, rhs):
        variables = [names[i] for i, included in enumerate(row) if included]
        expression = " xor ".join(variables) if variables else "False"
        checks.append(f"{expression} = {bool(expected)}")

    return (
        "Exactly one of these expressions is incorrect:\n"
        f"{assignments}\n\n"
        "The following expressions are all correct:\n"
        + "\n".join(checks)
        + "\n\n"
        f"What is the value of ({', '.join(names)})?"
    )


def run_test(llm, args):

    rng = np.random.default_rng()

    H = random_hamming_parity_check(r=args.hamming, rng=rng)
    corig, cerr, rhs, error_index = make_random_puzzle(H=H, rng=rng)

    assert np.all((H @ corig) % 2 == rhs)
    assert np.any((H @ cerr) % 2 != rhs)

    logic = hamming_to_logic(H, cerr, rhs)

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
                print('Retrying in 10 seconds ...')
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
            reasoning_tokens = response.usage_metadata["output_token_details"]["reasoning"]
        else:
            reasoning_tokens = 0
        reasoning_tokens += max(0, response.usage_metadata["output_tokens"] - 80)

    try:
        correct = False
        tmp = content.strip().removeprefix("(").removesuffix(")")
        parts = tmp.split(',')
        if len(parts) == len(corig):
            if all(parse_bool(x) == bool(y) for x, y in zip(parts, corig)):
                correct = True
    except ValueError:
        correct = False

    valid = reasoning_tokens == 0

    print(f"Model answer: {content}")
    print(f"Correct answer: {tuple(bool(y) for y in corig)}")
    print(f"Reasoning tokens: {reasoning_tokens}")
    print(("CORRECT" if correct else "WRONG") if valid else "INVALID")
    print()

    return (correct, valid)


def run_sample(llm, args):

    test = BernoulliSPRT(max_trials=args.repeats)

    score = 0
    invalid = 0
    total = 0
    for rep in range(args.repeats):
        (correct, valid) = run_test(llm=llm, args=args)
        score += int(correct and valid)
        invalid += int(not valid)
        total += 1

    print(f"Provider:      {args.provider}")
    print(f"Model:         {args.model}")
    print(f"Total score:   {score} / {total}  ({100 * score // total}%)")
    print(
        f"Total invalid: {invalid} / {total}  ({100 * invalid // total}%)"
    )

    return (score, invalid, total)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a long Python calculation exercise."
    )
    parser.add_argument(
        "--hamming",
        type=int,
        default=3,
        help="Hamming `r` parameter",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=50,
        help="number of times the test is repeated",
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

    args = parser.parse_args()

    kwargs = dict()
    if args.model in ("gpt-4", "gpt-4.1", "claude-sonnet-4-5", "claude-haiku-4-5"):
        pass  # non-reasoning model or no effort parameter
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

    if args.dry_run:
        llm = None
    else:
        llm = init_chat_model(
            model_provider=args.provider,
            model=args.model,
            max_tokens=100,
            **kwargs,
        )

    run_sample(llm=llm, args=args)


if __name__ == "__main__":
    main()

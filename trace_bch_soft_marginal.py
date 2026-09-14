"""Trace Astra's marginal estimates as dot padding grows.

The selectable puzzles come from trusted local benchmark outputs. Only load a
replacement pickle file if you trust it: loading pickle files can execute
arbitrary code.
"""

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TEST_CASES = {
    "original": {
        "kind": "pickle",
        "path": WORKSPACE_ROOT
        / "rawdump-bch-soft-v3"
        / "astra-v11-r100-300dots-lora5.pkl",
        "puzzle_index": 71,
        "prompt_alignment_dots": 100,
        "expected_initial_input_tokens": 384,
    },
    "3000dots-six-flips": {
        "kind": "text",
        "path": WORKSPACE_ROOT / "rawdump-bch-soft-v2" / "astra-horizon-3000dots.txt",
        "puzzle_block_index": 39,
        "line_number": 1405,
        "input_dots": 3000,
        "prompt_alignment_dots": 74,
        "expected_initial_input_tokens": 384,
    },
    "1000dots-three-flips": {
        "kind": "text",
        "path": WORKSPACE_ROOT / "rawdump-bch-soft-v2" / "astra-horizon-1000dots.txt",
        "puzzle_block_index": 7,
        "line_number": 267,
        "input_dots": 1000,
        "prompt_alignment_dots": 40,
        "expected_initial_input_tokens": 384,
    },
    "300dots-two-flips": {
        "kind": "text",
        "path": WORKSPACE_ROOT / "rawdump-bch-soft-v2" / "astra-horizon-300dots.txt",
        "puzzle_block_index": 116,
        "line_number": 4059,
        "input_dots": 300,
        "prompt_alignment_dots": 76,
        "expected_initial_input_tokens": 384,
    },
}
TEXT_PUZZLE_MARKER = "Answer immediately with one tuple and nothing else.\n\n"
CACHE_MINIMUM_TOKENS = 1024
CACHE_REPORTING_GRANULARITY = 128
EXPECTED_MESSAGE_FRAMING_TOKENS = 4
EXPECTED_INPUT_TOKEN_INCREMENT = 128
DEFAULT_OUTPUT = "trace-bch-soft-marginal.pkl"

MARGINAL_SYSTEM_PROMPT = (
    "Do not reason, deliberate, or work anything out. Emit the final answer on your "
    "very first token. No explanation, no steps, no scratch work. Return only the "
    "requested tuple of decimal probabilities and nothing else."
)
INFORMAL_SYSTEM_PROMPT = (
    "Do not reason, deliberate, or work anything out. Emit the final answer on your "
    "very first token. No explanation, no steps, no scratch work. Return only the "
    "requested tuple of percentages and nothing else."
)
ORIGINAL_SYSTEM_PROMPT = (
    "Do not reason, deliberate, or work anything out. Emit the final answer on your "
    "very first token. No explanation, no steps, no scratch work."
)


def _exact_solution_scores(problem):
    matrix = np.asarray(problem["matrix"], dtype=np.uint8)
    rhs = np.asarray(problem["rhs"], dtype=np.uint8)
    probabilities = np.asarray(problem["percentages"], dtype=np.float64) / 100.0
    variable_count = matrix.shape[1]
    assignments = np.arange(1 << variable_count, dtype=np.uint64)[:, None]
    bits = ((assignments >> np.arange(variable_count, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    satisfies = np.all((bits @ matrix.T) % 2 == rhs, axis=1)
    solutions = bits[satisfies]
    if len(solutions) == 0:
        raise ValueError("selected puzzle has no satisfying assignment")
    log_false = np.log1p(-probabilities)
    log_scores = solutions @ (np.log(probabilities) - log_false) + log_false.sum()
    return solutions, log_scores


def _exact_marginals(problem):
    """Enumerate the posterior and return exact P(True) for every variable."""
    solutions, log_scores = _exact_solution_scores(problem)
    weights = np.exp(log_scores - log_scores.max())
    return (weights @ solutions) / weights.sum()


def _parse_bool_tuple(text):
    values = text.strip().removeprefix("(").removesuffix(")").split(",")
    parsed = []
    for value in values:
        value = value.strip()
        if value not in ("True", "False"):
            raise ValueError(f"invalid Boolean tuple value: {value!r}")
        parsed.append(value == "True")
    return tuple(parsed)


def _load_pickle_puzzle(test_case):
    source_path = test_case["path"]
    puzzle_index = test_case["puzzle_index"]
    with open(source_path, "rb") as dump_file:
        payload = pickle.load(dump_file)
    if payload.get("format") != "benchmark_bch_soft" or payload.get("version") != 1:
        raise ValueError(f"unsupported source dump: {source_path}")
    problem = payload["problems"][puzzle_index]
    if not problem["valid"] or not problem["correct"]:
        raise ValueError("selected puzzle was not valid and correctly solved")
    source = {
        "pickle_file": str(source_path),
        "puzzle_index": puzzle_index,
    }
    return problem, source


def _load_text_puzzle(test_case):
    source_path = test_case["path"]
    block_index = test_case["puzzle_block_index"]
    blocks = source_path.read_text().split(TEXT_PUZZLE_MARKER)[1:]
    if block_index >= len(blocks):
        raise ValueError(f"puzzle block {block_index} is absent from {source_path}")
    block = blocks[block_index]

    logic_start = block.find("All variables are boolean.")
    question_match = re.search(r"What is the most likely value of \([^\n]+\)\?", block)
    if logic_start < 0 or question_match is None:
        raise ValueError(f"could not parse puzzle block {block_index} in {source_path}")
    logic = block[logic_start : question_match.end()]

    prior_matches = re.findall(
        r"^([a-z]) = (\d+)% chance of being True$", logic, flags=re.MULTILINE
    )
    variable_names = [name for name, _ in prior_matches]
    expected_names = [chr(ord("a") + index) for index in range(len(prior_matches))]
    if variable_names != expected_names:
        raise ValueError("the text puzzle does not define consecutive variables")
    percentages = np.asarray(
        [int(percentage) for _, percentage in prior_matches], dtype=np.int64
    )

    expression_marker = "The following expressions are all correct:\n"
    expression_start = logic.find(expression_marker)
    if expression_start < 0:
        raise ValueError("the text puzzle has no expression block")
    expression_text = logic[
        expression_start + len(expression_marker) : question_match.start() - logic_start
    ].strip()
    expressions = expression_text.splitlines()
    matrix = np.zeros((len(expressions), len(percentages)), dtype=np.uint8)
    rhs = np.zeros(len(expressions), dtype=np.uint8)
    for row_index, expression in enumerate(expressions):
        left, separator, right = expression.partition(" = ")
        if not separator or right not in ("True", "False"):
            raise ValueError(f"invalid expression in text puzzle: {expression!r}")
        target = right == "True"
        for term in left.split(" xor "):
            if term == "True":
                target = not target
            elif term != "False":
                if term not in variable_names:
                    raise ValueError(f"unknown expression term: {term!r}")
                matrix[row_index, ord(term) - ord("a")] ^= 1
        rhs[row_index] = target

    outcome_match = re.search(
        r"Model answer: (?P<model>[^\n]+)\n"
        r"Correct answer: (?P<correct>[^\n]+)\n"
        r"Best/second probability ratio: (?P<ratio>[^\n]+)\n"
        r"Reasoning tokens: (?P<reasoning>\d+)\n"
        r"(?P<outcome>CORRECT|ALTERNATIVE|WRONG|INVALID)",
        block,
    )
    if outcome_match is None:
        raise ValueError("could not parse the text puzzle's model outcome")
    model_answer = _parse_bool_tuple(outcome_match.group("model"))
    correct_answer = _parse_bool_tuple(outcome_match.group("correct"))
    if outcome_match.group("outcome") != "CORRECT" or model_answer != correct_answer:
        raise ValueError("selected text puzzle was not correctly solved")
    if int(outcome_match.group("reasoning")) != 0:
        raise ValueError("selected text puzzle used reasoning tokens")

    problem = {
        "variables": len(percentages),
        "matrix": matrix,
        "rhs": rhs,
        "percentages": percentages,
        "correct_answer": correct_answer,
        "probability_ratio": float(outcome_match.group("ratio")),
        "logic": logic,
    }
    source = {
        "text_file": str(source_path),
        "puzzle_block_index": block_index,
        "line_number": test_case["line_number"],
        "input_dots": test_case["input_dots"],
        "correctable_errors": 2,
    }
    return problem, source


def _load_selected_puzzle(test_case_name):
    test_case = TEST_CASES[test_case_name]
    if test_case["kind"] == "pickle":
        problem, source = _load_pickle_puzzle(test_case)
    else:
        problem, source = _load_text_puzzle(test_case)
    source["test_case"] = test_case_name
    return problem, source


def _problem_prompt(problem, phrasing):
    question_marker = "\n\nWhat is the most likely value"
    logic, separator, _ = problem["logic"].partition(question_marker)
    if not separator:
        raise ValueError("could not remove the MAP question from the stored puzzle")
    variable_names = [chr(ord("a") + index) for index in range(problem["variables"])]
    names = ", ".join(variable_names)
    if phrasing == "marginal":
        return (
            logic
            + "\n\nReport the exact marginal probability P(variable=True), conditioned "
            "on all expressions above, for each variable in this order: "
            f"({names}). Return exactly one tuple of "
            f"{len(variable_names)} decimal numbers between 0 and 1."
        )
    if phrasing == "informal":
        return (
            logic
            + "\n\nGiven that all expressions above are correct, what is the probability "
            f"that each of ({names}) is True? Return exactly one tuple of "
            f"{len(variable_names)} percentage values between 0% and 100%, in "
            "that order."
        )
    if phrasing == "original":
        return (
            "Answer immediately with one tuple and nothing else.\n\n" + problem["logic"]
        )
    if phrasing == "confidence":
        return (
            "Answer immediately with two tuples and nothing else.\n\n"
            + problem["logic"]
            + "\n\nAfter the solution tuple, return a second tuple containing your "
            "confidence that each corresponding value in the solution tuple is "
            f"correct, as {len(variable_names)} percentage values between 0% and "
            "100%."
        )
    raise ValueError(f"unsupported phrasing: {phrasing!r}")


def _system_prompt(phrasing):
    if phrasing == "marginal":
        return MARGINAL_SYSTEM_PROMPT
    if phrasing == "informal":
        return INFORMAL_SYSTEM_PROMPT
    return ORIGINAL_SYSTEM_PROMPT


def _dot_message(dot_count):
    """Return N dots that tokenize to N tokens in o200k_base/cl100k_base."""
    return " ".join("." for _ in range(dot_count))


def _response_text(response):
    if isinstance(response.content, str):
        return response.content
    if (
        isinstance(response.content, list)
        and len(response.content) == 1
        and isinstance(response.content[0], dict)
        and "text" in response.content[0]
    ):
        return response.content[0]["text"]
    return "(unrecognized response)"


def _parse_marginals(content, variable_count):
    text = content.strip().removeprefix("(").removesuffix(")")
    parts = text.split(",")
    if len(parts) != variable_count:
        raise ValueError(f"expected {variable_count} comma-separated probabilities")
    values = np.asarray([float(part.strip()) for part in parts], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and between 0 and 1")
    return values


def _parse_percentages(content, variable_count):
    text = content.strip().removeprefix("(").removesuffix(")")
    parts = text.split(",")
    if len(parts) != variable_count:
        raise ValueError(f"expected {variable_count} comma-separated percentages")
    values = np.asarray(
        [float(part.strip().removesuffix("%").strip()) for part in parts],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("percentages must be finite and between 0 and 100")
    return values / 100.0


def _parse_reported_marginals(content, variable_count, phrasing):
    if phrasing == "marginal":
        return _parse_marginals(content, variable_count)
    if phrasing == "informal":
        return _parse_percentages(content, variable_count)
    if phrasing == "original":
        solution = _parse_bool_tuple(content)
        if len(solution) != variable_count:
            raise ValueError(f"expected {variable_count} Boolean values")
        return np.asarray(solution, dtype=np.float64)
    if phrasing != "confidence":
        raise ValueError(f"unsupported phrasing: {phrasing!r}")

    match = re.fullmatch(r"\s*(\([^()]*\))\s*,?\s*(\([^()]*\))\s*", content)
    if match is None:
        raise ValueError("expected a Boolean tuple followed by a confidence tuple")
    solution = _parse_bool_tuple(match.group(1))
    if len(solution) != variable_count:
        raise ValueError(f"expected {variable_count} Boolean values")
    confidences = _parse_percentages(match.group(2), variable_count)
    return np.where(solution, confidences, 1.0 - confidences)


def _usage_detail(input_details, name):
    direct = input_details.get(name)
    if direct is not None:
        return int(direct)
    tier_values = [
        int(value)
        for key, value in input_details.items()
        if key.endswith(f"_{name}") and value is not None
    ]
    return sum(tier_values)


def _cache_validation(previous_input_tokens, cache_read_tokens):
    """Return cache validation metadata, raising for an eligible cache miss."""
    if previous_input_tokens < CACHE_MINIMUM_TOKENS:
        return {
            "status": "previous_prompt_below_cache_minimum",
            "required_cached_tokens": 0,
            "valid": None,
        }
    required = (
        previous_input_tokens // CACHE_REPORTING_GRANULARITY
    ) * CACHE_REPORTING_GRANULARITY
    valid = cache_read_tokens >= required
    result = {
        "status": "passed" if valid else "failed",
        "required_cached_tokens": required,
        "valid": valid,
    }
    if not valid:
        raise RuntimeError(
            "prompt-cache validation failed: the previous request used "
            f"{previous_input_tokens} input tokens, so at least {required} cached "
            f"tokens were required, but the API reported {cache_read_tokens}"
        )
    return result


def _save_trace(filename, payload):
    path = Path(filename)
    with open(path, "wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def _model_kwargs(model):
    if model in ("gpt-4", "gpt-4.1", "claude-sonnet-4-5", "claude-haiku-4-5"):
        return {}
    if model in ("gpt-5", "gpt-5-mini", "gpt-5-nano"):
        return {"reasoning_effort": "minimal"}
    if model.startswith(("gpt-6", "claude-")):
        return {"reasoning_effort": "low"}
    return {"reasoning_effort": "none"}


def main():
    parser = argparse.ArgumentParser(
        description="Trace marginal estimates for a selected BCH-soft puzzle."
    )
    parser.add_argument("-p", "--provider", default="openai")
    parser.add_argument("-m", "--model", default="gpt-6-astra")
    parser.add_argument(
        "--phrasing",
        choices=("marginal", "informal", "original", "confidence"),
        default="marginal",
        help="prompt/output phrasing to use (default: marginal)",
    )
    parser.add_argument(
        "--test-case",
        choices=TEST_CASES,
        default="original",
        help="puzzle to trace (default: original)",
    )
    parser.add_argument(
        "--max-dots",
        type=int,
        default=2048,
        help="largest experimental dot count to test (default: 2048)",
    )
    parser.add_argument(
        "--dot-step",
        type=int,
        default=124,
        help="dot tokens per new message; 124 plus framing targets 128 (default: 124)",
    )
    parser.add_argument(
        "--prompt-alignment-dots",
        type=int,
        default=None,
        help="dots padding the fixed prompt (default: calibrated per test case)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"pickle output file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--prompt-cache-key",
        default=None,
        help="stable prompt-cache routing/accounting key (default: prompt-specific)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_dots < 0:
        parser.error("--max-dots must be non-negative")
    if args.dot_step < 1:
        parser.error("--dot-step must be positive")
    if args.prompt_alignment_dots is not None and args.prompt_alignment_dots < 0:
        parser.error("--prompt-alignment-dots must be non-negative")

    test_case = TEST_CASES[args.test_case]
    if args.prompt_alignment_dots is None:
        args.prompt_alignment_dots = test_case["prompt_alignment_dots"]
    if args.prompt_cache_key is None:
        args.prompt_cache_key = f"trace-bch-soft-marginal-{args.test_case}-v1"
        if args.phrasing != "marginal":
            args.prompt_cache_key = (
                f"trace-bch-soft-marginal-{args.test_case}-{args.phrasing}-v1"
            )

    problem, source = _load_selected_puzzle(args.test_case)
    exact_marginals = _exact_marginals(problem)
    prior_probabilities = np.asarray(problem["percentages"], dtype=np.float64) / 100.0
    map_assignment = np.asarray(problem["correct_answer"], dtype=np.uint8)
    solutions, log_scores = _exact_solution_scores(problem)
    exact_map_assignment = solutions[np.argmax(log_scores)]
    if not np.array_equal(map_assignment, exact_map_assignment):
        raise ValueError("logged correct answer does not match the brute-force MAP")
    flip_indices = np.flatnonzero(
        (prior_probabilities - 0.5) * (exact_marginals - 0.5) < 0.0
    )
    unpadded_prompt = _problem_prompt(problem, args.phrasing)
    prompt = unpadded_prompt
    if args.prompt_alignment_dots:
        prompt += "\n\n" + _dot_message(args.prompt_alignment_dots)
    system_prompt = _system_prompt(args.phrasing)
    messages = [SystemMessage(system_prompt), HumanMessage(prompt)]
    expected_initial_input_tokens = (
        test_case["expected_initial_input_tokens"]
        if args.phrasing == "marginal"
        else None
    )

    payload = {
        "format": "trace_bch_soft_marginal",
        "version": 1,
        "source": source,
        "configuration": {
            "provider": args.provider,
            "model": args.model,
            "phrasing": args.phrasing,
            "reasoning_effort": _model_kwargs(args.model).get("reasoning_effort"),
            "max_output_tokens": 200,
            "max_dots": args.max_dots,
            "dot_step": args.dot_step,
            "prompt_alignment_dots": args.prompt_alignment_dots,
            "prompt_cache_key": args.prompt_cache_key,
            "prompt_cache_options": {"mode": "implicit"},
            "cache_minimum_tokens": CACHE_MINIMUM_TOKENS,
            "cache_reporting_granularity": CACHE_REPORTING_GRANULARITY,
            "expected_initial_input_tokens_with_default_padding": (
                expected_initial_input_tokens
            ),
            "expected_message_framing_tokens": EXPECTED_MESSAGE_FRAMING_TOKENS,
            "expected_input_token_increment": EXPECTED_INPUT_TOKEN_INCREMENT,
            "tokenizer_encodings_checked": ["o200k_base", "cl100k_base"],
            "dry_run": args.dry_run,
        },
        "puzzle": {
            "variables": problem["variables"],
            "matrix": np.asarray(problem["matrix"], dtype=np.uint8),
            "rhs": np.asarray(problem["rhs"], dtype=np.uint8),
            "percentages": np.asarray(problem["percentages"]),
            "correct_answer": map_assignment,
            "exact_marginals": exact_marginals,
            "prior_to_marginal_flip_indices": flip_indices,
            "probability_ratio": float(problem["probability_ratio"]),
            "maximum_prior_marginal_difference": float(
                np.max(np.abs(prior_probabilities - exact_marginals))
            ),
            "maximum_marginal_map_difference": float(
                np.max(np.abs(exact_marginals - map_assignment))
            ),
            "logic": problem["logic"],
            "system_prompt": system_prompt,
            "unpadded_trace_prompt": unpadded_prompt,
            "trace_prompt": prompt,
        },
        "traces": [],
    }

    print(f"Test case: {args.test_case}")
    print(f"Phrasing: {args.phrasing}")
    if "pickle_file" in source:
        print(f"Source puzzle: {source['pickle_file']} index {source['puzzle_index']}")
    else:
        print(
            f"Source puzzle: {source['text_file']} block "
            f"{source['puzzle_block_index']} (line {source['line_number']})"
        )
    print(f"Prior-to-marginal flips: {flip_indices.tolist()}")
    print(f"Exact marginals: {tuple(float(value) for value in exact_marginals)}")
    print(f"Initial prompt padding: {args.prompt_alignment_dots} dots")
    if expected_initial_input_tokens is not None:
        print(f"Expected initial total input: {expected_initial_input_tokens} tokens")
    print(
        f"Each subsequent dot message: {args.dot_step} dot tokens + approximately "
        f"{EXPECTED_MESSAGE_FRAMING_TOKENS} framing tokens"
    )
    print(f"Reasoning effort: {_model_kwargs(args.model).get('reasoning_effort')}")
    print(f"Output: {args.output}")
    print()

    llm = None
    if not args.dry_run:
        llm = init_chat_model(
            model_provider=args.provider,
            model=args.model,
            max_tokens=200,
            prompt_cache_options={"mode": "implicit"},
            **_model_kwargs(args.model),
        )

    previous_input_tokens = None
    for request_index, total_dots in enumerate(
        range(0, args.max_dots + 1, args.dot_step)
    ):
        if request_index:
            messages.append(HumanMessage(_dot_message(args.dot_step)))

        started = time.monotonic()
        if args.dry_run:
            content = "(dry run)"
            response_repr = None
            response_metadata = {}
            usage_metadata = {}
            reasoning_tokens = None
            valid_no_reasoning = None
            reported_marginals = None
            parse_error = "dry run"
            cache_validation = {"status": "not_checked", "valid": None}
        else:
            response = llm.invoke(
                messages,
                prompt_cache_key=args.prompt_cache_key,
            )
            print("Full model response:")
            print(response)
            print()
            response_repr = repr(response)
            content = _response_text(response)
            response_metadata = dict(response.response_metadata)
            usage_metadata = dict(response.usage_metadata or {})
            input_details = usage_metadata.get("input_token_details", {})
            output_details = usage_metadata.get("output_token_details", {})
            cache_read_tokens = _usage_detail(input_details, "cache_read")
            reasoning_tokens = int(output_details.get("reasoning", 0) or 0)
            valid_no_reasoning = reasoning_tokens == 0
            if previous_input_tokens is None:
                cache_validation = {"status": "initial_request", "valid": None}
            else:
                try:
                    cache_validation = _cache_validation(
                        previous_input_tokens, cache_read_tokens
                    )
                except RuntimeError as error:
                    cache_validation = {
                        "status": "failed",
                        "required_cached_tokens": (
                            previous_input_tokens // CACHE_REPORTING_GRANULARITY
                        )
                        * CACHE_REPORTING_GRANULARITY,
                        "valid": False,
                        "error": str(error),
                    }
            try:
                reported_marginals = _parse_reported_marginals(
                    content, problem["variables"], args.phrasing
                )
                parse_error = None
            except ValueError as error:
                reported_marginals = None
                parse_error = str(error)

        elapsed_seconds = time.monotonic() - started
        input_tokens = usage_metadata.get("input_tokens")
        input_details = usage_metadata.get("input_token_details", {})
        cache_read_tokens = _usage_detail(input_details, "cache_read")
        record = {
            "request_index": request_index,
            "total_dots": total_dots,
            "total_dots_including_prompt_alignment": (
                total_dots + args.prompt_alignment_dots
            ),
            "dot_message_count": request_index,
            "response_repr": response_repr,
            "raw_response": content,
            "reported_marginals": reported_marginals,
            "parse_error": parse_error,
            "response_metadata": response_metadata,
            "usage_metadata": usage_metadata,
            "reasoning_tokens": reasoning_tokens,
            "valid_no_reasoning": valid_no_reasoning,
            "input_token_alignment_valid": (
                input_tokens % CACHE_REPORTING_GRANULARITY == 0
                if input_tokens is not None
                else None
            ),
            "cache_validation": cache_validation,
            "elapsed_seconds": elapsed_seconds,
        }
        payload["traces"].append(record)
        _save_trace(args.output, payload)

        print(
            f"dots={total_dots}: input_tokens={input_tokens}, "
            f"cached_tokens={cache_read_tokens}, "
            f"cache={cache_validation['status']}, "
            f"reasoning_tokens={reasoning_tokens}"
        )
        if input_tokens is not None and input_tokens % CACHE_REPORTING_GRANULARITY:
            padding_adjustment = (-input_tokens) % CACHE_REPORTING_GRANULARITY
            print(
                f"WARNING: input token count is not 128-aligned; add "
                f"{padding_adjustment} to --prompt-alignment-dots on the next run"
            )
        print(f"response={content}")
        if parse_error is not None:
            print(f"parse_error={parse_error}")
        if valid_no_reasoning is False:
            print("INVALID: response used reasoning tokens")
        print()

        if cache_validation.get("valid") is False:
            print(f"WARNING: {cache_validation['error']}")
            print()
        if input_tokens is not None:
            previous_input_tokens = int(input_tokens)


if __name__ == "__main__":
    main()

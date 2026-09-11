import argparse
import math
import random
import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

OPERATORS = ("and", "or")
# OPERATORS = ("xor",)
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


def apply_operator(left, operator, right):
    if operator == "and":
        return left and right
    if operator == "or":
        return left or right
    if operator == "xor":
        return left ^ right
    raise ValueError(f"Unknown operator: {operator}")


def generate_calculation(
    variable_count=4,
    step_count=10,
    seed=None,
    max_attempts=10000,
):
    if variable_count < 3:
        raise ValueError("At least three variables are required.")

    rng = random.Random(seed)
    variable_names = [chr(ord("a") + i) for i in range(variable_count)]

    for _attempt in range(max_attempts):
        # Give every variable a different initial value.
        initial_values = rng.choices([False, True], k=variable_count)

        values = dict(zip(variable_names, initial_values))

        # A variable may be overwritten only after its current value has
        # been used in an expression. Initially, none have been used.
        used_since_assignment = {name: False for name in variable_names}

        # Tracks whether every variable has been used at least once.
        ever_used = {name: False for name in variable_names}

        operations = []
        failed = False

        previous = (None, None)
        previous_dest = None

        for _step in range(step_count):
            candidates = []

            for left_name in variable_names:
                for right_name in variable_names:
                    if left_name == right_name:
                        continue
                    if left_name in previous and right_name in previous:
                        continue
                    if previous_dest is not None and previous_dest not in (left_name, right_name):
                        continue

                    left_value = values[left_name]
                    right_value = values[right_name]

                    for destination in variable_names:

                        # Keep operations reversible
                        if destination in (left_name, right_name):
                            continue

                        for operator in OPERATORS:
                            result = apply_operator(
                                left_value,
                                operator,
                                right_value,
                            )

                            # Keep every value in the requested range.
                            if result is None:
                                continue

                            candidates.append(
                                (
                                    destination,
                                    left_name,
                                    operator,
                                    right_name,
                                    result,
                                )
                            )

            if not candidates:
                failed = True
                break

            # Prefer expressions that introduce variables not yet used.
            # This helps ensure that no variable is left as an unused result.
            unused_variables = {name for name in variable_names if not ever_used[name]}

            useful_candidates = [
                candidate
                for candidate in candidates
                if candidate[1] in unused_variables or candidate[3] in unused_variables
            ]

            if useful_candidates and rng.random() < 0.75:
                candidates = useful_candidates

            (
                destination,
                left_name,
                operator,
                right_name,
                result,
            ) = rng.choice(candidates)

            values[destination] = values[destination] ^ result

            previous = (left_name, right_name)
            previous_dest = destination

            operations.append(
                (
                    destination,
                    left_name,
                    operator,
                    right_name,
                    tuple(values.values()),
                )
            )

            # The operands were used, but the newly assigned destination
            # value has not yet been used.
            used_since_assignment[left_name] = True
            used_since_assignment[right_name] = True
            used_since_assignment[destination] = False

            ever_used[left_name] = True
            ever_used[right_name] = True

        if failed:
            continue

        # if not all(ever_used.values()):
        #     continue

        return variable_names, initial_values, operations

    raise RuntimeError(
        "Could not generate a calculation. Try fewer steps or variables."
    )


def speculative_depth(
    variable_count, variable_names, initial_values, operations
):
    indices = dict(zip(variable_names, range(variable_count)))
    
    assignments = variable_names + [op[0] for op in operations]
    
    source_map = []
    for _step in range(len(operations)):
        destination, left, operator, right, result = operations[_step]
        for i in reversed(range(variable_count + _step)):
            if assignments[i] == destination:
                dst_source = i
                break
        else:
            raise Exception('Dst source not found')
        for i in reversed(range(variable_count + _step)):
            if assignments[i] == left:
                left_source = i
                break
        else:
            raise Exception('Left source not found')
        for i in reversed(range(variable_count + _step)):
            if assignments[i] == right:
                right_source = i
                break
        else:
            raise Exception('Right source not found')
        source_map.append((dst_source, left_source, right_source))
    
    states = initial_values + [initial_values[indices[op[0]]] for op in operations]
    
    for iter in range(len(operations)):
        
        states2 = states.copy()
        
        for _step in range(len(operations)):
            destination, left, operator, right, result = operations[_step]
            dst_source, left_source, right_source = source_map[_step]
            dst_value = states[dst_source]
            left_value = states[left_source]
            right_value = states[right_source]
            result = apply_operator(
                left_value,
                operator,
                right_value,
            )
            states2[variable_count + _step] = dst_value ^ result
        
        if states == states2:
            break
        states = states2
    else:
        iter = len(operations)
    
    final_result = operations[-1][4]

    for _ind in range(variable_count):
        for i in reversed(range(variable_count + len(operations))):
            if assignments[i] == variable_names[_ind]:
                value = states[i]
                break
        else:
            raise Exception('Final source not found')
        assert value == final_result[_ind], f"Incorrect value {variable_names[_ind]}"

    return iter


def run_test(llm, args, steps):

    while True:
        variable_names, initial_values, operations = generate_calculation(
            variable_count=args.variables,
            step_count=steps,
        )
        sd = speculative_depth(
            variable_count=args.variables,
            variable_names=variable_names,
            initial_values=initial_values,
            operations=operations,
        )
        if args.speculative_depth is None or args.speculative_depth == sd:
            break

    print(f"Variable_names: {repr(variable_names)}")
    print(f"Initial values: {repr(initial_values)}")
    print(f"Operations: {repr(operations)}")
    print(f"Speculative depth: {sd}")
    print()

    lines = []

    lines.append("This task tests your ability to do long sequences of mental math.")
    lines.append(
        "Calculate intermediate results as you are processing the input, then in your response, output a tuple corresponding to the final result."
    )
    lines.append("")

    for name, value in zip(variable_names, initial_values):
        lines.append(f"{name} = {value}")

    lines.append("")
    
    ordered_vars = variable_names.copy()
    
    for destination, left, operator, right, result in operations:
        lines.append(
            f"{destination} = {destination} xor ({left} {operator} {right})"
            f"  # think: {destination} is now ..."
        )
        ordered_vars.remove(destination)
        ordered_vars.append(destination)

    allvars = ", ".join(reversed(ordered_vars))
    expect = tuple(result[variable_names.index(x)] for x in reversed(ordered_vars))

    lines.append("")
    lines.append(f"What is the final value of ({allvars})?")

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
        if len(parts) == len(expect):
            if all(parse_bool(x) == y for x, y in zip(parts, expect)):
                correct = True
    except ValueError:
        correct = False

    valid = reasoning_tokens == 0

    print(f"Model answer: {content}")
    print(f"Correct answer: {expect}")
    print(f"Reasoning tokens: {reasoning_tokens}")
    print(("CORRECT" if correct else "WRONG") if valid else "INVALID")
    print()

    return (correct, valid)


def run_sample(llm, args, steps):

    test = BernoulliSPRT(max_trials=args.repeats)

    score = 0
    invalid = 0
    total = 0
    for rep in range(args.repeats):
        (correct, valid) = run_test(llm=llm, args=args, steps=steps)
        score += int(correct and valid)
        invalid += int(not valid)
        total += 1
        if args.horizon:
            status = test.step(correct and valid)
            if status != "CONTINUE":
                print(test.status)
                break

    print(f"Provider:      {args.provider}")
    print(f"Model:         {args.model}")
    print(f"Variables:     {args.variables}")
    print(f"Steps:         {steps}")
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
        "-v",
        "--variables",
        type=int,
        default=4,
        help="number of variables, between 2 and 21",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=10,
        help="number of calculation steps",
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
        "--horizon",
        action="store_true",
        help="calculate success horizon using binary search",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without doing LLM calls (response is always empty)",
    )
    parser.add_argument(
        "--speculative-depth",
        type=int,
        default=None,
        help="require a specific speculative depth"
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("steps must be at least 1")

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

    if args.horizon:
        lo = 0
        hi = 1
        while True:
            (score, invalid, total) = run_sample(llm=llm, args=args, steps=hi)
            if score * 2 < total:
                break
            if hi >= 256:
                lo = hi
                break
            lo = hi
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            (score, invalid, total) = run_sample(llm=llm, args=args, steps=mid)
            if score * 2 < total:
                hi = mid
            else:
                lo = mid
        print()
        print(f"Success horizon: {lo} steps")
    else:
        run_sample(llm=llm, args=args, steps=args.steps)


if __name__ == "__main__":
    main()

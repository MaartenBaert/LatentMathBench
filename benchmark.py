import argparse
import random

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

MIN_VALUE = 1
MAX_VALUE = 9
OPERATORS = ("+", "-", "*", "//")
SYSTEM_PROMPT = "Do not reason, deliberate, or work anything out. Emit the final answer on your very first token. No explanation, no steps, no scratch work."


def apply_operator(left, operator, right):
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "//":
        if right == 0:
            return None
        return left // right
    raise ValueError(f"Unknown operator: {operator}")


def generate_calculation(
    variable_count=6,
    step_count=10,
    seed=None,
    max_attempts=10000,
):
    if variable_count < 2:
        raise ValueError("At least two variables are required.")

    if variable_count > MAX_VALUE - MIN_VALUE + 1:
        raise ValueError("There cannot be more variables than distinct values.")

    rng = random.Random(seed)
    variable_names = [chr(ord("a") + i) for i in range(variable_count)]
    variable_names_ext = variable_names + list(range(MIN_VALUE, MAX_VALUE + 1))

    for _attempt in range(max_attempts):
        # Give every variable a different initial value.
        initial_values = rng.sample(
            range(MIN_VALUE, MAX_VALUE + 1),
            variable_count,
        )

        values = dict(zip(variable_names, initial_values))

        # A variable may be overwritten only after its current value has
        # been used in an expression. Initially, none have been used.
        used_since_assignment = {name: False for name in variable_names}

        # Tracks whether every variable has been used at least once.
        ever_used = {name: False for name in variable_names}

        operations = []
        failed = False

        for _step in range(step_count):
            candidates = []

            for left_name in variable_names:
                for right_name in variable_names_ext:
                    if left_name == right_name:
                        continue

                    left_value = values[left_name]
                    right_value = (
                        right_name
                        if isinstance(right_name, int)
                        else values[right_name]
                    )

                    for destination in variable_names:
                        # A variable can be overwritten if:
                        #   1. its current value was previously used, or
                        #   2. it is one of the operands in this expression.
                        #
                        # The second case permits expressions such as:
                        #     a = a + b
                        if not used_since_assignment[
                            destination
                        ] and destination not in (left_name, right_name):
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
                            if not (MIN_VALUE <= result <= MAX_VALUE):
                                continue

                            # Remove the old destination value, then check
                            # that the new state contains no duplicates.
                            other_values = [
                                values[name]
                                for name in variable_names
                                if name != destination
                            ]

                            if result in other_values:
                                continue

                            # Avoid pointless assignments such as a = a + 0.
                            if result == values[destination]:
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
            nonlit_candidates = [
                candidate
                for candidate in candidates
                if isinstance(candidate[1], str) and isinstance(candidate[3], str)
            ]

            if useful_candidates and rng.random() < 0.75:
                candidates = useful_candidates
            elif nonlit_candidates and rng.random() < 0.75:
                candidates = nonlit_candidates

            (
                destination,
                left_name,
                operator,
                right_name,
                result,
            ) = rng.choice(candidates)

            operations.append(
                (
                    destination,
                    left_name,
                    operator,
                    right_name,
                    result,
                )
            )

            values[destination] = result

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


def run_test(llm, args, steps):

    variable_names, initial_values, operations = generate_calculation(
        variable_count=args.variables,
        step_count=steps,
    )

    lines = []

    lines.append("This task tests your ability to do long sequences of mental math.")
    lines.append(
        "Calculate intermediate results as you are processing the input, then in your response, output just a single number corresponding to the final result."
    )
    lines.append("")

    for name, value in zip(variable_names, initial_values):
        lines.append(f"{name} = {value}")

    lines.append("")

    for destination, left, operator, right, result in operations:
        lines.append(
            f"{destination} = {left} {operator} {right}"
            f"  # think: {destination} is now ..."
        )

    lines.append("")
    lines.append(f"What is the final value of {destination}?")

    prompt = "\n".join(lines)
    print(prompt)
    print()

    history = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
    response = llm.invoke(history)
    print(response)
    print()

    if isinstance(response.content, str):
        content = response.content
    elif isinstance(response.content, list):
        if len(response.content) != 1:
            content = "(incorrect length list)"
        else:
            content = response.content[0]["text"]
    else:
        content = "(unknown type)"

    try:
        correct = int(content) == result
    except ValueError:
        correct = False

    if "output_token_details" in response.usage_metadata:
        reasoning_tokens = response.usage_metadata["output_token_details"]["reasoning"]
    else:
        reasoning_tokens = max(0, response.usage_metadata["output_tokens"] - 10)
    valid = reasoning_tokens == 0

    print(f"Model answer: {content}")
    print(f"Correct answer: {result}")
    print(f"Reasoning tokens: {reasoning_tokens}")
    print(("CORRECT" if correct else "WRONG") if valid else "INVALID")
    print()

    return (correct, valid)


def run_sample(llm, args, steps):

    score = 0
    invalid = 0
    for rep in range(args.repeats):
        (correct, valid) = run_test(llm=llm, args=args, steps=steps)
        score += int(correct and valid)
        invalid += int(not valid)

    print(f"Provider:      {args.provider}")
    print(f"Model:         {args.model}")
    print(f"Variables:     {args.variables}")
    print(f"Steps:         {steps}")
    print(f"Total score:   {score} / {args.repeats}  ({100 * score // args.repeats}%)")
    print(
        f"Total invalid: {invalid} / {args.repeats}  ({100 * invalid // args.repeats}%)"
    )

    return (score, invalid)


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
        default=20,
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

    llm = init_chat_model(
        model_provider=args.provider,
        model=args.model,
        max_tokens=1000,
        **kwargs,
    )

    if args.horizon:
        lo = 0
        hi = 1
        while True:
            (score, invalid) = run_sample(llm=llm, args=args, steps=hi)
            if score < args.repeats // 2:
                break
            if hi >= 256:
                lo = hi
                break
            lo = hi
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            (score, invalid) = run_sample(llm=llm, args=args, steps=mid)
            if score < args.repeats // 2:
                hi = mid
            else:
                lo = mid
        print()
        print(f"Success horizon: {lo} steps")
    else:
        run_sample(llm=llm, args=args, steps=args.steps)


if __name__ == "__main__":
    main()

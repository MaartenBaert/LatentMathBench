# LatentMathBench

*LatentMathBench* is a microbenchmark that measures an LLM's ability to perform long sequences of simple mathematical calculations in latent space, i.e. *without using its chain-of-thought*.

[Motivation and benchmark results](docs/index.md)

Credits to [Rohan Subramani](https://rohansubramani.github.io/astra-no-cot/) for coming up with the idea of using a well-crafted system prompt + reasoning token monitoring via the API to suppress chain-of-thought reasoning for GPT-6 Astra.

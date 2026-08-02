"""Entry point for the Financial Research Assistant demo."""

from time import perf_counter

from dotenv import load_dotenv

from src.pipeline import run_pipeline

load_dotenv()

SAMPLE_QUESTION = (
    "How can delayed power connections and slower-than-expected AI workload utilisation affect"
    " a cloud provider's cash flow, depreciation, and valuation?"
)


def get_question() -> str:
    """Prompt for a research question and fall back to the sample question.

    Returns:
        The user's question, or ``SAMPLE_QUESTION`` when the user presses
        Enter without typing a question.
    """
    question = input(
        "Enter a financial research question, or press Enter to use the sample question:\n"
        f"{SAMPLE_QUESTION}\n\n"
        "Question: "
    ).strip()
    return question or SAMPLE_QUESTION


def main() -> None:
    """Run a selected question and print the answer with total latency."""
    question = get_question()
    start_time = perf_counter()
    response = run_pipeline(question)
    elapsed_time = perf_counter() - start_time

    print("\n=== FINAL ANSWER ===\n")
    print(response)
    print(f"\nTotal latency: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

"""Factory assistant that answers questions like "Why is Line 4 behind?".

The assistant does not guess. It is given a context built from:
    * current line status from the simulator,
    * model predictions (risk of missing target),
    * SHAP drivers from :func:`loomiq.models.explain.explain_line`,
    * recommendations from :mod:`loomiq.optimization.recommender`,
and is instructed to answer only from that context.

Requires ``ANTHROPIC_API_KEY`` in ``.env``. ``anthropic`` is imported inside
functions so the rest of the project works without it.
"""

from typing import Any

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are LoomIQ, an assistant for a garment factory manager. "
    "Answer only using the factory data provided in the context. "
    "Explain causes in plain language, mention the top drivers, and suggest "
    "the recommended action. If the data does not answer the question, say so."
)


def build_context(
    line_status: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    explanations: dict[str, list[dict[str, Any]]],
    recommendations: list[dict[str, Any]],
) -> str:
    """Turn factory data into a compact text block for the LLM.

    Returns:
        A string the model can read, e.g. one section per line.
    """
    # TODO: Format each line: status, risk %, top SHAP drivers, recommendation.
    raise NotImplementedError("build_context is not implemented yet")


class FactoryAssistant:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None) -> None:
        """Set up the assistant.

        Args:
            model: Model ID; defaults to ``assistant.model`` in config.yaml.
            max_tokens: Response length cap; defaults to config.
        """
        # TODO: Read defaults from loomiq.config.load_config()["assistant"].
        # TODO: Read ANTHROPIC_API_KEY via loomiq.config.get_env and create
        #       anthropic.Anthropic(api_key=...). Raise a clear error if missing.
        self.model = model
        self.max_tokens = max_tokens
        self.client: Any = None

    def ask(self, question: str, context: str, history: list[dict[str, str]] | None = None) -> str:
        """Answer a question using the given factory context.

        Args:
            question: The manager's question.
            context: Output of :func:`build_context`.
            history: Previous chat turns as ``{"role", "content"}`` dicts.

        Returns:
            The assistant's answer as plain text.
        """
        # TODO: client.messages.create(model=..., max_tokens=..., system=SYSTEM_PROMPT,
        #       messages=history + [{"role": "user", "content": context + question}])
        raise NotImplementedError("FactoryAssistant.ask is not implemented yet")

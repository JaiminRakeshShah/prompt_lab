"""Errors used by the Week 2 local model lab."""


class UnknownModelError(ValueError):
    """Raised when a model identifier is not present in the configured model table."""


class PermanentProviderError(Exception):
    """Non-retryable: malformed request, unavailable model, or unsupported parameter."""



class TransientProviderError(Exception):
    """Timeout, connection failure, or temporary Ollama/server failure. Retryable."""


class TruncatedResponseError(Exception):
    """Ollama reported that the output token ceiling was reached. Not retried."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Singleton pattern to cache the client instance
_client_instance = None


def get_openai_client():
    """
    Initialize and return the OpenAI client using singleton pattern.

    Returns:
        OpenAI: The OpenAI client instance

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set in environment variables
    """
    global _client_instance

    if _client_instance is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set in the environment variables."
            )
        _client_instance = OpenAI(api_key=api_key)
        print("🔑 OpenAI client initialized successfully")

    return _client_instance


def reset_client():
    """Reset the client instance (useful for testing or re-initialization)."""
    global _client_instance
    _client_instance = None

"""Helper functions for various utilities."""

def validate_name(name: str) -> bool:
    """Validate if a name is valid.

    A valid name consists of only letters, numbers, and underscores.

    Args:
        name (str): The name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    import re

    pattern = r'^[A-Za-z0-9_]+$'
    return bool(re.match(pattern, name))

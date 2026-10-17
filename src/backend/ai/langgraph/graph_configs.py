from src.backend.constants import DEFAULT_RANKING_GROUP_SIZE, MAX_RANKING_GROUP_SIZE

# Reasoning mode controls generated candidates; ranking group size is independent.
TOURNAMENT_MODE_CONFIG = {
    "low": 1,
    "medium": 3,
    "high": 5,
    "extra": 7,
}


def get_config(mode: str) -> int:
    """Return the candidate count for a reasoning mode, independently of match size.

    Args:
        mode: low, medium, high or extra.

    Returns:
        Respectively 1, 3, 5 or 7 generated candidates.

    Raises:
        KeyError: The mode is unsupported.
    """
    return TOURNAMENT_MODE_CONFIG[mode]


def validate_ranking_group_size(group_size: int) -> int:
    """Validate the maximum candidate count per match, independently of pool size.

    Args:
        group_size: Strict integer from 2 to 7; booleans are rejected.

    Returns:
        The validated group size, which may exceed the number of candidates.

    Raises:
        ValueError: If group_size is not a strict integer in the supported range.
    """
    if type(group_size) is not int or not 2 <= group_size <= MAX_RANKING_GROUP_SIZE:
        raise ValueError("ranking_group_size must be an integer from 2 to 7")
    return group_size

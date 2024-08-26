from platform import system


def get_system() -> str:
    """
    Gets the system / operating system name.

    Returns:
        str: the system / operating system name.
    """
    return system()

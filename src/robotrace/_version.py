"""SDK version. Single source of truth - re-exported from `robotrace.__version__`.

Bump in lockstep with `pyproject.toml`. Until 1.0 the public API
(notably `log_episode`) may change between minor versions; once we
hit 1.0 the contract is locked per AGENTS.md and breakages require
a major bump.
"""

__version__ = "0.1.0a12"


def install_command(*extras: str, pin: bool = True) -> str:
    """Suggested pip line for ConfigurationError hints and docs."""
    if extras:
        tag = ",".join(extras)
        if pin:
            return f"pip install 'robotrace-dev[{tag}]=={__version__}'"
        return f"pip install 'robotrace-dev[{tag}]'"
    if pin:
        return f"pip install robotrace-dev=={__version__}"
    return "pip install robotrace-dev"

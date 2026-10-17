"""Process-level setup that must run before the training or evaluation entry points."""

import logging
import logging.config
import os

TORCH_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "torch_cache"))
QUIET_LIBRARIES = ("PIL", "matplotlib", "urllib3")
PROJECT_LOGGERS = ("dataset", "models", "utils", "__main__")


def init() -> None:
    """Configure logging, silence Albumentations version checks and enable the Inductor cache."""
    init_logger_basicconfig()
    disable_albumentations_update_checks()
    init_torch_cache()


def disable_albumentations_update_checks() -> None:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


def init_torch_cache() -> None:
    """Persist compiled Inductor graphs under ``cache/torch_cache`` so later runs skip recompilation."""
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_CACHE_DIR
    # Lets Dynamo trace scalar reads such as .item() without breaking the graph.
    os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
    logging.getLogger(__name__).debug(f"Torch cache will be saved to: {TORCH_CACHE_DIR}")


def init_logger_basicconfig() -> None:
    """Console logging; project loggers use ``LOG_LEVEL`` (default DEBUG), PIL, matplotlib and urllib3 WARNING."""
    log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()

    def console_logger(level: str) -> dict:
        return {"handlers": ["console"], "level": level, "propagate": False}

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": log_level,
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "": {"handlers": ["console"], "level": "INFO", "propagate": True},
                **{name: console_logger("WARNING") for name in QUIET_LIBRARIES},
                **{name: console_logger(log_level) for name in PROJECT_LOGGERS},
            },
        }
    )

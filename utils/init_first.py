import logging.config
import os

def init():
    init_logger_basicconfig()
    disable_albumentations_update_checks()
    init_torch_cache()

def disable_albumentations_update_checks():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

def init_torch_cache():
    logger = logging.getLogger(__name__)
    
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache/torch_cache")
    )
    
    os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

    logger.debug(f"\x1b[38;5;240mTorch cache will be saved to: {os.environ['TORCHINDUCTOR_CACHE_DIR']}\x1b[0m")

def init_logger_basicconfig():
    log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": log_level,
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": True
            },
            "PIL": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
            "matplotlib": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
            "urllib3": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
            "dataset": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            },
            "models": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            },
            "utils": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            },
            "__main__": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)

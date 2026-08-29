import logging
import logging.config
import pathlib


class GlobalRoundFormatter(logging.Formatter):
    """Formatter that guarantees record.global_round exists."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "global_round"):
            record.global_round = 0
        return super().format(record)


class RoundFilter(logging.Filter):
    """Injects current global_round into every log record."""

    def __init__(self):
        super().__init__()
        self.global_round = 0

    def set_round(self, r: int):
        self.global_round = r

    def filter(self, record: logging.LogRecord) -> bool:
        record.global_round = self.global_round
        return True


round_filter = RoundFilter()


def setup_round_logging(log_path="logs/fl_simulation.log"):
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {
                "()": GlobalRoundFormatter,
                "format": "[Global Round %(global_round)s]%(asctime)s %(levelname)s %(name)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "std",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "std",
                "filename": log_path,
                "maxBytes": 50_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(cfg)

    root = logging.getLogger()
    root.filters.clear()
    for handler in root.handlers:
        handler.filters.clear()
    root.addFilter(round_filter)
    for handler in root.handlers:
        handler.addFilter(round_filter)
    round_filter.set_round(0)

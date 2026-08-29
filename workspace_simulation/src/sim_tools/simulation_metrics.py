from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsConfig:
    training_acc: bool = True
    training_loss: bool = True
    training_recall: bool = False
    training_precision: bool = False
    test_acc: bool = True
    test_loss: bool = True
    test_recall: bool = False
    test_precision: bool = False


@dataclass
class MetricValues:
    acc: dict
    loss: dict
    recall: dict
    precision: dict


class SimulationMetricsRecorder:
    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._tables: dict[str, pd.DataFrame | None] = {
            "training_acc": None,
            "training_loss": None,
            "training_recall": None,
            "training_precision": None,
            "validation_accuracy_pre_agg": None,
            "validation_loss_pre_agg": None,
            "validation_recall_pre_agg": None,
            "validation_precision_pre_agg": None,
            "test_accuracy_post_agg": None,
            "test_loss_post_agg": None,
            "test_recall_post_agg": None,
            "test_precision_post_agg": None,
        }

    def initialize(self, training: MetricValues, test: MetricValues):
        missing_training = self._missing_metric_values(training)
        missing_test = self._missing_metric_values(test)
        if self.config.training_acc:
            self._set_table("training_acc", missing_training.acc, 0)
        if self.config.training_loss:
            self._set_table("training_loss", missing_training.loss, 0)
        if self.config.training_recall:
            self._set_table("training_recall", missing_training.recall, 0)
        if self.config.training_precision:
            self._set_table("training_precision", missing_training.precision, 0)

        if self.config.test_acc:
            self._set_table("validation_accuracy_pre_agg", missing_test.acc, 0)
            self._set_table("test_accuracy_post_agg", missing_test.acc, 0)
        if self.config.test_loss:
            self._set_table("validation_loss_pre_agg", missing_test.loss, 0)
            self._set_table("test_loss_post_agg", missing_test.loss, 0)
        if self.config.test_recall:
            self._set_table("validation_recall_pre_agg", missing_test.recall, 0)
            self._set_table("test_recall_post_agg", missing_test.recall, 0)
        if self.config.test_precision:
            self._set_table("validation_precision_pre_agg", missing_test.precision, 0)
            self._set_table("test_precision_post_agg", missing_test.precision, 0)

    def append_training(self, index: int, metrics: MetricValues):
        if self.config.training_acc:
            self._append_table("training_acc", metrics.acc, index)
        if self.config.training_loss:
            self._append_table("training_loss", metrics.loss, index)
        if self.config.training_recall:
            self._append_table("training_recall", metrics.recall, index)
        if self.config.training_precision:
            self._append_table("training_precision", metrics.precision, index)

    def append_validation_pre_aggregation(self, index: int, metrics: MetricValues):
        self._append_evaluation("validation", "pre_agg", index, metrics)

    def append_test_post_aggregation(self, index: int, metrics: MetricValues):
        self._append_evaluation("test", "post_agg", index, metrics)

    def save_excel(self, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                for sheet, df in self._tables.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=sheet)
        except ModuleNotFoundError as exc:
            if exc.name != "xlsxwriter":
                raise
            fallback_dir = file_path.with_suffix("")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            for sheet, df in self._tables.items():
                if isinstance(df, pd.DataFrame):
                    df.to_csv(fallback_dir / f"{sheet}.csv")
            log.warning(
                "XlsxWriter is not installed; saved metrics as CSV files under [%s] instead of [%s].",
                fallback_dir,
                file_path,
            )

    def _append_evaluation(self, split: str, stage: str, index: int, metrics: MetricValues):
        if self.config.test_acc:
            self._append_table(f"{split}_accuracy_{stage}", metrics.acc, index)
        if self.config.test_loss:
            self._append_table(f"{split}_loss_{stage}", metrics.loss, index)
        if self.config.test_recall:
            self._append_table(f"{split}_recall_{stage}", metrics.recall, index)
        if self.config.test_precision:
            self._append_table(f"{split}_precision_{stage}", metrics.precision, index)

    @staticmethod
    def _missing_metric_values(metrics: MetricValues) -> MetricValues:
        def missing(values):
            return {key: float("nan") for key in values}

        return MetricValues(
            acc=missing(metrics.acc),
            loss=missing(metrics.loss),
            recall=missing(metrics.recall),
            precision=missing(metrics.precision),
        )
    def _set_table(self, sheet: str, values: dict, index: int):
        self._tables[sheet] = pd.DataFrame(values, index=[index])

    def _append_table(self, sheet: str, values: dict, index: int):
        current = self._tables[sheet]
        new_row = pd.DataFrame(values, index=[index])
        self._tables[sheet] = new_row if current is None else pd.concat([current, new_row])

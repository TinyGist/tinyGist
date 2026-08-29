import csv
from collections import defaultdict
from dataclasses import dataclass
from operator import index
from pathlib import Path


FLOAT32_BYTES = 4
PACKET_STATUSES = {"delivered", "dropped"}


@dataclass(frozen=True)
class PacketPayload:
    """Logical packet payload using the simulation's float32 wire format."""

    model_parameter_elements: int = 0
    aggregation_weight_elements: int = 0
    batch_norm_elements: int = 0
    bitmap_bits: int = 0

    def __post_init__(self):
        for field_name in (
                "model_parameter_elements",
                "aggregation_weight_elements",
                "batch_norm_elements",
                "bitmap_bits",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer, not bool")
            try:
                value = index(value)
            except TypeError as exc:
                raise TypeError(f"{field_name} must be an integer") from exc
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)

    @property
    def model_parameter_bytes(self) -> int:
        return self.model_parameter_elements * FLOAT32_BYTES

    @property
    def aggregation_weight_bytes(self) -> int:
        return self.aggregation_weight_elements * FLOAT32_BYTES

    @property
    def batch_norm_bytes(self) -> int:
        return self.batch_norm_elements * FLOAT32_BYTES

    @property
    def bitmap_bytes(self) -> int:
        return (self.bitmap_bits + 7) // 8

    @property
    def total_bytes(self) -> int:
        return (
            self.model_parameter_bytes
            + self.aggregation_weight_bytes
            + self.batch_norm_bytes
            + self.bitmap_bytes
        )

    @property
    def contents(self) -> str:
        content_sizes = (
            ("model_parameters", self.model_parameter_elements),
            ("aggregation_weight", self.aggregation_weight_elements),
            ("batch_norm", self.batch_norm_elements),
            ("bitmap", self.bitmap_bits),
        )
        return "|".join(name for name, size in content_sizes if size)


@dataclass(frozen=True)
class CommunicationRoundSummary:
    global_round: int
    packet_count: int
    delivered_packet_count: int
    dropped_packet_count: int
    sent_bytes: int
    delivered_bytes: int
    dropped_bytes: int
    model_parameter_bytes: int
    aggregation_weight_bytes: int
    batch_norm_bytes: int
    bitmap_bytes: int


class CommunicationRecorder:
    """Append-only per-packet communication ledger for one simulation run."""

    FIELDNAMES = (
        "global_round",
        "packet_id",
        "method",
        "packet_kind",
        "selection_mode",
        "source_device",
        "destination_device",
        "source_local_round",
        "destination_local_round",
        "status",
        "segment_id",
        "parameter_scope",
        "segment_unit",
        "bn_distribution",
        "model_parameter_elements",
        "model_parameter_bytes",
        "aggregation_weight_metric",
        "aggregation_weight_elements",
        "aggregation_weight_bytes",
        "batch_norm_elements",
        "batch_norm_bytes",
        "bitmap_bits",
        "bitmap_bytes",
        "total_bytes",
        "total_bits",
        "contents",
    )

    def __init__(
            self,
            output_path,
            *,
            method: str,
            aggregation_weight_metric: str,
            parameter_scope: str,
            segment_unit: str,
            bn_distribution: str,
    ):
        self.output_path = Path(output_path)
        self.method = str(method)
        self.aggregation_weight_metric = str(aggregation_weight_metric)
        self.parameter_scope = str(parameter_scope)
        self.segment_unit = str(segment_unit)
        self.bn_distribution = str(bn_distribution)
        self._rows_by_round = defaultdict(list)
        self._packet_sequence_by_round = defaultdict(int)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as output_file:
            csv.DictWriter(output_file, fieldnames=self.FIELDNAMES).writeheader()

    def record_packet(
            self,
            *,
            global_round: int,
            packet_kind: str,
            selection_mode: str,
            source_device,
            destination_device,
            status: str,
            payload: PacketPayload,
            source_local_round=None,
            destination_local_round=None,
            segment_id=None,
    ) -> dict:
        status = str(status)
        if status not in PACKET_STATUSES:
            raise ValueError(f"status must be one of {sorted(PACKET_STATUSES)}")
        if not isinstance(payload, PacketPayload):
            raise TypeError("payload must be a PacketPayload")

        global_round = int(global_round)
        self._packet_sequence_by_round[global_round] += 1
        packet_id = f"r{global_round}-p{self._packet_sequence_by_round[global_round]:06d}"
        row = {
            "global_round": global_round,
            "packet_id": packet_id,
            "method": self.method,
            "packet_kind": str(packet_kind),
            "selection_mode": str(selection_mode),
            "source_device": str(source_device),
            "destination_device": str(destination_device),
            "source_local_round": "" if source_local_round is None else int(source_local_round),
            "destination_local_round": "" if destination_local_round is None else int(destination_local_round),
            "status": status,
            "segment_id": "" if segment_id is None else str(segment_id),
            "parameter_scope": self.parameter_scope,
            "segment_unit": self.segment_unit,
            "bn_distribution": self.bn_distribution,
            "model_parameter_elements": payload.model_parameter_elements,
            "model_parameter_bytes": payload.model_parameter_bytes,
            "aggregation_weight_metric": self.aggregation_weight_metric,
            "aggregation_weight_elements": payload.aggregation_weight_elements,
            "aggregation_weight_bytes": payload.aggregation_weight_bytes,
            "batch_norm_elements": payload.batch_norm_elements,
            "batch_norm_bytes": payload.batch_norm_bytes,
            "bitmap_bits": payload.bitmap_bits,
            "bitmap_bytes": payload.bitmap_bytes,
            "total_bytes": payload.total_bytes,
            "total_bits": payload.total_bytes * 8,
            "contents": payload.contents,
        }
        self._rows_by_round[global_round].append(row)
        return row

    def flush_round(self, global_round: int) -> CommunicationRoundSummary:
        global_round = int(global_round)
        rows = self._rows_by_round.get(global_round, [])
        if rows:
            with self.output_path.open("a", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=self.FIELDNAMES)
                writer.writerows(rows)
        self._rows_by_round.pop(global_round, None)
        return self._summarize(global_round, rows)

    def close(self):
        for global_round in sorted(self._rows_by_round):
            self.flush_round(global_round)

    @staticmethod
    def _summarize(global_round, rows) -> CommunicationRoundSummary:
        delivered_packet_count = 0
        dropped_packet_count = 0
        sent_bytes = 0
        delivered_bytes = 0
        dropped_bytes = 0
        model_parameter_bytes = 0
        aggregation_weight_bytes = 0
        batch_norm_bytes = 0
        bitmap_bytes = 0
        for row in rows:
            row_total_bytes = row["total_bytes"]
            sent_bytes += row_total_bytes
            if row["status"] == "delivered":
                delivered_packet_count += 1
                delivered_bytes += row_total_bytes
            else:
                dropped_packet_count += 1
                dropped_bytes += row_total_bytes
            model_parameter_bytes += row["model_parameter_bytes"]
            aggregation_weight_bytes += row["aggregation_weight_bytes"]
            batch_norm_bytes += row["batch_norm_bytes"]
            bitmap_bytes += row["bitmap_bytes"]

        return CommunicationRoundSummary(
            global_round=global_round,
            packet_count=len(rows),
            delivered_packet_count=delivered_packet_count,
            dropped_packet_count=dropped_packet_count,
            sent_bytes=sent_bytes,
            delivered_bytes=delivered_bytes,
            dropped_bytes=dropped_bytes,
            model_parameter_bytes=model_parameter_bytes,
            aggregation_weight_bytes=aggregation_weight_bytes,
            batch_norm_bytes=batch_norm_bytes,
            bitmap_bytes=bitmap_bytes,
        )

from collections import defaultdict

import numpy as np


RECIPIENT_PICK_METHODS = {
    "uniform_with_replacement",
    "grouped_probabilistic",
    "grouped_unique_probabilistic",
    "grouped_round_robin",
    "uniform_without_replacement",
}

RECIPIENT_PICK_METHOD_ALIASES = {
    "uniform": "uniform_with_replacement",
    "random": "uniform_with_replacement",
    "uniform_random": "uniform_with_replacement",
    "with_replacement": "uniform_with_replacement",
    "uniform_with_replacement": "uniform_with_replacement",
    "random_with_replacement": "uniform_with_replacement",
    "probabilistic": "grouped_probabilistic",
    "grouped_probabilistic": "grouped_probabilistic",
    "grouped_probability": "grouped_probabilistic",
    "balanced_probabilistic": "grouped_probabilistic",
    "grouped_unique_probabilistic": "grouped_unique_probabilistic",
    "balanced_unique_probabilistic": "grouped_unique_probabilistic",
    "round_robin": "grouped_round_robin",
    "grouped_round_robin": "grouped_round_robin",
    "balanced_round_robin": "grouped_round_robin",
    "without_replacement": "uniform_without_replacement",
    "uniform_without_replacement": "uniform_without_replacement",
    "random_without_replacement": "uniform_without_replacement",
}


def _normalize_key(value) -> str:
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def canonical_recipient_pick_method(value) -> str:
    key = _normalize_key(value)
    try:
        return RECIPIENT_PICK_METHOD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid recipient.pick_method [{value}], "
            f"supported values are {sorted(RECIPIENT_PICK_METHODS)}"
        ) from exc


class RecipientSelector:
    def __init__(self, pick_method="uniform_with_replacement", balance_strength=1.0):
        self.pick_method = canonical_recipient_pick_method(pick_method)
        self.balance_strength = float(balance_strength)
        if self.balance_strength < 0:
            raise ValueError("recipient.balance_strength must be non-negative")
        self._assignment_counts = defaultdict(int)
        self._round_robin_cursor = defaultdict(int)

    def select(self, sender_id, segment_ids, recipient_ids):
        segment_ids = list(segment_ids)
        recipient_ids = list(recipient_ids)
        if not segment_ids or not recipient_ids:
            return []
        if self.pick_method == "uniform_with_replacement":
            return self._uniform_with_replacement(segment_ids, recipient_ids)
        if self.pick_method == "uniform_without_replacement":
            return self._uniform_without_replacement(segment_ids, recipient_ids)
        if self.pick_method == "grouped_probabilistic":
            return self._grouped_select(sender_id, segment_ids, recipient_ids, self._probabilistic_recipient)
        if self.pick_method == "grouped_unique_probabilistic":
            return self._grouped_unique_probabilistic(sender_id, segment_ids, recipient_ids)
        if self.pick_method == "grouped_round_robin":
            return self._grouped_select(sender_id, segment_ids, recipient_ids, self._round_robin_recipient)
        raise RuntimeError(f"Unsupported recipient pick method [{self.pick_method}]")

    def _uniform_with_replacement(self, segment_ids, recipient_ids):
        recipient_array = np.asarray(recipient_ids, dtype=object)
        return np.random.choice(recipient_array, size=len(segment_ids), replace=True).tolist()

    def _uniform_without_replacement(self, segment_ids, recipient_ids):
        picked = []
        recipient_array = np.asarray(recipient_ids, dtype=object)
        while len(picked) < len(segment_ids):
            remaining = len(segment_ids) - len(picked)
            sample_size = min(remaining, len(recipient_ids))
            picked.extend(
                np.random.choice(recipient_array, size=sample_size, replace=False).tolist()
            )
        return picked

    def _grouped_select(self, sender_id, segment_ids, recipient_ids, pick_one):
        selected_recipients = [None] * len(segment_ids)
        grouped_positions = self._segment_positions(segment_ids)
        for segment_id, positions in grouped_positions.items():
            for position in positions:
                recipient_id = pick_one(sender_id, segment_id, recipient_ids)
                selected_recipients[position] = recipient_id
                self._assignment_counts[(sender_id, segment_id, recipient_id)] += 1
        return selected_recipients

    def _grouped_unique_probabilistic(self, sender_id, segment_ids, recipient_ids):
        selected_recipients = [None] * len(segment_ids)
        for segment_id, positions in self._segment_positions(segment_ids).items():
            available_recipients = list(recipient_ids)
            for position in positions:
                if not available_recipients:
                    available_recipients = list(recipient_ids)
                recipient_id = self._probabilistic_recipient(
                    sender_id,
                    segment_id,
                    available_recipients,
                )
                selected_recipients[position] = recipient_id
                self._assignment_counts[(sender_id, segment_id, recipient_id)] += 1
                available_recipients.remove(recipient_id)
        return selected_recipients

    @staticmethod
    def _segment_positions(segment_ids):
        grouped_positions = {}
        for position, segment_id in enumerate(segment_ids):
            grouped_positions.setdefault(segment_id, []).append(position)
        return grouped_positions

    def _probabilistic_recipient(self, sender_id, segment_id, recipient_ids):
        counts = np.asarray(
            [
                self._assignment_counts[(sender_id, segment_id, recipient_id)]
                for recipient_id in recipient_ids
            ],
            dtype=np.float64,
        )
        gaps = counts - counts.min()
        weights = 1.0 / np.power(1.0 + gaps, self.balance_strength)
        probabilities = weights / weights.sum()
        recipient_array = np.asarray(recipient_ids, dtype=object)
        return np.random.choice(recipient_array, p=probabilities)

    def _round_robin_recipient(self, sender_id, segment_id, recipient_ids):
        cursor_key = (sender_id, segment_id)
        counts = [
            self._assignment_counts[(sender_id, segment_id, recipient_id)]
            for recipient_id in recipient_ids
        ]
        min_count = min(counts)
        start = self._round_robin_cursor[cursor_key] % len(recipient_ids)
        for offset in range(len(recipient_ids)):
            candidate_idx = (start + offset) % len(recipient_ids)
            if counts[candidate_idx] == min_count:
                self._round_robin_cursor[cursor_key] = candidate_idx + 1
                return recipient_ids[candidate_idx]
        raise RuntimeError("Round-robin recipient selection failed")

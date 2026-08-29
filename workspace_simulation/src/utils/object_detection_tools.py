from importlib import import_module

from src.sim_tools.definitions import OBJECT_DETECTION_METRIC_IMPORTS


def infer_object_detection_task(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if "fomo" in normalized:
        return "fomo"
    if "yolo" in normalized:
        return "yolo"
    raise NotImplementedError(f"Target {normalized} not implemented.")


def validate_object_detection_configuration(
        dataset_name: str,
        model_name: str,
        loss_function_name: str,
        output_class_number: int,
):
    specs = {
        "fomo_person": ("fomo", "fomo_person_loss", 2),
        "fomo_vehicle": ("fomo", "fomo_vehicle_loss", 6),
        "fomo_vehicle_binary": ("fomo", "fomo_vehicle_binary_loss", 2),
        "yolo_person": ("yolo", "yolo_person_loss", 1),
        "yolo_vehicle": ("yolo", "yolo_vehicle_loss", 5),
        "yolo_vehicle_binary": ("yolo", "yolo_vehicle_binary_loss", 1),
    }
    if dataset_name not in specs:
        raise ValueError(
            f"Object detection requires one of {sorted(specs)}, got {dataset_name!r}"
        )
    expected_family, expected_loss, expected_classes = specs[dataset_name]
    actual_family = infer_object_detection_task(model_name)
    if actual_family != expected_family:
        raise ValueError(
            f"Dataset {dataset_name} requires a {expected_family.upper()} model, "
            f"got {model_name}"
        )
    if loss_function_name != expected_loss:
        raise ValueError(
            f"Dataset {dataset_name} requires loss {expected_loss}, "
            f"got {loss_function_name}"
        )
    if output_class_number != expected_classes:
        raise ValueError(
            f"Dataset {dataset_name} requires model.num_classes={expected_classes}, "
            f"got {output_class_number}"
        )


def build_detection_metrics(task: str, loss_function_name: str):
    target_name = loss_function_name.lower()
    target = _target_from_loss_name(target_name)
    binary = "binary" in target_name

    metric_imports = OBJECT_DETECTION_METRIC_IMPORTS.get(task)
    if metric_imports is None:
        raise NotImplementedError(f"Object detection task {task} is not implemented.")

    metric_import = metric_imports.get((target, binary)) or metric_imports.get((target, False))
    if metric_import is None:
        raise NotImplementedError(f"Loss function {loss_function_name} is not implemented yet")
    module_name, class_name = metric_import
    metric_class = getattr(import_module(module_name), class_name)
    return metric_class()


def _target_from_loss_name(loss_function_name: str) -> str:
    if "vehicle" in loss_function_name:
        return "vehicle"
    if "person" in loss_function_name:
        return "person"
    raise NotImplementedError(f"Loss function {loss_function_name} is not implemented yet")

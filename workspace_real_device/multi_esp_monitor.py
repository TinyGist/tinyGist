#!/usr/bin/env python3
"""Synchronously restart and monitor every attached ESP serial device."""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import serial


DEFAULT_BAUD = 115200
DEFAULT_DEVICE_ROOT = "/dev"
DEFAULT_LOG_DIR = "./logs"
RESET_PULSE_SECONDS = 0.1
PORT_PATTERN = re.compile(r"^tty(ACM|USB)([0-9]+)$")


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return value


def discover_esp_ports(device_root: str | os.PathLike[str]) -> list[Path]:
    """Return ttyACM<n> ports first, followed by ttyUSB<n>, naturally sorted."""
    root = Path(device_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Serial-device directory does not exist: {root}")

    discovered: list[tuple[int, int, Path]] = []
    family_order = {"ACM": 0, "USB": 1}
    for candidate in root.iterdir():
        match = PORT_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        family, numeric_suffix = match.groups()
        discovered.append((family_order[family], int(numeric_suffix), candidate))

    return [entry[2] for entry in sorted(discovered)]


def open_serial_devices(
    ports: Iterable[Path],
    baud: int,
) -> list[tuple[Path, serial.Serial]]:
    """Open every port before changing any reset-control line."""
    opened: list[tuple[Path, serial.Serial]] = []
    try:
        for port in ports:
            connection = serial.Serial(
                port=None,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.2,
                write_timeout=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                exclusive=True,
            )
            connection.dtr = False
            connection.rts = False
            connection.port = str(port)
            try:
                connection.open()
            except BaseException:
                connection.close()
                raise
            opened.append((port, connection))
    except BaseException:
        close_serial_devices(opened)
        raise
    return opened


def restart_all_devices(
    devices: Iterable[tuple[Path, serial.Serial]],
    pulse_seconds: float = RESET_PULSE_SECONDS,
) -> None:
    """Hold every ESP in reset, then release all devices as one barrier."""
    device_list = list(devices)
    try:
        for _, connection in device_list:
            connection.dtr = False
        for _, connection in device_list:
            connection.rts = True
        time.sleep(pulse_seconds)
        # Clear bytes left from a previous run while every board is still held
        # in reset. Output produced after reset release remains available.
        for _, connection in device_list:
            connection.reset_input_buffer()
    finally:
        release_error: BaseException | None = None
        for _, connection in device_list:
            try:
                connection.rts = False
            except BaseException as exc:
                if release_error is None:
                    release_error = exc
        if release_error is not None and sys.exc_info()[0] is None:
            raise RuntimeError("Failed to release at least one device from reset") from release_error


def close_serial_devices(devices: Iterable[tuple[Path, serial.Serial]]) -> None:
    for _, connection in devices:
        try:
            connection.close()
        except Exception:
            pass


def monitor_device(
    name: str,
    port: Path,
    connection: serial.Serial,
    log_path: Path,
    stop_event: threading.Event,
    error_queue: queue.Queue[tuple[str, Exception]],
) -> None:
    try:
        with log_path.open("w", buffering=1, encoding="utf-8") as log_file:
            while not stop_event.is_set():
                raw_line = connection.readline()
                if not raw_line:
                    continue
                text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                timestamp = datetime.now().isoformat(timespec="milliseconds")
                log_file.write(f"[{timestamp}] {text}\n")
                try:
                    print(f"[{timestamp}] [{name}] {text}", flush=True)
                except (BrokenPipeError, OSError):
                    pass
    except Exception as exc:
        error_queue.put((f"{name} ({port})", exc))
        stop_event.set()


def create_run_directory(log_root: str | os.PathLike[str]) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = Path(log_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def write_device_map(
    run_directory: Path,
    devices: Iterable[tuple[Path, serial.Serial]],
    baud: int,
) -> None:
    mapping = [
        {
            "monitor_index": index,
            "name": f"device_{index}",
            "port": str(port),
            "baud": baud,
        }
        for index, (port, _) in enumerate(devices)
    ]
    (run_directory / "device_map.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    device_root = os.environ.get("TINYGIST_DEVICE_ROOT", DEFAULT_DEVICE_ROOT)
    log_root = os.environ.get("TINYGIST_LOG_DIR", DEFAULT_LOG_DIR)
    try:
        baud = _positive_int_from_env("TINYGIST_MONITOR_BAUD", DEFAULT_BAUD)
        ports = discover_esp_ports(device_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not ports:
        print(
            f"Error: no ttyACM<number> or ttyUSB<number> ports found in {device_root}",
            file=sys.stderr,
        )
        return 1

    print("Discovered ESP serial ports:")
    for index, port in enumerate(ports):
        print(f"  monitor_index={index}: {port}")

    devices: list[tuple[Path, serial.Serial]] = []
    try:
        devices = open_serial_devices(ports, baud)
        restart_all_devices(devices)
    except KeyboardInterrupt:
        close_serial_devices(devices)
        print("Interrupted before monitoring started.", file=sys.stderr)
        return 130
    except Exception as exc:
        close_serial_devices(devices)
        print(f"Error: could not open and restart every device: {exc}", file=sys.stderr)
        return 1

    stop_event = threading.Event()
    error_queue: queue.Queue[tuple[str, Exception]] = queue.Queue()
    threads: list[threading.Thread] = []

    try:
        # No log file is created until every serial port has opened and every
        # device has been released from the synchronized reset barrier.
        run_directory = create_run_directory(log_root)
        for index, (port, connection) in enumerate(devices):
            name = f"device_{index}"
            thread = threading.Thread(
                target=monitor_device,
                args=(
                    name,
                    port,
                    connection,
                    run_directory / f"{name}.log",
                    stop_event,
                    error_queue,
                ),
                name=f"monitor-{name}",
            )
            thread.start()
            threads.append(thread)
        write_device_map(run_directory, devices, baud)

        print(f"Logging {len(devices)} devices to {run_directory}")
        while not stop_event.wait(0.5):
            if not any(thread.is_alive() for thread in threads):
                break

        if not error_queue.empty():
            source, exc = error_queue.get()
            print(f"Error: serial monitoring stopped at {source}: {exc}", file=sys.stderr)
            return 1
        return 0
    except KeyboardInterrupt:
        print("Stopping serial monitors...")
        return 130
    finally:
        stop_event.set()
        for _, connection in devices:
            cancel_read = getattr(connection, "cancel_read", None)
            if callable(cancel_read):
                try:
                    cancel_read()
                except Exception:
                    pass
        close_serial_devices(devices)
        for thread in threads:
            thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (edit if needed)
# =========================
BRIDGE_NAME="br0"
BRIDGE_IP="172.17.0.1/24"

QEMU_BIN="qemu-system-xtensa"
QEMU_RAM_MB=4
MACHINE="esp32"

CLEANUP_ON_EXIT=1
QEMU_PIDS=()
CLEANUP_DONE=0

die() { echo "Error: $*" >&2; exit 1; }

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "Please run as root (inside container it's usually root by default)."
  fi
}

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

check_deps() {
  command -v ip >/dev/null 2>&1 || die "'ip' command not found (install iproute2)."
  command -v "${QEMU_BIN}" >/dev/null 2>&1 || die "'${QEMU_BIN}' not found in PATH."
}

ensure_tun() {
  [[ -c /dev/net/tun ]] || die "/dev/net/tun not available. Run container with --device /dev/net/tun (or --privileged)."
}

# Select bins in order:
# - If flash0.bin exists, strictly use flash0.bin ... flash(N-1).bin (missing one => error)
# - Else fallback: version-sort all .bin and take first N
select_bins_in_order() {
  local dir="$1"
  local n="$2"
  local -a bins=()

  if [[ -f "${dir}/flash0.bin" ]]; then
    for ((i=0; i<n; i++)); do
      local f="flash${i}.bin"
      [[ -f "${dir}/${f}" ]] || die "Missing ${f}. You requested ${n} devices, so flash0.bin .. flash$((n-1)).bin must exist."
      bins+=("${f}")
    done
    printf '%s\n' "${bins[@]}"
    return 0
  fi

  mapfile -t bins < <(
    find "${dir}" -maxdepth 1 -type f -name "*.bin" -printf "%f\n" | LC_ALL=C sort -V
  )
  [[ "${#bins[@]}" -ge "${n}" ]] || die "Not enough .bin files: need ${n}, found ${#bins[@]} in ${dir}."
  printf '%s\n' "${bins[@]:0:${n}}"
}

setup_bridge_and_taps() {
  local n="$1"

  if ! ip link show "${BRIDGE_NAME}" >/dev/null 2>&1; then
    ip link add "${BRIDGE_NAME}" type bridge
  fi

  # Assign IP (best-effort idempotent)
  if ! ip addr show "${BRIDGE_NAME}" | grep -q "${BRIDGE_IP%/*}"; then
    ip addr flush dev "${BRIDGE_NAME}" || true
    ip addr add "${BRIDGE_IP}" dev "${BRIDGE_NAME}" || true
  fi
  ip link set "${BRIDGE_NAME}" up

  # Create only n taps: tap1..tapN
  for ((i=1; i<=n; i++)); do
    local tap="tap${i}"
    if ip link show "${tap}" >/dev/null 2>&1; then
      ip link set "${tap}" down || true
      ip link delete "${tap}" type tuntap || true
    fi
    ip tuntap add dev "${tap}" mode tap
    ip link set "${tap}" master "${BRIDGE_NAME}"
    ip link set "${tap}" up
  done

  echo "Bridge ${BRIDGE_NAME} (${BRIDGE_IP}) is up; created ${n} TAP devices (tap1..tap${n})."
}

cleanup_network() {
  local n="$1"
  echo "Cleaning up network (taps + bridge)..."

  for ((i=1; i<=n; i++)); do
    local tap="tap${i}"
    if ip link show "${tap}" >/dev/null 2>&1; then
      ip link set "${tap}" down || true
      ip link delete "${tap}" type tuntap || true
    fi
  done

  if ip link show "${BRIDGE_NAME}" >/dev/null 2>&1; then
    ip link set "${BRIDGE_NAME}" down || true
    ip link delete "${BRIDGE_NAME}" type bridge || true
  fi
}

cleanup_qemu() {
  echo "Stopping QEMU instances started by this script..."
  local pid
  local sent_term=0

  for pid in "${QEMU_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      sent_term=1
    fi
  done

  if [[ "${sent_term}" -eq 1 ]]; then
    sleep 2
  fi

  for pid in "${QEMU_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  for pid in "${QEMU_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  QEMU_PIDS=()
}

launch_qemu() {
  local dir="$1"
  local log_dir="$2"
  shift 2
  local -a bins=("$@")
  local n="${#bins[@]}"

  mkdir -p "${dir}/${log_dir}"
  echo "Launching ${n} QEMU instances. Logs will be stored in: ${dir}/${log_dir}/"

  for ((i=0; i<n; i++)); do
    local tap_if="tap$((i+1))"
    local bin_name="${bins[$i]}"
    local flash_file="${dir}/${bin_name}"
    local base="${bin_name%.bin}"
    local log_file="${dir}/${log_dir}/${base}.log"

    echo "[$i] Using ${bin_name} with ${tap_if}"

    # Keep QEMU away from the controlling TTY so Ctrl-C reaches this script.
    "${QEMU_BIN}" -nographic -machine "${MACHINE}" -m "${QEMU_RAM_MB}" \
      -drive file="${flash_file}",if=mtd,format=raw \
      -nic tap,model=open_eth,ifname="${tap_if}",downscript=no,script=no \
      > "${log_file}" 2>&1 </dev/null &
    QEMU_PIDS+=("$!")
    sleep 0.05
  done
}

wait_for_cluster() {
  local status
  echo "All QEMU instances are running. Press Ctrl-C to stop the group."

  set +e
  wait -n "${QEMU_PIDS[@]}"
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "Error: a QEMU instance exited unexpectedly." >&2
    return 1
  fi
  echo "Error: a QEMU instance exited with status ${status}." >&2
  return "${status}"
}

on_signal() {
  case "$1" in
    INT)
      echo "Received Ctrl-C; stopping the emulation group."
      exit 130
      ;;
    TERM)
      echo "Received SIGTERM; stopping the emulation group."
      exit 143
      ;;
  esac
}

cleanup_all() {
  local status=$?

  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    exit "${status}"
  fi
  CLEANUP_DONE=1
  trap - INT TERM EXIT

  cleanup_qemu
  if [[ "${CLEANUP_ON_EXIT}" -eq 1 ]]; then
    cleanup_network "${TARGET_NUM}"
  fi
  exit "${status}"
}

# =========================
# Main
# =========================
need_root
check_deps
ensure_tun

DIR="$(script_dir)"

TARGET_NUM="${1:-}"
[[ -n "${TARGET_NUM}" ]] || die "Usage: $0 <num_devices>"
[[ "${TARGET_NUM}" =~ ^[0-9]+$ ]] || die "num_devices must be a positive integer."
[[ "${TARGET_NUM}" -ge 1 ]] || die "num_devices must be >= 1."
if [[ "${TARGET_NUM}" -gt 200 ]]; then
  echo "Requested ${TARGET_NUM} devices, cap to 200."
  TARGET_NUM=200
fi

LOG_DIR="logs_${TARGET_NUM}_devices"

mapfile -t BIN_FILES < <(select_bins_in_order "${DIR}" "${TARGET_NUM}")

echo "Will launch ${TARGET_NUM} device(s) using the following bins in order:"
printf '  - %s\n' "${BIN_FILES[@]}"
echo "Log directory: ${DIR}/${LOG_DIR}"

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap cleanup_all EXIT

setup_bridge_and_taps "${TARGET_NUM}"
launch_qemu "${DIR}" "${LOG_DIR}" "${BIN_FILES[@]}"
wait_for_cluster

#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: $0 [number_of_devices]"
  exit 1
fi

NUM_DEVICES="${1:-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}"

# -----------------------------
# Input checks
# -----------------------------
if ! [[ "${NUM_DEVICES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: number_of_devices must be a positive decimal integer without leading zeros."
  exit 1
fi

DATASET_ROOT=""
for candidate in "${SCRIPT_DIR}"/main/dataset_*; do
  [[ -d "${candidate}" ]] || continue
  if [[ -n "${DATASET_ROOT}" ]]; then
    echo "Error: multiple dataset roots found under ${SCRIPT_DIR}/main."
    exit 1
  fi
  DATASET_ROOT="${candidate}"
done

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "Error: no dataset root found under ${SCRIPT_DIR}/main."
  exit 1
fi

for ((device=0; device<NUM_DEVICES; device++)); do
  if [[ ! -d "${DATASET_ROOT}/device_${device}" ]]; then
    echo "Error: dataset shard not found: ${DATASET_ROOT}/device_${device}"
    exit 1
  fi
done

MAXIMUM_NUM_DEVICES="${NUM_DEVICES}"

if ! command -v idf.py >/dev/null 2>&1; then
  echo "Error: idf.py not found. Did you run 'source export.sh'?"
  exit 1
fi

if [[ "$(idf.py --version)" != "ESP-IDF v5.2.0" ]]; then
  echo "Error: build_flash.sh requires ESP-IDF v5.2.0."
  exit 1
fi

# -----------------------------
# Helper: pack a 4MB flash image
# Layout (same as your flash.sh):
#   0x1000  bootloader.bin
#   0x8000  partition-table.bin
#   0x10000 app.bin
# -----------------------------
pack_flash_image() {
  local build_dir="$1"
  local app_bin="$2"
  local out_bin="$3"

  local boot_bin="${build_dir}/bootloader/bootloader.bin"
  local part_bin="${build_dir}/partition_table/partition-table.bin"

  if [[ ! -f "${boot_bin}" ]]; then
    echo "Error: bootloader not found: ${boot_bin}"
    exit 1
  fi
  if [[ ! -f "${part_bin}" ]]; then
    echo "Error: partition table not found: ${part_bin}"
    exit 1
  fi
  if [[ ! -f "${app_bin}" ]]; then
    echo "Error: app bin not found: ${app_bin}"
    exit 1
  fi

  mkdir -p "$(dirname "${out_bin}")"

  # Create empty 4MB flash image
  dd if=/dev/zero bs=1M count=4 of="${out_bin}" status=none

  # Write bootloader @ 0x1000
  dd if="${boot_bin}" bs=1 count="$(stat -c%s "${boot_bin}")" seek=$((16#1000)) conv=notrunc of="${out_bin}" status=none

  # Write partition table @ 0x8000
  dd if="${part_bin}" bs=1 count="$(stat -c%s "${part_bin}")" seek=$((16#8000)) conv=notrunc of="${out_bin}" status=none

  # Write app @ 0x10000
  dd if="${app_bin}" bs=1 count="$(stat -c%s "${app_bin}")" seek=$((16#10000)) conv=notrunc of="${out_bin}" status=none
}

# -----------------------------
# Methods mapping
# -----------------------------
method_name() {
  case "$1" in
    1) echo "dfa" ;;
    2) echo "sdfa" ;;
    3) echo "gist" ;;
    *) echo "unknown" ;;
  esac
}

# -----------------------------
# Build + pack for each method
# -----------------------------
for METHOD in 1 2 3; do
  NAME="$(method_name "${METHOD}")"
  OUT_DIR="bin_dir_${NAME}_${MAXIMUM_NUM_DEVICES}"
  BUILD_DIR="build_${NAME}_${MAXIMUM_NUM_DEVICES}"

  mkdir -p "${OUT_DIR}"
  # Optional: clean old outputs to avoid mixing stale files.
  rm -f "${OUT_DIR}/flash"*.bin 2>/dev/null || true

  echo "===================================================="
  echo "MAXIMUM_NUM_DEVICES=${MAXIMUM_NUM_DEVICES}"
  echo "METHOD=${METHOD} (${NAME})"
  echo "Build dir : ${BUILD_DIR}"
  echo "Out dir   : ${OUT_DIR}"
  echo "Devices   : ${NUM_DEVICES}"
  echo "===================================================="

  # Clean build dir once per method to avoid cross-method contamination.
  idf.py -B "${BUILD_DIR}" fullclean >/dev/null

  for ((device=0; device<NUM_DEVICES; device++)); do
    echo "[${NAME}] Building device=${device} (DATASET=${device}, METHOD=${METHOD}, MAXIMUM_NUM_DEVICES=${MAXIMUM_NUM_DEVICES})..."

    # Build (DATASET == device id).
    idf.py -B "${BUILD_DIR}" -DDATASET="${device}" -DMETHOD="${METHOD}" -DMAXIMUM_NUM_DEVICES="${MAXIMUM_NUM_DEVICES}" build >/dev/null

    # ESP-IDF app output name in your project is build/esp-adfo.bin.
    APP_BIN="${BUILD_DIR}/esp-adfo.bin"
    OUT_BIN="${OUT_DIR}/flash${device}.bin"

    pack_flash_image "${BUILD_DIR}" "${APP_BIN}" "${OUT_BIN}"

    echo "[${NAME}] -> ${OUT_BIN}"
  done
done

echo ""
echo "Done. Generated:"
echo "  bin_dir_dfa_${MAXIMUM_NUM_DEVICES}   (METHOD=1)"
echo "  bin_dir_sdfa_${MAXIMUM_NUM_DEVICES}  (METHOD=2)"
echo "  bin_dir_gist_${MAXIMUM_NUM_DEVICES}  (METHOD=3)"

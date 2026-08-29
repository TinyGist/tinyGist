#!/usr/bin/env bash
set -Eeuo pipefail

export LC_ALL=C

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DATASET_ROOT="${SCRIPT_DIR}/main/dataset_mnist"
readonly DEVICE_ROOT="${TINYGIST_DEVICE_ROOT:-/dev}"
readonly IDF_PY="${TINYGIST_IDF_PY:-idf.py}"
readonly SDKCONFIG_INPUT="${TINYGIST_SDKCONFIG:-}"
readonly BUILD_DIR_INPUT="${TINYGIST_BUILD_DIR:-}"

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    echo "Set TINYGIST_SDKCONFIG and TINYGIST_BUILD_DIR for one homogeneous hardware tier." >&2
    echo "Serial devices and dataset indices are detected automatically." >&2
    exit 2
fi

if ! command -v "${IDF_PY}" >/dev/null 2>&1; then
    echo "Error: ${IDF_PY} was not found. Activate the ESP-IDF environment." >&2
    exit 1
fi

if [[ "$("${IDF_PY}" --version)" != "ESP-IDF v5.2.0" ]]; then
    echo "Error: build_flash.sh requires ESP-IDF v5.2.0." >&2
    exit 1
fi

if [[ -z "${SDKCONFIG_INPUT}" || -z "${BUILD_DIR_INPUT}" ]]; then
    echo "Error: TINYGIST_SDKCONFIG and TINYGIST_BUILD_DIR are required." >&2
    echo "Generate a tier-specific SDKCONFIG with ESP-IDF v5.2.0 menuconfig first." >&2
    exit 2
fi

collect_device_family()
{
    local family="$1"
    local device
    local device_name
    local -a family_devices=()

    for device in "${DEVICE_ROOT}/tty${family}"*; do
        [[ -e "${device}" ]] || continue
        device_name="${device##*/}"
        [[ "${device_name}" =~ ^tty${family}([0-9]+)$ ]] || continue
        family_devices+=("${device}")
    done

    if (( ${#family_devices[@]} > 0 )); then
        printf '%s\n' "${family_devices[@]}" | sort -V
    fi
}

cd "${SCRIPT_DIR}"

if [[ "${SDKCONFIG_INPUT}" = /* ]]; then
    sdkconfig_path="${SDKCONFIG_INPUT}"
else
    sdkconfig_path="${SCRIPT_DIR}/${SDKCONFIG_INPUT}"
fi
if [[ "${BUILD_DIR_INPUT}" = /* ]]; then
    build_dir="${BUILD_DIR_INPUT}"
else
    build_dir="${SCRIPT_DIR}/${BUILD_DIR_INPUT}"
fi

if [[ ! -f "${sdkconfig_path}" ]]; then
    echo "Error: tier-specific SDKCONFIG does not exist: ${sdkconfig_path}" >&2
    exit 1
fi

mapfile -t acm_devices < <(collect_device_family "ACM")
mapfile -t usb_devices < <(collect_device_family "USB")
devices=("${acm_devices[@]}" "${usb_devices[@]}")

if (( ${#devices[@]} == 0 )); then
    echo "Error: no ttyACM<number> or ttyUSB<number> devices found in ${DEVICE_ROOT}." >&2
    exit 1
fi

echo "Detected ${#devices[@]} device(s) in flash order:"
echo "All detected boards will use the same configuration: ${sdkconfig_path}"
for dataset_index in "${!devices[@]}"; do
    if [[ ! -d "${DATASET_ROOT}/device_${dataset_index}" ]]; then
        echo "Error: dataset shard ${DATASET_ROOT}/device_${dataset_index} does not exist; nothing was flashed." >&2
        exit 1
    fi

    printf '  -DDATASET=%d -> %s\n' \
        "${dataset_index}" "${devices[dataset_index]}"
done

for dataset_index in "${!devices[@]}"; do
    device="${devices[dataset_index]}"
    echo
    echo "=== Building and flashing ${device} with -DDATASET=${dataset_index} ==="
    "${IDF_PY}" -B "${build_dir}" -DSDKCONFIG="${sdkconfig_path}" \
        -DDATASET="${dataset_index}" -p "${device}" build flash
done

echo
echo "Finished flashing ${#devices[@]} device(s)."

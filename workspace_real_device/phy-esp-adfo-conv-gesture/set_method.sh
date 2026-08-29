#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s <1|2|3>\n' "$(basename "$0")"
    printf '  1: DFA\n'
    printf '  2: SDFA\n'
    printf '  3: Gist_Ada\n'
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

only_2="false"

case "$1" in
    1)
        method_name="DFA"
        segment_count="1"
        random_segmentation="false"
        adastair="false"
        sb_aggr="false"
        ;;
    2)
        method_name="SDFA"
        segment_count="3"
        random_segmentation="true"
        adastair="false"
        sb_aggr="false"
        ;;
    3)
        method_name="Gist_Ada"
        segment_count="3"
        random_segmentation="false"
        adastair="true"
        sb_aggr="true"
        ;;
    *)
        printf 'Error: method must be 1, 2, or 3.\n' >&2
        usage >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
header="${script_dir}/main/adfo-com.h"

if [[ ! -f "${header}" ]]; then
    printf 'Error: header not found: %s\n' "${header}" >&2
    exit 1
fi

for macro in SEGMENT_COUNT RANDOM_SEGMENTATION ADASTAIR SB_AGGR ONLY_2; do
    match_count="$(grep -Ec "^[[:space:]]*#define[[:space:]]+${macro}[[:space:]]+" "${header}" || true)"
    if [[ "${match_count}" -ne 1 ]]; then
        printf 'Error: expected exactly one #define for %s in %s; found %s.\n' \
            "${macro}" "${header}" "${match_count}" >&2
        exit 1
    fi
done

temporary_file="$(mktemp "${header}.tmp.XXXXXX")"
cleanup() {
    rm -f -- "${temporary_file}"
}
trap cleanup EXIT

sed -E \
    -e "s|^([[:space:]]*#define[[:space:]]+SEGMENT_COUNT[[:space:]]+).*$|\\1${segment_count}|" \
    -e "s|^([[:space:]]*#define[[:space:]]+RANDOM_SEGMENTATION[[:space:]]+).*$|\\1${random_segmentation}|" \
    -e "s|^([[:space:]]*#define[[:space:]]+ADASTAIR[[:space:]]+).*$|\\1${adastair}|" \
    -e "s|^([[:space:]]*#define[[:space:]]+SB_AGGR[[:space:]]+).*$|\\1${sb_aggr}|" \
    -e "s|^([[:space:]]*#define[[:space:]]+ONLY_2[[:space:]]+).*$|\\1${only_2}|" \
    "${header}" > "${temporary_file}"

grep -Eq "^[[:space:]]*#define[[:space:]]+SEGMENT_COUNT[[:space:]]+${segment_count}([[:space:]]|$)" "${temporary_file}"
grep -Eq "^[[:space:]]*#define[[:space:]]+RANDOM_SEGMENTATION[[:space:]]+${random_segmentation}([[:space:]]|$)" "${temporary_file}"
grep -Eq "^[[:space:]]*#define[[:space:]]+ADASTAIR[[:space:]]+${adastair}([[:space:]]|$)" "${temporary_file}"
grep -Eq "^[[:space:]]*#define[[:space:]]+SB_AGGR[[:space:]]+${sb_aggr}([[:space:]]|$)" "${temporary_file}"
grep -Eq "^[[:space:]]*#define[[:space:]]+ONLY_2[[:space:]]+${only_2}([[:space:]]|$)" "${temporary_file}"

if cmp -s -- "${header}" "${temporary_file}"; then
    rm -f -- "${temporary_file}"
    trap - EXIT
    printf 'Method already set to %s in %s\n' "${method_name}" "${header}"
else
    chmod --reference="${header}" "${temporary_file}"
    mv -- "${temporary_file}" "${header}"
    trap - EXIT
    printf 'Method set to %s in %s\n' "${method_name}" "${header}"
fi

printf '  SEGMENT_COUNT=%s\n' "${segment_count}"
printf '  RANDOM_SEGMENTATION=%s\n' "${random_segmentation}"
printf '  ADASTAIR=%s\n' "${adastair}"
printf '  SB_AGGR=%s\n' "${sb_aggr}"
printf '  ONLY_2=%s\n' "${only_2}"

ARG PYTHON_BASE=python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91
ARG DEBIAN_BASE=debian:bookworm@sha256:813017f3d62be4b5891a7acca6a01bdcd4b8513daa81b1ab99d3a50385b26931

FROM ${PYTHON_BASE} AS idf-builder

ARG IDF_VERSION=v5.2
ARG IDF_COMMIT=11eaf41b37267ad7709c0899c284e3683d2f0b5e
ARG DEBIAN_FRONTEND=noninteractive

ENV IDF_PATH=/opt/esp/esp-idf \
    IDF_TOOLS_PATH=/opt/esp/tools \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bison \
        ccache \
        ca-certificates \
        cmake \
        dfu-util \
        flex \
        git \
        gperf \
        libffi-dev \
        libssl-dev \
        libusb-1.0-0 \
        ninja-build \
        pkg-config \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/esp \
    && git clone --depth 1 --single-branch --branch "${IDF_VERSION}" \
        --recurse-submodules --shallow-submodules \
        https://github.com/espressif/esp-idf.git /opt/esp/esp-idf \
    && test "$(git -C /opt/esp/esp-idf rev-parse HEAD)" = "${IDF_COMMIT}" \
    && cd /opt/esp/esp-idf \
    && export IDF_GITHUB_ASSETS=dl.espressif.com/github_assets \
    && ./install.sh esp32,esp32s3 \
    && find /opt/esp/esp-idf -name .git -prune -exec rm -rf {} + \
    && find /opt/esp/esp-idf -type f -name '*.sha256' -delete \
    && rm -rf /opt/esp/tools/dist /root/.cache /tmp/*

FROM ${DEBIAN_BASE} AS qemu-builder

ARG QEMU_REF=esp-develop-8.1.3-20231206
ARG QEMU_COMMIT=1c8a31275ec4fc36734465f447ea23f16eec0998
ARG QEMU_REPOSITORY=https://github.com/espressif/qemu.git
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgcrypt20-dev \
        libglib2.0-dev \
        libpixman-1-dev \
        libslirp-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --single-branch --branch "${QEMU_REF}" \
        "${QEMU_REPOSITORY}" /tmp/qemu \
    && test "$(git -C /tmp/qemu rev-parse HEAD)" = "${QEMU_COMMIT}" \
    && cd /tmp/qemu \
    && ./configure \
        --prefix=/opt/qemu \
        --target-list=xtensa-softmmu \
        --without-default-features \
        --enable-gcrypt \
        --enable-slirp \
        --disable-capstone \
        --disable-debug-info \
        --disable-docs \
        --disable-guest-agent \
        --disable-gtk \
        --disable-tools \
        --disable-user \
        --disable-vnc \
    && ninja -C build qemu-system-xtensa \
    && install -d /opt/qemu/bin /opt/qemu/share/qemu \
    && install -m 0755 build/qemu-system-xtensa /opt/qemu/bin/ \
    && install -m 0644 \
        pc-bios/esp32-v3-rom.bin \
        pc-bios/esp32-v3-rom-app.bin \
        /opt/qemu/share/qemu/ \
    && strip --strip-unneeded /opt/qemu/bin/qemu-system-xtensa \
    && rm -rf /tmp/qemu


FROM ${DEBIAN_BASE} AS emu-binary-validator

COPY binary_bank/emulation/ /validated/

RUN test -f /validated/idf52_flash_images.sha256 \
    && grep -Fxq '# Generated with ESP-IDF v5.2.0 by the two workspace_emulation build_flash.sh scripts.' /validated/idf52_flash_images.sha256 \
    && test "$(find /validated -type f -name 'flash*.bin' | wc -l)" -eq 30 \
    && (cd /validated && sha256sum -c idf52_flash_images.sha256) \
    && test -z "$(find /validated -type f -name '*.log' -print -quit)" \
    && test -z "$(find /validated -type f -name 'flash*.bin' -exec grep -aEl 'v5\.[^2]' {} +)" \
    && for image in $(find /validated -type f -name 'flash*.bin'); do \
        test "$(stat -c %s "${image}")" -eq 4194304 || exit 1; \
        test "$(od -An -tx1 -N1 -j4096 "${image}" | tr -d ' ')" = e9 || exit 1; \
        test "$(od -An -tx1 -N2 -j32768 "${image}" | tr -d ' ')" = aa50 || exit 1; \
        test "$(od -An -tx1 -N1 -j65536 "${image}" | tr -d ' ')" = e9 || exit 1; \
    done \
    && rm -f /validated/idf52_flash_images.sha256 \
    && test -z "$(find /validated -type f -name '*.sha256' -print -quit)"


FROM ${PYTHON_BASE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive

ENV IDF_PATH=/opt/esp/esp-idf \
    IDF_TOOLS_PATH=/opt/esp/tools \
    PATH=/opt/qemu/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    WORKSPACE_SIM_CACHE_DIR=/tmp/workspace_sim_cache \
    MPLCONFIGDIR=/tmp/workspace_sim_cache/matplotlib \
    XDG_CACHE_HOME=/tmp/workspace_sim_cache/xdg \
    PYTHONPATH=/usr/sim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bison \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        dfu-util \
        flex \
        git \
        gperf \
        iproute2 \
        libffi-dev \
        libgcrypt20 \
        libglib2.0-0 \
        libgomp1 \
        libpixman-1-0 \
        libslirp0 \
        libssl-dev \
        libusb-1.0-0 \
        ninja-build \
        pkg-config \
        procps \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY workspace_simulation/requirements.txt /tmp/simulation-requirements.txt

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
# The CUDA installation variant is intentionally kept here for reference. It
# is not used by this CPU-only image.
# ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

# Install simulation dependencies into the default image Python. There is no
# simulation venv to copy into the final image.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url "${TORCH_CPU_INDEX_URL}" \
        torch=="${TORCH_VERSION}" \
        torchvision=="${TORCHVISION_VERSION}" \
    && grep -vE '^(torch|torchvision)==' /tmp/simulation-requirements.txt \
        > /tmp/simulation-requirements-no-torch.txt \
    && python -m pip install -r /tmp/simulation-requirements-no-torch.txt \
    && python -m pip check \
    && python -c 'import torch, torchvision; assert torch.version.cuda is None' \
    && rm -f /tmp/simulation-requirements.txt \
        /tmp/simulation-requirements-no-torch.txt \
    && find /usr/local/lib/python3.11/site-packages -type d \
        -name '__pycache__' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -type f \
        \( -name '*.pyc' -o -name '*.pyo' \) -delete

# CUDA alternative retained as a commented command, per request:
# RUN python -m pip install \
#     --index-url "${TORCH_CUDA_INDEX_URL}" \
#     torch=="${TORCH_VERSION}" \
#     torchvision=="${TORCHVISION_VERSION}"

COPY --from=idf-builder /opt/esp /opt/esp
COPY --from=qemu-builder /opt/qemu /opt/qemu

COPY workspace_simulation/ /usr/sim/
COPY workspace_emulation/ /usr/emu/
COPY workspace_real_device/ /usr/real/
COPY --from=emu-binary-validator /validated/ /usr/binary_bank/emulation/

RUN printf '%s\n' \
        'case ":${PATH}:" in' \
        '    *:/opt/qemu/bin:*) ;;' \
        '    *) export PATH="/opt/qemu/bin:${PATH}" ;;' \
        'esac' \
        '' \
        'get_idf() {' \
        '    if [ -z "${TINYGIST_SIM_PATH+x}" ]; then' \
        '        export TINYGIST_SIM_PATH="${PATH}"' \
        '    fi' \
        '    . /opt/esp/esp-idf/export.sh' \
        '    hash -r' \
        '}' \
        '' \
        'get_sim() {' \
        '    if [ -n "${TINYGIST_SIM_PATH+x}" ]; then' \
        '        export PATH="${TINYGIST_SIM_PATH}"' \
        '        unset TINYGIST_SIM_PATH' \
        '    fi' \
        '    unset IDF_PYTHON_ENV_PATH ESP_IDF_VERSION OPENOCD_SCRIPTS ESP_ROM_ELF_DIR IDF_CCACHE_ENABLE IDF_DEACTIVATE_FILE_PATH IDF_TOOLS_EXPORT_CMD IDF_TOOLS_INSTALL_CMD' \
        '    hash -r' \
        '}' \
        > /etc/profile.d/tinygist.sh \
    && printf '%s\n' '. /etc/profile.d/tinygist.sh' >> /root/.bashrc \
    && chmod 0644 /etc/profile.d/tinygist.sh \
    && test "$(command -v python)" = /usr/local/bin/python \
    && python -c 'import cv2, datasets, fiftyone, librosa, matplotlib, pandas, pycocotools, soundfile, torch, torchvision; import framework_runner; from src.sim_tools.simulation_manager_tool import FederatedLearningSim; assert torch.version.cuda is None' \
    && qemu-system-xtensa --version \
    && test -z "$(ldd /opt/qemu/bin/qemu-system-xtensa | awk '/not found/ { print; exit }')" \
    && bash -lc 'get_idf >/dev/null && idf.py --version 2>/dev/null | grep -Fxq "ESP-IDF v5.2.0" && case "$(command -v python)" in /opt/esp/tools/python_env/idf5.2_py3.11_env/bin/python) ;; *) exit 1 ;; esac && get_sim && test "$(command -v python)" = /usr/local/bin/python && python -c "import torch; assert torch.version.cuda is None"' \
    && test "$(find /usr/binary_bank/emulation -type f -name 'flash*.bin' | wc -l)" -eq 30 \
    && test -z "$(find /usr/binary_bank/emulation -type f -name '*.log' -print -quit)" \
    && test -z "$(find /usr/binary_bank/emulation -type f -name 'flash*.bin' -exec grep -aEl 'v5\.[^2]' {} +)" \
    && for image in $(find /usr/binary_bank/emulation -type f -name 'flash*.bin'); do \
        test "$(stat -c %s "${image}")" -eq 4194304 || exit 1; \
        test "$(od -An -tx1 -N1 -j4096 "${image}" | tr -d ' ')" = e9 || exit 1; \
        test "$(od -An -tx1 -N2 -j32768 "${image}" | tr -d ' ')" = aa50 || exit 1; \
        test "$(od -An -tx1 -N1 -j65536 "${image}" | tr -d ' ')" = e9 || exit 1; \
    done \
    && test "$(find /usr/emu -mindepth 2 -maxdepth 2 -type f -name sdkconfig | wc -l)" -eq 2 \
    && test "$(grep -l '^CONFIG_IDF_INIT_VERSION="5.2.0"$' /usr/emu/*/sdkconfig | wc -l)" -eq 2 \
    && for project in /usr/emu/esp-adfo_conv /usr/emu/esp-adfo_fcn; do \
        test -f "${project}/sdkconfig" || exit 1; \
        grep -Fxq 'CONFIG_IDF_INIT_VERSION="5.2.0"' "${project}/sdkconfig" || exit 1; \
        test -f "${project}/dependencies.lock" || exit 1; \
        grep -Fxq '    version: 5.2.0' "${project}/dependencies.lock" || exit 1; \
        grep -Fxq 'target: esp32' "${project}/dependencies.lock" || exit 1; \
    done \
    && test -z "$(find /usr/real -type f \( -name sdkconfig -o -name sdkconfig.old \) -print -quit)" \
    && test "$(find /usr/real -type f -name 'sdkconfig.defaults.idf52.*' | wc -l)" -eq 8 \
    && for project in \
        /usr/real/phy-esp-adfo-conv-gesture \
        /usr/real/phy-esp-adfo-fcn-mnist; do \
        test -f "${project}/sdkconfig.defaults.idf52.common" || exit 1; \
        test -f "${project}/sdkconfig.defaults.idf52.high" || exit 1; \
        test -f "${project}/sdkconfig.defaults.idf52.mid" || exit 1; \
        test -f "${project}/sdkconfig.defaults.idf52.low" || exit 1; \
        grep -Fxq '# ESP-IDF 5.2 target-neutral TinyGIST intent.' "${project}/sdkconfig.defaults.idf52.common" || exit 1; \
        grep -Fxq 'CONFIG_PARTITION_TABLE_CUSTOM=y' "${project}/sdkconfig.defaults.idf52.common" || exit 1; \
        grep -Fxq 'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y' "${project}/sdkconfig.defaults.idf52.high" || exit 1; \
        grep -Fxq 'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_160=y' "${project}/sdkconfig.defaults.idf52.mid" || exit 1; \
        grep -Fxq 'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_80=y' "${project}/sdkconfig.defaults.idf52.low" || exit 1; \
    done \
    && test -z "$(find /usr/emu /usr/real -type f \( -name 'sdkconfig*' -o -name dependencies.lock \) -exec grep -EH 'ESP-IDF[^0-9]*5\.[^2]|CONFIG_IDF_INIT_VERSION="5\.[^2]|/(home/[^/[:space:]]+|root)/esp/v5\.[^2]|version:[[:space:]]*5\.[^2]' {} +)" \
    && test -z "$(find /usr/sim /usr/emu /usr/real /usr/binary_bank/emulation -type f -iname README.md -print -quit)" \
    && test -z "$(find /usr/sim /usr/emu /usr/real -type d \( -name build -o -name 'build_*' -o -name 'bin_dir_*' \) -print -quit)" \
    && test -z "$(find /usr/sim /usr/emu /usr/real -type f \( -name '*.bin' -o -name '*.elf' -o -name '*.hex' -o -name '*.uf2' -o -name '*.map' -o -name '*.o' -o -name '*.obj' -o -name '*.so' -o -name '*.dylib' -o -name '*.dll' \) -print -quit)" \
    && test -z "$(find /usr/sim /usr/emu /usr/real -type f -name '*.a' -print -quit)" \
    && test "$(find /usr/emu /usr/real -type d -name aifes | wc -l)" -eq 4 \
    && for component in \
        /usr/emu/esp-adfo_conv/components/aifes \
        /usr/emu/esp-adfo_fcn/components/aifes \
        /usr/real/phy-esp-adfo-conv-gesture/components/aifes \
        /usr/real/phy-esp-adfo-fcn-mnist/components/aifes; do \
        test -f "${component}/CMakeLists.txt" || exit 1; \
        test -f "${component}/aifes.h" || exit 1; \
        test -s "${component}/custom_fcn/custom_avgpool.c" || exit 1; \
        test -s "${component}/custom_fcn/custom_avgpool.h" || exit 1; \
        test -n "$(find "${component}" -type f -name '*.c' -print -quit)" || exit 1; \
        test -n "$(find "${component}" -type f -name '*.h' -print -quit)" || exit 1; \
        test -z "$(find "${component}" -type f -name '*.a' -print -quit)" || exit 1; \
        grep -Fq 'custom_avgpool2d_chw_f32_default' "${component}/custom_fcn/custom_avgpool.c" || exit 1; \
    done \
    && test -z "$(find / -xdev -type f -name '*.sha256' -print -quit)"

WORKDIR /usr/sim

CMD ["bash"]

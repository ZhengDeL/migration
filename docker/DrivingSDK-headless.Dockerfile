ARG BASE_IMAGE=local/drivingsdk:26.1.0-a3-arm64-20261010
FROM ${BASE_IMAGE}

# The official A3 image has opencv-python and opencv-python-headless in both
# model environments. The cv2 binary from the GUI build needs libGL.so.1,
# which is absent. Keep only the pinned headless wheel for server training.
COPY wheelhouse/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/wheelhouse/
RUN set -eu; \
    wheel=/tmp/wheelhouse/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl; \
    echo "6efabcaa9df731f29e5ea9051776715b1bdd1845d7c9530065c7951d2a2899eb  $wheel" | sha256sum -c -; \
    for env_name in bevformer sparse4d; do \
      python_bin=/opt/conda/envs/$env_name/bin/python; \
      "$python_bin" -m pip uninstall -y opencv-python opencv-python-headless; \
      "$python_bin" -m pip install --no-index --no-deps "$wheel"; \
      TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 "$python_bin" -c 'import cv2, mmcv; print(cv2.__version__, mmcv.__version__)'; \
    done; \
    rm -f "$wheel"

ARG BASE_IMAGE=local/migration-drivingsdk:26.1.0-a3-headless-20261010
FROM ${BASE_IMAGE}

# Preserve the prebuilt Ascend/OpenMMLab stack. Geospatial binary dependencies
# need a separate conda prefix because ARM64 wheels are absent from PyPI for
# the selected rasterio/Fiona versions.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
RUN /opt/conda/bin/conda create --yes --prefix /opt/worldengine-geo \
      --override-channels --channel conda-forge \
      python=3.10 numpy=1.23.5 rasterio=1.3.11 fiona=1.9.6

COPY docker/worldengine_extra_requirements.txt /tmp/worldengine_extra_requirements.txt
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/conda/envs/bevformer/bin/python \
      -m pip install --no-deps --no-cache-dir --index-url https://pypi.org/simple \
      --target /opt/worldengine-deps -r /tmp/worldengine_extra_requirements.txt

ENV PYTHONPATH=/opt/worldengine-deps:/opt/worldengine-geo/lib/python3.10/site-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/worldengine-geo/lib:${LD_LIBRARY_PATH}

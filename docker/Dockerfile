FROM continuumio/miniconda3:24.1.2-0

WORKDIR /workspace

# CPU-only environment aligned with MolPAL CI (.github/test-env.yaml)
COPY molpal/.github/test-env.yaml /tmp/test-env.yaml

RUN conda env create -f /tmp/test-env.yaml \
    && conda clean -afy

ENV PATH=/opt/conda/envs/molpal-test/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV RAY_TMPDIR=/tmp/ray

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

COPY molpal /workspace/molpal
WORKDIR /workspace/molpal

RUN pip install --no-cache-dir -e . --no-deps

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

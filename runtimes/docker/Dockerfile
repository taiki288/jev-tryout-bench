# Same Python stack as host .venv-fast, on NVIDIA's PyTorch container (26.10, CUDA 13.4 forward-compat)
FROM nvcr.io/nvidia/pytorch:26.10-py3
ENV TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=8 PIP_CONSTRAINT=
RUN pip install --no-cache-dir transformers==5.10.2 huggingface_hub==1.33.0 accelerate==1.15.0 \
      safetensors==0.8.0 einops==0.8.2 flash-linear-attention==0.6.0
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation causal-conv1d==1.7.0

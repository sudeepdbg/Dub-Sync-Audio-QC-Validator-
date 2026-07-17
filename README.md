# Audio Transcription Service (Whisper ASR)

A scalable, on-premise audio transcription service powered by OpenAI's Whisper. This service is optimized for speed and resource efficiency, making it suitable for batch processing, real-time streaming, and integration into larger media pipelines. It supports quantization (`int8`), variable model sizes, and GPU acceleration via CUDA.

---

## Features

- **🎯 Flexible Model Support** – Choose from `tiny`, `base`, `small`, or `medium` models to balance speed and accuracy.
- **⚡ Optimized for CPU & GPU** – Seamlessly switch between CPU and CUDA-enabled GPUs.
- **🚀 Quantized Performance** – Ships with `int8` quantization to drastically reduce latency on CPU hardware.
- **⏱️ Smart Sampling** – Transcribe only the first N seconds of a file to accelerate processing for long-form audio.
- **🐳 Container-Ready** – Designed to run with simple environment variable configuration.

---

## Quick Start

### 1. Prerequisites
- Python 3.9+
- (Optional) NVIDIA GPU with CUDA 11.x+ for `cuda` device support.
- Install dependencies:
  ```bash
  pip install faster-whisper
  python app.py
  
  3. Customize with Environment Variables
Set the desired variables before starting:
export WHISPER_MODEL_SIZE=small
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE_TYPE=int8
export WHISPER_SAMPLE_DURATION_SEC=30.0
python app.py
Configuration Reference
The table below lists all available environment variables and their effects. For maximum performance, tune these values based on your production hardware and latency requirements.

Constant	Default	Valid Values / Options	Description
WHISPER_MODEL_SIZE	base	tiny, base, small, medium	Controls model capacity. Larger models provide higher word-error-rate (WER) improvement but require significantly more compute.
WHISPER_DEVICE	cpu	cpu, cuda	Set to cuda to offload inference to a NVIDIA GPU. Ensure your driver and PyTorch/CUDA stack are compatible.
WHISPER_COMPUTE_TYPE	int8	int8, int16, float16, float32	int8 offers the fastest inference on CPU with minimal accuracy loss. float32 retains full precision but is slower.
WHISPER_SAMPLE_DURATION_SEC	60.0	Positive integer or float (e.g., 120.0)	Limits processing to the first N seconds of the audio file. Reduces latency for long recordings where the initial context is most relevant.

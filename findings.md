# Whisper Implementation Benchmarks & Recommendations

## Overview

This document compares different Whisper ASR implementations across platforms, specifically for the **large-v3-turbo model** which offers 6x faster inference than large-v3 with minimal accuracy loss (1-2% drop).

**Key Stats:**
- Whisper large-v3-turbo: 809M parameters, ~6GB VRAM
- Whisper large-v3: 1.5B parameters, ~10GB VRAM

---

## 1. Mac M-Series Performance

### Performance Ranking (Fastest to Slowest)

| Rank | Implementation | Relative Speed | Inference Time (10 min audio) |
|------|---|---|---|
| 1 | mlx-whisper | **6-10x** | 15-20s |
| 2 | whisper.cpp | **4-6x** | 25-30s |
| 3 | faster-whisper | **2-4x** | 40-60s |
| 4 | OpenAI Whisper (PyTorch) | 1x | ~120s |

### Detailed Comparison

#### **MLX-Whisper** ⭐ RECOMMENDED
- **Backend:** Apple's MLX framework (native Apple Silicon)
- **Speed:** 6-10x faster than original Whisper
- **VRAM:** ~3-4GB
- **Pros:**
  - Fastest on M-series
  - Simple pip install: `pip install mlx-whisper`
  - Native Python integration
  - Low latency
- **Cons:** Apple Silicon only
- **Best for:** Production use, single requests, real-time transcription
- **Code:**
  ```python
  import mlx_whisper
  result = mlx_whisper.transcribe("audio.mp3", path_or_hf_repo="mlx-community/whisper-large-v3-turbo-mlx")
  ```

#### **whisper.cpp**
- **Backend:** C++ with Metal acceleration
- **Speed:** 4-6x faster than original Whisper
- **VRAM:** ~2-3GB
- **Pros:**
  - Explicit Metal optimization
  - Very low resource usage
- **Cons:** Requires compilation, less Python-friendly
- **Best for:** Embedded systems, resource-constrained environments

#### **faster-whisper** (Current Setup)
- **Backend:** CTranslate2 (C++ inference engine)
- **Speed:** 2-4x faster than original Whisper
- **VRAM:** ~4-6GB
- **Pros:**
  - Works across platforms (CPU/GPU)
  - Easy setup
  - Good starting point
- **Cons:** No native Metal optimization on Mac
- **Best for:** Cross-platform compatibility

#### **OpenAI Whisper (PyTorch)**
- **Speed:** Baseline (1x)
- **Limitations:** Even with `device="mps"` (Metal), PyTorch has limited MPS optimization
- **Not recommended for Mac production use**

### Mac M2 Recommendation
**Switch to MLX-Whisper** for 2-3x speed improvement over your current setup with minimal changes.

---

## 2. GPU Server Performance (NVIDIA)

### Performance Ranking (large-v3-turbo)

| Rank | Implementation | Relative Speed | Best VRAM | RTF (Real-Time Factor) |
|------|---|---|---|---|
| 1 | insanely-fast-whisper | **8-12x** | 12GB+ | 200-400x |
| 2 | faster-whisper | **4-6x** | 6GB+ | 100-200x |
| 3 | WhisperX | **4x** | 6GB+ | 100x |
| 4 | OpenAI Whisper | 1x | 10GB | ~25x |

**RTF Example:** On H100 GPU, a 1-hour audio file transcribes in:
- insanely-fast-whisper: ~10 seconds
- faster-whisper: ~20 seconds
- OpenAI Whisper: ~2.5 minutes

### Detailed Comparison

#### **insanely-fast-whisper** ⭐ FOR MAX THROUGHPUT
- **Backend:** HuggingFace Transformers + FlashAttention-2
- **Speed:** 8-12x faster than original Whisper
- **VRAM Required:** 12GB+ (A100, H100, RTX 4090)
- **Pros:**
  - Maximum throughput
  - Best for batch processing
  - Scales well with concurrent requests (1300x RTF on H100)
- **Cons:**
  - Complex setup (FlashAttention-2 compilation)
  - Overkill for small requests
  - High VRAM requirements
- **Best for:** Large-scale batch transcription, high-volume production
- **Hardware Requirements:** 12GB+ VRAM

#### **faster-whisper** ⭐ RECOMMENDED FOR GPU
- **Backend:** CTranslate2 (C++ inference engine)
- **Speed:** 4-6x faster than original Whisper
- **VRAM Required:** 6GB+
- **Pros:**
  - Simple setup (pip install)
  - Balanced speed/complexity
  - Works with modest GPUs (RTX 3070+)
  - Production-ready
- **Cons:** Not as fast as insanely-fast-whisper for batch jobs
- **Best for:** General production use, mixed single/batch requests
- **Code:**
  ```python
  from faster_whisper import WhisperModel
  model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
  segments, info = model.transcribe("audio.mp3")
  ```

#### **WhisperX**
- **Backend:** HuggingFace Transformers
- **Speed:** 4x faster than original Whisper
- **Special Feature:** Word-level timestamps + speaker diarization
- **Best for:** Detailed transcription with speaker identification
- **Trade-off:** Slower than insanely-fast-whisper, but includes features

#### **OpenAI Whisper (PyTorch)**
- **Speed:** Baseline (1x)
- **VRAM:** 10GB+
- **Not recommended for production GPU servers**

### GPU Server Recommendation
- **Default choice:** faster-whisper (best balance)
- **High-volume processing:** insanely-fast-whisper (if hardware supports it)
- **With diarization needed:** WhisperX

---

## 3. Cross-Platform Summary Table

| Metric | Mac M2 | GPU Server |
|--------|--------|-----------|
| **Fastest Option** | mlx-whisper (6-10x) | insanely-fast-whisper (8-12x) |
| **Recommended Option** | mlx-whisper | faster-whisper |
| **Easiest Setup** | mlx-whisper | faster-whisper |
| **Resource Usage** | 3-4GB | 6-12GB |
| **Setup Complexity** | Simple | Simple-Moderate |

---

## 4. Implementation Recommendations

### For Your M2 Mac:
```python
# Current: 40-60s per 10min audio
# Switch to:
import mlx_whisper
result = mlx_whisper.transcribe(
    "audio.mp3",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo-mlx"
)
# Expected: 15-20s per 10min audio (3x faster)
```

### For GPU Server (NVIDIA):
```python
# Balanced approach (recommended)
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# OR for maximum throughput
from insanely_fast_whisper import InferenceEngine
engine = InferenceEngine(
    "large-v3-turbo",
    device="cuda",
    flash_attention=True  # Requires FlashAttention-2
)
```

---

## 5. Specifications & VRAM Requirements

| Model | Original Whisper | faster-whisper | mlx-whisper | whisper.cpp | insanely-fast |
|-------|---|---|---|---|---|
| **large-v3-turbo** | 6GB | 6GB | 3-4GB | 2-3GB | 12GB+ |
| **large-v3** | 10GB | 8GB | 4-5GB | 3-4GB | 16GB+ |
| **base** | 1GB | 1GB | 500MB | 500MB | 2GB |

---

## 6. Sources

- [Mac Whisper Speedtest Benchmarks](https://github.com/anvanvan/mac-whisper-speedtest)
- [mlx_whisper vs whisper.cpp Benchmark (2026)](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)
- [Whisper ASR in MLX Performance](https://medium.com/@ingridwickstevens/whisper-asr-in-mlx-how-much-faster-is-speech-recognition-really-5389e3c87aa2)
- [Choosing Between Whisper Variants (Modal)](https://modal.com/blog/choosing-whisper-variants)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [Whisper on Mac M4 Analysis](https://dev.to/theinsyeds/whisper-speech-recognition-on-mac-m4-performance-analysis-and-benchmarks-2dlp)
- [Whisper Large V3 Turbo Overview](https://medium.com/axinc-ai/whisper-large-v3-turbo-high-accuracy-and-fast-speech-recognition-model-be2f6af77bdc)

---

## 7. Decision Flow Chart

```
For Mac M2?
├─ YES → Use mlx-whisper ⭐
│        (6-10x faster, simple setup)
└─ NO → For GPU Server?
         ├─ YES, High Volume → insanely-fast-whisper (if 12GB+ VRAM)
         ├─ YES, Mixed Use → faster-whisper ⭐
         └─ YES, Need Diarization → WhisperX
```

---

# セットアップ（Jetson AGX Orin 64GB）

計測した環境: L4T R39.2（JetPack 7 系）、CUDA 13.2 ドライバ、電源モード MAXN + `sudo jetson_clocks`。

## 1. 共通

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks   # MAXN（nvpmodel の変更は再起動を求められることがある）
sudo usermod -aG docker $USER              # docker を sudo なしで使う（再ログイン）
```

モデルは Hugging Face のキャッシュ（`~/.cache/huggingface`）に置く。スクリプトはネットにつながず、ここから読む。

```bash
.venv/bin/hf download Cloudflare/clef-flash
.venv/bin/hf download Vishva007/clef-W4A16-AutoRound-GPTQ        # Clef INT4（host / SGLang）
.venv/bin/hf download moeshawky/clef-omni-int4-autoround         # Clef-omni INT4（host / vLLM）
.venv/bin/hf download ggml-org/Clef-Flash-GGUF Clef-Flash-BF16.gguf mmproj-Clef-Flash-BF16.gguf
.venv/bin/hf download ggml-org/Clef-GGUF Clef-Q4_K_M.gguf mmproj-Clef-BF16.gguf
```

## 2. ランタイムごとの準備

| ランタイム | 準備 |
|---|---|
| host（transformers） | `setup/setup_venv.sh`（`.venv` を作る。ベンチのクライアントもこれを使う） |
| Docker（NVIDIA PyTorch） | `docker build -t clef-torch:26.10 runtimes/docker` |
| vLLM | `docker build -t clef-vllm:0.32.0-patched runtimes/vllm` |
| SGLang | `docker pull lmsysorg/sglang:dev-cu13-clef`（51.9GB） |
| llama.cpp | `setup/build_llamacpp.sh`（`~/llama.cpp` にビルド） |
| ollama | `docker pull ollama/ollama:latest`（0.40.15 で計測） |

## 注意

- メモリ 64GB を CPU と GPU で共有している。足りなくなると Jetson ごと固まるので、どのスクリプトも空きが 3GB（`MIN_AVAIL_MB`）を切ったら処理を止める
- サーバーは一度に 1 つだけ動かす
- 計測中にビルドやダウンロードをしない（時間がぶれる）

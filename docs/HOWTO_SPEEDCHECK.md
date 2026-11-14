# 各ランタイムの速さを手元で確かめる手順（Jetson AGX Orin）

すべて Orin 上のリポジトリのフォルダ（以下 `~/jev-tryout-bench`）で実行する。計測で使った質問・選択肢・画像は `dataset/` にある。

- **一度に動かすサーバーは 1 つだけ**にする（メモリ 64GB を共有しているため）。終わったら必ず止める
- サーバーを立てて `try_request.py` で質問を送ると、1 回ごとの時間と答えが出る
- `try_request.py` のオプション
  - `--k 10|50|100`: 選択肢の数
  - `--img 0|1024`: 画像なし / あり
  - `--q 0`: 質問の番号（0〜9、`-1` で 10 問全部）
  - `--n 3`: 同じ質問を送る回数
- 1 回目は準備（ウォームアップ）を含むので遅いことがある。平均は 2 回目以降で計算する
- 下の「目安」は本番計測（選択肢 100 の平均）の値

```bash
cd ~/jev-tryout-bench
```

---

## 1. vLLM — Clef-omni INT4（一番速い）

```bash
runtimes/vllm/serve.sh clef-omni-int4 moeshawky/clef-omni-int4-autoround \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.55
# 準備完了まで待つ（重みの読み込みに数分〜9 分）
until docker logs clef-omni-int4 2>&1 | grep -q "Application startup complete"; do sleep 10; done

python3 bench/try_request.py --url http://localhost:8000/v1/systemone --model clef-omni-int4 --k 100 --img 0
python3 bench/try_request.py --url http://localhost:8000/v1/systemone --model clef-omni-int4 --k 100 --img 1024

docker rm -f clef-omni-int4   # 停止
```

目安: 画像なし 1.0 秒、画像あり 1.4 秒

INT8 の場合: `clef-omni-int8 moeshawky/clef-omni-int8-autoround --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 --max-num-seqs 16`（目安 1.2 秒 / 1.65 秒）

---

## 2. SGLang — Clef INT4 / Clef-flash bf16

```bash
# Clef INT4（初回は Orin 向けのカーネルを作るので 11 分ほどかかる。作ったものは保存済みなので 2 回目からは速い）
runtimes/sglang/serve.sh clef-int4 Vishva007/clef-W4A16-AutoRound-GPTQ \
  --mem-fraction-static 0.55 --watchdog-timeout 7200 --skip-server-warmup
until docker logs clef-int4 2>&1 | grep -q "Uvicorn running"; do sleep 10; done

python3 bench/try_request.py --url http://localhost:30000/v1/systemone --model clef-int4 --k 100 --img 0
python3 bench/try_request.py --url http://localhost:30000/v1/systemone --model clef-int4 --k 100 --img 1024

docker rm -f clef-int4   # 停止
```

目安: 画像なし 4.8 秒、画像あり 6.8 秒

Clef-flash の場合:
```bash
runtimes/sglang/serve.sh clef-flash Cloudflare/clef-flash --mem-fraction-static 0.55 --skip-server-warmup
until docker logs clef-flash 2>&1 | grep -q "Uvicorn running"; do sleep 10; done
python3 bench/try_request.py --url http://localhost:30000/v1/systemone --model clef-flash --k 100 --img 0
docker rm -f clef-flash
```
目安: 画像なし 2.7 秒、画像あり 3.95 秒

---

## 3. llama.cpp — Clef-flash BF16 / Clef Q4_K_M

```bash
S=$(ls -d ~/.cache/huggingface/hub/models--ggml-org--Clef-Flash-GGUF/snapshots/*)
~/llama.cpp/build/bin/llama-server -m $S/Clef-Flash-BF16.gguf --mmproj $S/mmproj-Clef-Flash-BF16.gguf \
  -ngl 99 -c 8192 -b 8192 -ub 8192 -np 1 --port 8080 &
until curl -sf localhost:8080/health >/dev/null; do sleep 2; done

python3 bench/try_request.py --url http://localhost:8080/v1/systemone --k 100 --img 0
python3 bench/try_request.py --url http://localhost:8080/v1/systemone --k 100 --img 1024

kill %1   # 停止（別のシェルからなら: pkill -x llama-server）
```

目安: 画像なし 2.5 秒、画像あり 5.3 秒

Clef の場合は `models--ggml-org--Clef-GGUF` の `Clef-Q4_K_M.gguf` と `mmproj-Clef-BF16.gguf`（読み込みに 9 分ほど）。目安: 10.0 秒 / 16.6 秒

---

## 4. ollama — clef-flash（Q8_0）/ clef（Q4_K_M）

```bash
docker rm -f ollama 2>/dev/null   # 前回のコンテナが残っていれば消す
docker run -d --name ollama --gpus all --ipc=host -p 11434:11434 \
  -v ~/jev-tryout-bench/runtimes/ollama/models:/root/.ollama -e OLLAMA_KEEP_ALIVE=-1 ollama/ollama:latest

# --raw: ollama は画像を data URL ではなく base64 のまま受け取る
python3 bench/try_request.py --url http://localhost:11434/v1/systemone --model clef-flash --k 100 --img 0 --raw
python3 bench/try_request.py --url http://localhost:11434/v1/systemone --model clef-flash --k 100 --img 1024 --raw

docker rm -f ollama   # 停止
```

最初のリクエストでモデルを読み込むので、1 回目だけ 2〜6 分かかる。
目安: clef-flash 2.8 秒 / 5.6 秒、clef 9.8 秒 / 15.7 秒

---

## 5. host（transformers）/ Docker（NVIDIA PyTorch）— Clef-flash bf16

サーバーではなく、プログラムの中でモデルを動かして測る（本番と同じ `bench2.py`、量を減らして実行）。

```bash
# host
.venv/bin/python bench/bench2.py --model Cloudflare/clef-flash --options 100 --image 0,1024 \
  --n-questions 1 --reps 3 --label try-host --out /tmp/try_host

# Docker（NVIDIA PyTorch 26.10）
runtimes/docker/run.sh --model Cloudflare/clef-flash --options 100 --image 0,1024 \
  --n-questions 1 --reps 3 --label try-docker --out try_docker   # docker logs -f jev-docker-bench で進み具合を見る
docker logs -f jev-docker-bench
```

各行の `K=100 img=0 q0: 2050±… ms` が平均の時間。
目安: host 2.25 秒 / 3.67 秒、Docker 2.05 秒 / 3.32 秒

---

## 計測データ（本番）

| 場所 | 中身 |
|---|---|
| `RESULTS.md` | 全結果のまとめ（冒頭にランタイム横断の表） |
| `dataset/questions.json`, `options_K*.json` | 質問と選択肢 |
| `dataset/records_K*_img*.json` | 実際に送ったリクエストの中身 |
| `dataset/images/test_1024.png` | 画像ありで使った画像 |
| `dataset/answers.csv` | 全ランタイム・全条件の答え（1 回目） |
| `results/*.runs.jsonl` | 1 回ごとの時間・答え・確率 |
| `results/*.summary.jsonl` | 条件ごとの集計（時間・電力・メモリ） |

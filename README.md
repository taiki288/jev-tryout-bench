# jev-tryout-bench

Jetson AGX Orin 64GB で、Jev 互換の判定モデル（System One / decision model）を色々なランタイムで動かし、
**応答時間・メモリ・電力・答えのゆらぎ**を同じ条件で比べるためのリポジトリ。

判定モデルは文章を生成せず、入力を 1 回通して各選択肢の確率を出す（`POST /v1/systemone`）。
なので「1 回の判定に何秒かかるか」がそのまま実用上の速さになる。

これまでの結果（Cloudflare Clef / Clef-flash / Clef-omni）: [docs/RESULTS_clef.md](docs/RESULTS_clef.md)

## 結果の早見表（選択肢 100、平均応答時間 秒、モデル読み込み済み）

| モデル | 一番速いランタイム | 画像なし | 画像あり（1024px） |
|---|---|---:|---:|
| Clef-omni（30B-A3B, INT4） | vLLM | 1.01 | 1.43 |
| Clef-flash（9B, bf16） | Docker（NVIDIA PyTorch） | 2.05 | 3.32 |
| Clef（27B, INT4） | SGLang | 4.76 | 6.83 |

全モデル × 全ランタイム × 選択肢 10/50/100 の表は [docs/RESULTS_clef.md](docs/RESULTS_clef.md) の冒頭にある。

## 構成

```
bench/
  bench2.py        transformers でモデルを直接動かす計測（host / Docker）
  bench_vllm.py    動いているサーバーの /v1/systemone に送る計測（vLLM / SGLang / llama.cpp / ollama）
  bench.py         tegrastats（電力・メモリ・GPU 使用率）の記録、テスト画像の生成
  try_request.py   手で 1 問ずつ送って時間と答えを見る（標準ライブラリだけ）
  summarize2.py    results/*.summary.jsonl → Markdown の表
  export_dataset.py  使った質問・選択肢・答えを dataset/ に書き出す
runtimes/<runtime>/  サーバーの起動・計測スクリプト（common.sh にメモリ監視などの共通処理）
tools/selftest.sh  全ランタイムのスクリプトが動くかを最小の条件で確かめる（約40分）
setup/             環境の作り方（venv、llama.cpp のビルド、Docker イメージ）
dataset/           計測に使った質問・選択肢・画像・答え
results/           計測結果（1 回ごと: *.runs.jsonl、条件ごとの集計: *.summary.jsonl）
docs/              結果のまとめ、手で速さを確かめる手順
```

## 使い方

準備は [setup/README.md](setup/README.md)。準備ができたら `tools/selftest.sh` で全ランタイムが動くか確かめられる（`tools/selftest.sh vllm` のように 1 つだけも可）。リポジトリは Orin 上に置き、計測も Orin の中で完結させる
（リクエストは localhost に送るので、ネットワークの遅延は入らない）。

```bash
# 例1: host（transformers）で Clef-flash
runtimes/host/run.sh --model Cloudflare/clef-flash --label host-bf16 --out results/clef-flash_host_bf16

# 例2: vLLM で Clef-omni INT4（サーバー起動 → 計測 → 停止）
runtimes/vllm/serve.sh clef-omni-int4 moeshawky/clef-omni-int4-autoround --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.55
runtimes/vllm/bench.sh clef-omni-int4 vllm-int4 results/clef-omni_vllm_int4
docker rm -f clef-omni-int4

# 例3: llama.cpp（起動から停止まで 1 本で）
runtimes/llamacpp/run.sh ggml-org/Clef-Flash-GGUF Clef-Flash-BF16.gguf mmproj-Clef-Flash-BF16.gguf llamacpp-bf16 results/clef-flash_llamacpp_bf16

# 結果を表にする
.venv/bin/python bench/summarize2.py results/clef-flash_host_bf16.summary.jsonl
```

各ランタイムのコマンド例は、スクリプトの先頭のコメントにある。手で 1 問ずつ試すなら [docs/HOWTO_SPEEDCHECK.md](docs/HOWTO_SPEEDCHECK.md)。

## 条件の変え方

どの計測スクリプトも、最後に次のオプションを付けられる。

| オプション | 既定値 | 意味 |
|---|---|---|
| `--options 10,50,100` | 10,50,100 | 選択肢の数（最大 100。ラベルは 20 トピック × 5 種類） |
| `--image 0,1024` | 0,1024 | 画像の一辺のピクセル数（0 = 画像なし）。`0,224,448,1024` のように何個でも |
| `--n-questions 10` | 10 | 質問（問い合わせ文）の数（最大 10） |
| `--reps 10` | 10 | 同じ質問を何回送るか（ゆらぎを見る） |
| `--warmup 2` | 2 | 計測前の空回し |

質問・選択肢は `bench/bench2.py` の `TICKETS` / `LABELS` / `make_record()` で決まる（乱数の種は固定なので、誰が回しても同じ）。
テスト画像は `bench/bench.py` の `test_image()` が作る。

## 計測しているもの

- **応答時間**: 1 回の判定にかかった時間（モデル読み込み済み、ウォームアップ後）。サーバー型は HTTP の往復込み（localhost）
- **メモリ**: tegrastats の RAM（CPU と GPU で共有）、host / Docker は PyTorch のピーク
- **電力**: VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0（100ms ごと）、1 回あたりのエネルギー
- **ゆらぎ**: 同じ入力を繰り返したときの時間の標準偏差、答え（argmax）の一致率、確率のぶれ
- **正答率**: 各質問に決めた正解ラベルを選べた割合（量子化やランタイムの比較用の目安）

## 結果を追加するとき

1. `results/<モデル>_<ランタイム>_<量子化>` の名前で計測する
2. `bench/summarize2.py` で表にして、`docs/` のまとめに追記する
3. 使ったモデルの repo、量子化、ランタイムの版、起動オプションを書いておく

## 注意

- メモリを CPU と GPU で共有しているので、足りなくなると Jetson ごと固まる。スクリプトは空きが 3GB を切ったら処理を止める
- サーバーは一度に 1 つだけ。計測中にビルドやダウンロードをしない
- 電源モードは MAXN + `jetson_clocks` で計測している

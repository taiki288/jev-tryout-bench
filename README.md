# jev-tryout-bench

Jetson AGX Orin 64GB で、Jev 互換の判定モデル（System One / decision model）を色々なランタイムで動かし、
**応答時間・メモリ・電力・答えのゆらぎ**を同じ条件で比べるためのリポジトリ。

判定モデルは文章を生成せず、入力を 1 回通して各選択肢の確率を出す（`POST /v1/systemone`）。
なので「1 回の判定に何秒かかるか」がそのまま実用上の速さになる。

調べられること:
- **速さ**: ランタイム・量子化・選択肢の数・画像サイズを変えたときの応答時間、メモリ、電力 → [docs/RESULTS_clef.md](docs/RESULTS_clef.md)
- **精度**: 答えがはっきりした 330 問で、モデルごとの正解率と時間 → [docs/RESULTS_accuracy.md](docs/RESULTS_accuracy.md)

## 精度の早見表（330 問、選択肢 20、画像 448px）

| モデル | ランタイム | 正解率 | 画像付き 1 問 |
|---|---|---:|---:|
| d1-2B（bf16） | llama.cpp | 98.5% | 0.15 秒 |
| Smaug-Decisions（Gemma 4 E4B） | TensorRT（Edge-LLM） | 97.6% | 0.47 秒 |
| Smaug-Decisions（Gemma 4 E4B, INT8 W8A8） | TensorRT（Edge-LLM） | 97.0% | 0.35 秒 |
| Smaug-Decisions（Gemma 4 E4B） | host | 97.6% | 0.94 秒 |
| d3-nano（bf16） | llama.cpp | 97.3% | 0.29 秒 |
| Clef（27B, Q4_K_M） | llama.cpp | 96.7% | 3.16 秒 |
| Clef-flash（9B, bf16） | host | 95.8% | 0.85 秒 |
| Intern-Decision 2B | host | 94.5% | 0.21 秒 |

小型の判定モデルで Clef-flash と同じかそれ以上の正解率が出る。量子化（Q8〜Q2）は正解率をほぼ落とさないが、速くもならない。
TensorRT（Edge-LLM）は Smaug で正解率を変えずに約 2 倍、計算も INT8 にする量子化（W8A8）で約 2.7 倍速くなった。

## 速さの早見表（選択肢 100、平均応答時間 秒、モデル読み込み済み）

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
  evalset.py       精度用の問題セット（dataset/eval_v1）を作る
  eval_acc.py      精度の計測（1 問ずつ答えと時間を記録）
  adapters.py      モデルの種類ごとの問い合わせ方（/v1/systemone、Clef、Intern-Decision、Smaug）
  eval_report.py   精度の結果 → Markdown の表
runtimes/<runtime>/  サーバーの起動・計測スクリプト（common.sh にメモリ監視などの共通処理。edgellm/ は TensorRT のビルドと計測）
tools/selftest.sh  全ランタイムのスクリプトが動くかを最小の条件で確かめる（約40分）
tools/acc_run.sh   一覧ファイルのモデルを順に精度計測する
tools/acc_lists/   精度計測するモデルの一覧（small.txt: 小型モデル、quant.txt: Clef の量子化、tensorrt.txt: TensorRT）
setup/             環境の作り方（venv、llama.cpp のビルド、Docker イメージ）
dataset/           計測に使った質問・選択肢・画像・答え（eval_v1/ は精度用の 330 問）
results/           計測結果（1 回ごと: *.runs.jsonl、条件ごとの集計: *.summary.jsonl。精度の結果 acc_* は git に入れない）
docs/              結果のまとめ、手順（HOWTO_SPEEDCHECK.md: 手で速さを確かめる、HOWTO_accuracy.md: 精度の測り方）
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

精度を測るときは、一覧ファイルを書き換えて `tools/acc_run.sh` に渡す（詳しくは [docs/HOWTO_accuracy.md](docs/HOWTO_accuracy.md)）。

```bash
tools/acc_run.sh tools/acc_lists/small.txt download   # 重みを落とす
tools/acc_run.sh tools/acc_lists/small.txt run        # 一覧を上から順に測る（終わったものは飛ばす）
python3 bench/eval_report.py results/acc_*.acc.jsonl --options 20 --image 448 --ref clef-flash-bf16-host
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

速さの結果:

1. `results/<モデル>_<ランタイム>_<量子化>` の名前で計測する
2. `bench/summarize2.py` で表にして、`docs/` のまとめに追記する
3. 使ったモデルの repo、量子化、ランタイムの版、起動オプションを書いておく

精度の結果は、一覧ファイル（`tools/acc_lists/`）に行を足して測り、`bench/eval_report.py` の表を [docs/RESULTS_accuracy.md](docs/RESULTS_accuracy.md) に追記する。

## 注意

- メモリを CPU と GPU で共有しているので、足りなくなると Jetson ごと固まる。スクリプトは空きが 3GB を切ったら処理を止める
- サーバーは一度に 1 つだけ。計測中にビルドやダウンロードをしない
- 電源モードは MAXN + `jetson_clocks` で計測している

# 精度の測り方（eval_v1）

速さの計測（`bench/bench2.py` など）とは別に、**答えが合っているか**を同じ問題で比べる仕組み。
問題は `dataset/eval_v1/`（330 問、内容は [dataset/eval_v1/README.md](../dataset/eval_v1/README.md)）。
結果のまとめは [RESULTS_accuracy.md](RESULTS_accuracy.md)。

## 全体の流れ

```
tools/acc_lists/*.txt   測るモデルの一覧（1 行 = 1 回の計測）。ここを書き換えて条件を変える
        │
tools/acc_run.sh        一覧を上から順に、ダウンロード → 計測（終わっているものは飛ばす）
        │
        ├─ host:<backend>  bench/eval_acc.py がモデルを直接読み込む（bench/adapters.py）
        ├─ llamacpp        runtimes/llamacpp/run.sh が llama-server を立てて eval_acc.py が問い合わせる
        └─ edgellm         runtimes/edgellm/run.sh が TensorRT Edge-LLM のエンジンを読み込む（先に build.sh でビルド）
        │
results/acc_<name>_<runtime>_<precision>.acc.jsonl   1 問 1 行（答え、正誤、正解の確率、時間、トークン数）
        │
bench/eval_report.py    Markdown の表にする
```

## 使い方

```bash
tools/acc_run.sh tools/acc_lists/small.txt download    # 先に重みを落とす（計測中はダウンロードしない）
tools/acc_run.sh tools/acc_lists/small.txt run         # 全部測る（small.txt は約 3 時間）
tools/acc_run.sh tools/acc_lists/small.txt run d1-2b   # 名前を指定すると、その行だけ

# 表にする（選択肢 20 個、画像 448px の結果だけを並べる。--ref の答えとの一致率も出す）
python3 bench/eval_report.py results/acc_*.acc.jsonl --options 20 --image 448 --ref clef-flash-bf16-host
```

試しに数問だけ回すときは `ACC_OPTS` で一覧の `opts` を上書きする。結果のファイル名が本番と同じになるので、
試したあとは `results/acc_<name>_*` を消す（すでにあるものは飛ばされ、行は追記される）。

```bash
ACC_OPTS="--options 10 --image 448 --limit 1" tools/acc_run.sh tools/acc_lists/small.txt run intern-decision-2b
```

結果ファイル（`results/acc_*`）は git に入れていない。

## 一覧ファイルの書き方（tools/acc_lists/*.txt）

```
opts --options 10,20 --image 224,448,1024        # このファイルの全行に付ける eval_acc.py のオプション

# ランタイム   名前                精度     repo                          GGUF                       mmproj
host:intern    intern-decision-2b  bf16     internlm/Intern-Decision-2B
llamacpp       d1-2b               q4_k_m   LiquidAI/d1-2B-GGUF           d1-2B-Q4_K_M.gguf          mmproj-d1-2B-F16.gguf
llamacpp       clef-flash          q2_k     bartowski/Cloudflare_clef-flash-GGUF  Cloudflare_clef-flash-Q2_K.gguf  ggml-org/Clef-Flash-GGUF:mmproj-Clef-Flash-BF16.gguf
host:clef      clef                nf4      Cloudflare/clef               -- --bnb nf4
```

- **ランタイム**: `host:<backend>`（transformers でこのプロセスに読み込む。backend は `clef` / `intern` / `smaug`）、`llamacpp`、
  `edgellm`（5 列目にエンジン名。`runtimes/edgellm/build.sh <repo> <エンジン名>` で先に作る。Smaug と Intern-Decision に対応）
- **名前・精度**: 結果のファイル名（`results/acc_<名前>_<ランタイム>_<精度>`）と表のラベル（`<名前>-<精度>-<ランタイム>`）になる。精度は自由に書いてよい
- **mmproj**（llamacpp のみ）: 同じ repo のファイル名か、`<別の repo>:<ファイル名>`（mmproj を持たない GGUF 用）
- **`--` の後**: その行だけに付ける eval_acc.py のオプション（例: `--bnb nf4`、`--tasks intent`）

### eval_acc.py のオプション

| オプション | 既定値 | 意味 |
|---|---|---|
| `--options 10,20` | 10,20 | 選択肢の数 K（最大 30。Smaug は 20 まで） |
| `--image 448` | 448 | 画像付きの問題の画像サイズ（一辺 px）。`224,448,1024` のように複数可。画像なしの問題には効かない |
| `--tasks ...` | 全部 | `intent,shape,count,word,error,scene,receipt` から選ぶ |
| `--limit N` | 0（全部） | 各タスクの先頭 N 問だけ |
| `--warmup 2` | 2 | 計測前の空回し |

## モデルを足すとき

1. **llama.cpp で動くモデル**（GGUF に判定用の情報が入っているもの）: 一覧に `llamacpp` の行を足すだけ。
   llama.cpp の `/v1/systemone` が対応しているモデルの種類は `~/llama.cpp/tools/server/server-decision.cpp` にある
2. **vLLM / SGLang / ollama で動くモデル**: サーバーを立てて `bench/eval_acc.py --backend systemone --url ...` を直接使う（例は eval_acc.py の先頭）
3. **どのランタイムも対応していないモデル**: `bench/adapters.py` にクラスを足す。
   `ask(rec, img)` が「選択肢 id → 確率」と入力トークン数を返せばよい（モデルカードのプロンプトの形をそのまま書く）。
   `BACKENDS` に登録すると、一覧に `host:<登録名>` で書ける

### TensorRT（Edge-LLM）で測るとき

```bash
runtimes/edgellm/build.sh abacusai/Smaug-Decisions smaug-fp16 --max-image-tokens 560 --max-image-tokens-per-image 560   # 約 9 分
tools/acc_run.sh tools/acc_lists/tensorrt.txt run smaug-decisions
```

エンジンは `~/edgellm-engines/<エンジン名>`、設定を補正したチェックポイント（HF キャッシュへのリンク）は `~/edgellm-ckpt/<エンジン名>` にできる。
`build.sh` の後ろに `trtedge build` のオプションを付けられる（`--max-input-len` などは build.sh の既定値を上書き）。
Edge-LLM は 1 トークン生成して上位の log 確率を返す作りなので、`bench/adapters.py` の `EdgeLLM` は選択肢の記号に同じ logit bias を足して確率を読む。
ほかのモデルの種類を足すときは、`EdgeLLM.ask()` にそのモデルのプロンプトを足す。

## 問題を足すとき

`bench/evalset.py` の `TASKS` と、問題を作る部分（`build()`）に足して、`eval_v2` として作り直す
（`python3 bench/evalset.py dataset/eval_v2`、`--eval-dir dataset/eval_v2` で使う）。
eval_v1 を書き換えると、これまでの結果と比べられなくなる。

## 注意

- 速さは同じ条件で測ったものだけを比べる。host（transformers）と llama.cpp では処理の仕組みが違うので、時間は目安
- 時間は 1 問ずつ順番に投げたときの値（モデル読み込み済み、ウォームアップ後）。llama.cpp は HTTP の往復込み
- 計測中に別の計測・ビルド・ダウンロードをしない

# Diffusers Engine Q21

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Hugging Face Diffusersを基盤に、Qwen-Image 2.1の高速・低VRAMローカル推論へ
最適化した独立・非公式の推論エンジンです。旧
[`Local-Image-Studio`](https://github.com/animede/Local-Image-Studio) の全履歴と成果を
引き継ぎ、今後の修正と最適化はこのリポジトリで継続します。

> **License notice:** This repository's original source code and documentation
> are licensed under Apache-2.0. Qwen Image 2.1, Viggle Turbo, and optional
> Orbit LoRA weights are governed by the non-commercial Qwen Research License
> and are not included in this repository. See
> [Third-party notices](THIRD_PARTY_NOTICES.md).

公式Diffusersモデル [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) をローカルGPUで実行する、APIサーバーとWebフロントエンドです。QwenおよびHugging Faceの公式製品ではありません。

## 構成

```text
frontend/               独立したHTML/CSS/JavaScript UI（port 5173）
backend/app/            FastAPI推論サーバー（port 8000）
scripts/download_model.py
models/Qwen-Image-2.1/  公式Diffusersモデル（Git管理外）
outputs/                生成PNGと条件JSON（Git管理外）
.venv/                  Python仮想環境（Git管理外）
```

APIはモデルをGPUに常駐させ、生成ジョブを1件ずつ処理します。フロントエンドはAPIと分離されており、REST APIだけを別クライアントから利用することもできます。

高速化・低VRAM化の仕組み、全プロファイルの比較、GPU容量別の選び方は[高速化・低VRAM化ガイド](docs/performance-low-vram.md)にまとめています。

実験機能として、視点を相対角度で変更するOrbit LoRAについても、NVFP4、Viggle併用、RGB/RGBA、人物・キャラクターを24GB GPUで検証しています。結果と再現方法は[Orbit LoRA互換性・性能プローブ](docs/orbit-lora.md)を参照してください。

既定でDiTとQwen3-VLをTorchAO NVFP4（W4A4）で量子化します。BlackwellのネイティブNVFP4演算により、RTX PRO 4000 Blackwell 24GBでもCPUオフロードなしで高速に実行できます。Qwen3-VLのVision EncoderとDiTの非対応層だけはBF16のままです。量子化は読み込み時に行うため、別のモデルダウンロードは不要です。

## 起動

準備済みの環境では次のコマンドで、APIとフロントエンドを同時に起動できます。端末から起動すると推論プロファイルの選択メニューが表示されます。

```bash
./start.sh
```

選択できるプロファイル:

| プロファイル | 構成 | 用途 |
|---|---|---|
| `turbo` | Viggle r128 6-step LoRA + NVFP4、GPU常駐 | 最速。T2Iと参照1〜3枚のedit向け |
| `turbo-fp8` | Viggle r128 6-step + FP8、GPU常駐 | 非Blackwell向けTurbo候補 |
| `turbo-bf16` | Viggle r128 6-step + BF16、GPU常駐 | 非量子化重み＋Turboの比較用 |
| `turbo-minimum-vram` | Viggle r128 6-step + FP8、CPUオフロード | Turboを最小VRAMで実行 |
| `fast` | NVFP4、GPU常駐、参照解像度自動調整 | 24GB Blackwell推奨。高速かつ低VRAM |
| `compatible` | FP8、GPU常駐 | FP8互換パスの確認用 |
| `quality` | BF16元重み、GPU常駐 | 40GB以上のGPU向け |
| `minimum-vram` | FP8、CPUオフロード | 速度よりVRAM最小化を優先 |

非対話起動やサービス化ではプロファイルを直接指定できます。

```bash
./start.sh --profile turbo --device cuda:0
./start.sh --profile turbo-minimum-vram --device cuda:0
./start.sh --profile fast --device cuda:0
./start_backend.sh --profile minimum-vram --device cuda:1
```

高度な個別指定も可能です。

```bash
./start.sh --quantization nvfp4 --reference-mode adaptive --vae-tiling --trim-cache
./start.sh --help
```

- Web UI: <http://127.0.0.1:5173>
- APIドキュメント: <http://127.0.0.1:8000/docs>
- ヘルスチェック: <http://127.0.0.1:8000/api/health>

個別に起動する場合:

```bash
./start_backend.sh
./start_frontend.sh
```

初回生成時にモデルをGPUへ読み込みます。事前にWeb UIの「モデルをGPUへ読み込む」を押すこともできます。

## 初めからセットアップする場合

システムPythonへパッケージを追加せず、プロジェクト内の仮想環境だけを使います。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_model.py --accept-qwen-research-license
.venv/bin/python scripts/prepare_nvfp4.py
.venv/bin/python scripts/download_viggle.py --accept-qwen-research-license
```

Qwen-Image-2.1対応が最新Diffusersに入った直後のため、`requirements.txt` はViggle v0.2.1で検証されたDiffusers commitを指定しています。
`prepare_nvfp4.py` は量子化済みの再利用可能な起動キャッシュ（約10GB）を作ります。現在のマシンでは作成済みです。

## Web UI

- Text-to-Image
- 単一画像・複数画像を使った編集（最大10枚）
- 1:1、16:9、9:16、4:3、3:4、2Kプリセット
- ステップ数、CFG、シード指定
- 生成進捗表示
- PNGダウンロード
- 生成条件を同名JSONへ自動保存

通常はQwen推奨値の40ステップ、CFG 1.0を使います。ネガティブプロンプトはCFGを1より大きくした場合だけ有効になり、その場合は計算量が増えます。
`turbo`プロファイルではViggle r128の推奨条件に従い、6ステップ、CFG 1.0、専用sigma scheduleをサーバーが固定適用します。Web UIの該当欄も自動的に固定されます。

## API使用例

ジョブ登録:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F 'prompt=雨の東京、ネオンの反射、映画的な夜景' \
  -F 'width=1024' \
  -F 'height=1024' \
  -F 'steps=40' \
  -F 'guidance_scale=1' \
  -F 'seed=-1'
```

画像編集では、同じフィールド名で参照画像を追加します。

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F 'prompt=背景を雪山に変更する' \
  -F 'images=@input.png' \
  -F 'width=1024' -F 'height=1024' -F 'steps=40'
```

返されたIDで状態を取得します。

```bash
curl http://127.0.0.1:8000/api/jobs/JOB_ID
```

主なエンドポイント:

| Method | Path | 内容 |
|---|---|---|
| `GET` | `/api/health` | API、モデル、CUDA、GPU状態 |
| `POST` | `/api/model/load` | モデルの非同期ロード開始 |
| `POST` | `/api/model/unload` | VRAMからモデルを解放 |
| `POST` | `/api/jobs` | 生成ジョブ登録 |
| `GET` | `/api/jobs/{id}` | 進捗・結果取得 |
| `GET` | `/outputs/{file}` | 生成物取得 |

## 設定

環境変数で変更できます。

| 変数 | 既定値 | 用途 |
|---|---|---|
| `QWEN_DEVICE` | `cuda:0` | 使用GPU |
| `QWEN_QUANTIZATION` | `nvfp4` | `nvfp4`（Blackwell W4A4）、`fp8`（W8A8）、`bf16` |
| `QWEN_ACCELERATION` | `none` | `none` または `viggle-r128`（6-step Turbo） |
| `QWEN_CPU_OFFLOAD` | `0` | `1`でCPUオフロードを有効化 |
| `QWEN_START_PROFILE` | `custom` | 起動プロファイル名 |
| `QWEN_REFERENCE_MODE` | `adaptive` | `adaptive`または参照解像度を落とさない `full` |
| `QWEN_VAE_TILING` | `1` | VAEタイリング |
| `QWEN_TRIM_CUDA_CACHE` | `1` | プロンプト・Vision解析後の一時CUDAキャッシュ解放 |
| `QWEN_MODEL_DIR` | `models/Qwen-Image-2.1` | モデル保存先 |
| `QWEN_NVFP4_CACHE_DIR` | `models/Qwen-Image-2.1-nvfp4` | 事前量子化キャッシュ |
| `QWEN_VIGGLE_LORA_DIR` | `models/Qwen-Image-2.1-viggle-turbo` | Viggle r128 LoRAとscheduler |
| `QWEN_OUTPUT_DIR` | `outputs` | 生成物保存先 |
| `QWEN_API_HOST` | `127.0.0.1` | APIのバインド先 |
| `QWEN_API_PORT` | `8000` | APIポート |
| `QWEN_FRONTEND_HOST` | `127.0.0.1` | フロントのバインド先 |
| `QWEN_FRONTEND_PORT` | `5173` | フロントのポート |
| `QWEN_CORS_ORIGINS` | localhostの5173 | 許可するフロントURL（カンマ区切り） |

例として2枚目のGPUを使用する場合:

```bash
QWEN_DEVICE=cuda:1 ./start_backend.sh
```

BF16の元モデルで起動する場合:

```bash
QWEN_QUANTIZATION=bf16 ./start_backend.sh
```

NVFP4はBlackwell GPU専用で、CPUオフロードとは併用しません。他世代のGPUでは `fp8` または `bf16` を指定してください。

## GPU・速度・VRAM実測

以下は2026-09-25にRTX PRO 5000 Blackwell 48GBで測定した値です。環境はDriver 580.173.02、CUDA 13.0、PyTorch 2.14.0+cu130、TorchAO 0.18.0、Transformers 5.17.0、Diffusers 0.41.0.dev0です。CFG 1.0、batch 1、VAE tiling有効。VRAMは特記のない限り `nvidia-smi` で対象Pythonプロセスのピークをポーリングした値です。

### 全プロファイルと組み合わせ（1024×1024 T2I）

GPU0のRTX PRO 5000 Blackwellで同一条件を実測しました。通常プロファイルは40 steps、Viggleを組み合わせたプロファイルは公式6 stepsです。生成時間はモデルロードを含みません。

| プロファイル | 重み / 実行方式 | steps | ロード時間 | 生成時間 | ピークVRAM | `quality`比速度 | Viggle導入効果 |
|---|---|---:|---:|---:|---:|---:|---:|
| `quality` | BF16、GPU常駐 | 40 | 6.03秒 | 20.06秒 | 31.62 GiB | 1.00× | — |
| `fast` | NVFP4、GPU常駐 | 40 | 10.68秒 | 11.96秒 | 11.87 GiB | 1.68× | — |
| `compatible` | FP8、GPU常駐 | 40 | 37.67秒 | 18.48秒 | 18.54 GiB | 1.09× | — |
| `minimum-vram` | FP8、CPU offload | 40 | 27.09秒 | 27.20秒 | 9.80 GiB | 0.74× | — |
| `turbo-bf16` | BF16 + Viggle、GPU常駐 | 6 | 6.24秒 | 4.50秒 | 32.27 GiB | 4.46× | 4.46× vs `quality` |
| `turbo` | NVFP4 + Viggle、GPU常駐 | 6 | 12.07秒 | 3.57秒 | 12.59 GiB | 5.62× | 3.35× vs `fast` |
| `turbo-fp8` | FP8 + Viggle、GPU常駐 | 6 | 31.39秒 | 4.30秒 | 19.30 GiB | 4.67× | 4.30× vs `compatible` |
| `turbo-minimum-vram` | FP8 + Viggle、CPU offload | 6 | 28.29秒 | 13.93秒 | 9.81 GiB | 1.44× | 1.95× vs `minimum-vram` |

`quality`が量子化していない通常のBF16基準です。`quality`比速度は `20.06秒 ÷ 各生成時間` で、1より大きければ`quality`より高速、1より小さければ低速です。Viggle導入効果は、同じ重み形式・常駐方式の40-step構成との比較です。

Blackwellでは`turbo`が速度とVRAMの総合最適です。`turbo-fp8`と`turbo-bf16`は動作しますが、NVFP4 Turboより遅くVRAMも増えるため、主に互換性・比較用途です。VRAM最小は`minimum-vram`系で、CPU offloadの転送コストにより低step化の速度向上が小さくなります。NVFP4とCPU offloadの組み合わせは非対応です。

プロファイル別レポート: [fast](benchmarks/20260925-214049-nvidia-rtx-pro-5000-blackwell-fast.md) / [turbo](benchmarks/20260925-214218-nvidia-rtx-pro-5000-blackwell-turbo.md) / [compatible](benchmarks/20260925-215401-nvidia-rtx-pro-5000-blackwell-compatible.md) / [quality](benchmarks/20260925-215437-nvidia-rtx-pro-5000-blackwell-quality.md) / [minimum-vram](benchmarks/20260925-215540-nvidia-rtx-pro-5000-blackwell-minimum-vram.md) / [turbo-fp8](benchmarks/20260925-215627-nvidia-rtx-pro-5000-blackwell-turbo-fp8.md) / [turbo-bf16](benchmarks/20260925-215647-nvidia-rtx-pro-5000-blackwell-turbo-bf16.md) / [turbo-minimum-vram](benchmarks/20260925-215738-nvidia-rtx-pro-5000-blackwell-turbo-minimum-vram.md)

### 実用stepでの速度・VRAM比較

GPU0のRTX PRO 5000 Blackwellで、同じプロンプト、seed、入力画像、ベンチマーク処理を使って再測定しました。通常版は推奨40 steps、[Viggle Turbo v0.2.1 r128](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)は公式6-step scheduleです。2-step値は生成品質を評価できないため、この実用比較から除外しています。

| 処理 | 出力 | 参照 | 通常版 40-step 時間 / VRAM | Turbo 6-step 時間 / VRAM | 高速化 |
|---|---:|---:|---:|---:|---:|
| T2I | 1024² | 0 | 11.96秒 / 11.87 GiB | 3.57秒 / 12.59 GiB | 3.35× |
| T2I | 2048² | 0 | 76.22秒 / 14.11 GiB | 15.44秒 / 15.02 GiB | 4.94× |
| edit | 1024² | 1（1024px） | 14.98秒 / 15.24 GiB | 4.01秒 / 15.65 GiB | 3.74× |
| edit | 2048² | 1（1024px） | 86.72秒 / 16.89 GiB | 17.70秒 / 17.88 GiB | 4.90× |
| edit | 1024² | 3（896px） | 18.96秒 / 19.98 GiB | 5.54秒 / 19.92 GiB | 3.42× |
| edit | 2048² | 3（896px） | 98.62秒 / 21.94 GiB | 20.59秒 / 22.40 GiB | 4.79× |
| edit | 1024² | 10（512px） | 19.77秒 / 20.87 GiB | 対象外 | — |
| edit | 2048² | 10（512px） | 102.93秒 / 21.69 GiB | 対象外 | — |

LoRA本体は約649MiBです。TurboのVRAM差は条件により-0.06〜+0.99GiB、速度は約3.4〜4.9倍でした。参照3枚・2048²は22.40GiBなので24GB級GPUでは余裕が小さく、画面表示や他プロセスと共有する場合は1024²または参照1枚を推奨します。Viggleの対象範囲に合わせ、`turbo`では参照画像を最大3枚に制限しています。難しい編集では40-step通常版が画質面で有利な場合があります。

比較画像: [Turbo T2I](docs/assets/viggle-r128-t2i-1k-6step.png) / [Base T2I](docs/assets/base-t2i-1k-40step.png) / [Turbo edit](docs/assets/viggle-r128-edit1-1k-6step.png) / [Base edit](docs/assets/base-edit1-1k-40step.png)

再測定レポート: [通常版40-step](benchmarks/20260925-214049-nvidia-rtx-pro-5000-blackwell-fast.md) / [Viggle Turbo 6-step](benchmarks/20260925-214218-nvidia-rtx-pro-5000-blackwell-turbo.md)

`minimum-vram`も実用40-stepで再測定済みです。旧2-step値は実用速度を表さないため表から除外しました。CPUオフロードには十分なシステムRAM（64GB以上推奨）が必要です。

### VRAM容量別の目安

| GPU VRAM | 推奨プロファイル | 実測から判断できる範囲 |
|---:|---|---|
| 16GB | `turbo` / `fast` | T2Iは1024²と2048²で実測成功。Turbo 2048²は15.02GiBなので表示兼用では余裕小 |
| 24GB | `turbo` / `fast` | Turboは参照3枚・2048²まで実測成功（22.40GiB、共有GPUでは余裕小）。`compatible`は1024² T2Iで18.54GiB |
| 32GB | `fast` / `compatible` | NVFP4の全実測条件に余裕あり。BF16 1024²は31.62GiBなので実容量次第で不足 |
| 40GB以上 | `quality` / `turbo-bf16` | BF16 1024² T2Iを31.62〜32.27GiBで実測 |

この表は「その容量の実GPUすべて」を保証するものではありません。OSの画面表示、他プロセス、GPUごとのカーネルワークスペースの差を考慮し、実測ピークに1〜2 GiBの余裕を足してください。長辺ではなく総画素数がVRAMへ強く影響します。上表の限界はピクセル数が最も多い正方形で測定しています。

### 任意のGPUを自動測定

GPUごとの正確な速度と解像度限界は、同梱のベンチマークで測定できます。指定した解像度を小さい順に実行し、OOMになったcaseはそこで停止します。

```bash
.venv/bin/python scripts/benchmark_gpu.py \
  --profile fast \
  --device cuda:0 \
  --resolutions 1024,1536,2048,2560 \
  --cases t2i,edit1,edit3,edit10 \
  --steps 40
```

Viggle Turboの実用6-stepを測る場合:

```bash
.venv/bin/python scripts/benchmark_gpu.py --profile turbo --resolutions 1024,1536,2048 --cases t2i,edit1,edit3
```

`benchmarks/` に人が読めるMarkdownと機械処理用JSONを同時出力します。レポートにはGPU名、VRAM、Compute Capability、Driver/CUDA/PyTorch版、ロード時間、各caseの生成時間、`nvidia-smi` ピーク、PyTorch allocated/reserved、OOM結果が記録されます。

NVFP4キャッシュがある場合は起動時の量子化変換をスキップし、24GB GPUでのロードピークも抑えます。キャッシュがなければBF16モデルから自動変換します。複数画像の編集では、24GBに収めるため3枚目からVision Encoderへ入れる参照解像度だけを自動調整します。出力解像度は変わりません。

外部公開を行う場合は、認証のない状態で直接インターネットへ公開せず、認証付きリバースプロキシを前段に置いてください。

## ライセンスと帰属

本リポジトリで独自に作成したソースコードとドキュメントは[Apache License 2.0](LICENSE)で公開します。主な独自実装は、選択的NVFP4量子化、再利用可能な量子化キャッシュ、参照枚数に応じたVision入力制御、量子化・Viggle・CPU offloadを組み合わせる起動プロファイル、および速度・VRAMベンチマーク基盤です。これは学術的新規性や特許性を主張するものではありません。

Qwen Image 2.1、Viggle Turbo、Orbit LoRA、Diffusers、PyTorch、TorchAOなどの第三者素材・ソフトウェアは、それぞれのライセンスに従います。特にQwen、Viggle、Orbitのモデル重みはApache-2.0の対象外で、Qwen Research License上、研究・評価目的の非商用利用に限定されます。詳細は[NOTICE](NOTICE)、[Third-party notices](THIRD_PARTY_NOTICES.md)、[Security policy](SECURITY.md)を確認してください。

Built with Qwen. Qwenおよび各社の製品名は説明目的で使用しており、提携や推奨を意味しません。

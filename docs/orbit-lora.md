# Qwen Image 2.1 Orbit LoRA 互換性・性能プローブ

[`ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA`](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA) を、Diffusers Engine Q21のNVFP4量子化とViggle r128高速化へ重ねて実測した記録です。現時点ではAPI/Web UIに組み込んだ機能ではなく、再現用スクリプトによる実験機能です。

## 結論

- NVFP4量子化済みTransformerへOrbit LoRAを後からロードでき、同一seedのLoRAなし/あり比較で正面から指定した右90°へ変化した。
- RTX PRO 4000 Blackwell 24GBでは、768²・40 steps・CPU offloadなしでピーク12.69 GiB allocatedだった。
- OrbitとViggle r128は同時適用できる。6 stepsでは単発約5.0秒、連続生成の定常時約4.26秒/枚だった。
- 物体、アニメ調マスコット、ゲームキャラクター、人物のすべてで視点変更できた。ただし複雑な人物・キャラクターでは角度の取り違えと細部の再発明があり、完全な8方向キャラクターシートを保証しない。
- 単なるRGB→RGBAモード変換では背景は透明にならない。通常のRGB画像は、事前に背景除去して意味のあるalpha maskを作る必要がある。

## 測定環境

| 項目 | 値 |
|---|---|
| 測定日 | 2026-09-25 |
| GPU | NVIDIA RTX PRO 4000 Blackwell、24,467 MiB（PyTorch認識23.424 GiB） |
| 使用デバイス | GPU 1 / `cuda:1` |
| Driver / CUDA | 580.173.02 / PyTorch CUDA 13.0 |
| PyTorch | 2.14.0 |
| Diffusers | 0.41.0.dev0（プロジェクト固定commit） |
| Transformers / PEFT | 5.17.0 / 0.21.0 |
| 出力 | 768×768、batch 1、CFG 1.0 |
| 量子化 | Diffusers Engine Q21 NVFP4キャッシュ、CPU offloadなし |
| Orbit | rank 32 / alpha 32、step-2000、gate-up split版 |
| seed | 12345 |

Orbit weightはrevision `b5283f45e9291147ea428c5360445d6cee4d9d63`へ固定し、SHA-256 `bcea69afef59fe0d0126ec865590afb9052f9607e1e8c7898242fe816aac87f9`を確認しました。

ピーク値はPyTorch CUDA allocatorの `max_memory_allocated` / `max_memory_reserved` です。OS・画面表示・他プロセスを含む `nvidia-smi` の総使用量とは定義が異なります。

## 単一角度の速度とVRAM

同じ右90°、同じseedで測定しました。生成時間にモデルロード時間は含みません。

| 入力 / 構成 | steps | 生成時間 | peak allocated | peak reserved |
|---|---:|---:|---:|---:|
| robot、LoRAなし | 40 | 14.49秒 | 12.442 GiB | 13.092 GiB |
| robot、Orbit | 40 | 17.70秒 | 12.689 GiB | 12.891 GiB |
| robot、Orbit + Viggle | 6 | 5.04秒 | 13.327 GiB | 13.973 GiB |
| cartoon mascot、Orbit | 40 | 17.65秒 | 12.689 GiB | 12.891 GiB |
| cartoon mascot、Orbit + Viggle | 6 | 5.00秒 | 13.327 GiB | 13.973 GiB |
| game character、Orbit | 40 | 17.77秒 | 12.689 GiB | 12.891 GiB |
| game character、Orbit + Viggle | 6 | 5.01秒 | 13.328 GiB | 13.973 GiB |
| person、Orbit | 40 | 17.77秒 | 12.689 GiB | 12.891 GiB |
| person、Orbit + Viggle | 6 | 5.05秒 | 13.327 GiB | 13.973 GiB |

Orbitのみ4素材の平均は17.72秒、Orbit + Viggleは5.02秒で、初回生成を含む比較では3.53倍高速でした。Orbitの追加後、モデル常駐allocatedは10.591 GiBから10.747 GiBへ0.156 GiB増加しました。40-step生成ピークのLoRAなしとの差は0.247 GiBです。

## 7方向の連続生成

元画像を毎回入力し、相対方位 `left/right 45/90/135°` と `180°` を生成しました。生成結果を次の生成へ入力する連鎖方式ではありません。

| 素材 / 構成 | 方向数 | 合計 | 平均 | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| game character、Orbit 40-step | 7 | 120.67秒 | 17.24秒 | 12.696 GiB | 13.350 GiB |
| game character、Orbit + Viggle 6-step | 7 | 30.59秒 | 4.37秒 | 13.332 GiB | 14.332 GiB |
| person、Orbit 40-step | 7 | 121.48秒 | 17.35秒 | 12.696 GiB | 13.350 GiB |

Viggleスイープの初回は5.06秒、2枚目以降は平均4.26秒でした。ゲームキャラクターの7枚合計では40-step比3.95倍高速です。

## 画質の所見

| 素材 | 結果 |
|---|---|
| robot toy | 右90°で正面から側面へ明確に変化。LoRAなしは正面を維持したため、NVFP4上でOrbitが有効と確認できた |
| cartoon mascot | 右90°へ回転し、色と大きな輪郭は維持。隠れていた背面・突起の細部は再発明される |
| game character | 左右、正面寄り、背面の大分類は成立し、剣・盾・鎧を概ね維持。90°と135°が同じ側面へ寄る例があり、7方向が常に等間隔にはならない |
| person | 左45→90→135→180°は段階的に変化し、髪、青い上着、黒いパンツ、白い靴を概ね維持。右135°は右90°へ寄り、背面の上着に元画像にない中央線が生成された。顔も角度間で多少変化する |

これらは正解となる全周画像がない素材に対する定性的な評価です。プロダクションで8方向素材を作る場合は、複数seedを生成して角度とidentityを選別する運用が必要です。

Viggle 6-stepは40-stepとほぼ同じ大分類の角度を出しました。ゲームキャラクター7方向の40-step出力とのalpha-mask IoUは0.845〜0.973でした。透明画素に格納されたRGBは粒状に見えますが、alphaで白背景へ合成すると大半は見えず、40-stepと6-stepの透明マスクは近い品質でした。細部と角度の選別には40-step、反復プレビューにはViggle 6-stepという使い分けが安全です。

## RGBとRGBA

| 入力条件 | 出力alpha平均 | `alpha < 16` | 判定 |
|---|---:|---:|---|
| 元のRGBA robot | 68.91 | 72.90% | 透明背景を維持 |
| alphaを落としたRGB（透明領域の隠れRGBは保持） | 68.44 | 72.87% | プロンプトから透明を再生成 |
| 白背景へ合成した通常RGB | 254.95 | 0.00% | 背景はほぼ完全に不透明 |

「RGBA化」は画像モードを変えてalpha=255を足すだけでは不十分です。背景除去・セグメンテーションによってalpha maskを作り、被写体外を透明にしてからOrbitへ渡す必要があります。

## 再現方法

OrbitはQwen Research Licenseに従う非商用モデルです。ライセンスを確認したうえで取得します。

```bash
.venv/bin/python scripts/download_orbit_lora.py --accept-qwen-research-license
```

単一角度の通常40-step:

```bash
.venv/bin/python scripts/probe_orbit_lora.py \
  --input input.png \
  --prompt '<orbit> rotate the camera 90 degrees to the right, eye level. The image has alpha channel and the background is transparent.' \
  --device cuda:0 --quantization nvfp4 --steps 40 --label orbit-right-90
```

7方向のViggle 6-step比較:

```bash
.venv/bin/python scripts/probe_orbit_lora.py \
  --input input.png --sweep \
  --device cuda:0 --quantization nvfp4 \
  --acceleration viggle-r128 --steps 6 --label orbit-sweep
```

PNGと条件・時間・VRAMを含むJSONは `outputs/orbit-probe/` に保存され、Git管理から除外されます。

## ライセンスと制約

Orbit adapterはQwen Image 2.1の派生物で、Qwen Research Licenseによる非商用利用限定です。重みはこのリポジトリへ含めません。公式model cardも、学習対象はスタジオ撮影相当のGoogle Scanned Objectsであり、人物やアニメキャラクターは学習ドメイン外と明記しています。

本プローブが確認したのは現在の固定revisionと環境における実行互換性です。将来のDiffusers、PEFT、量子化実装、LoRA revisionとの互換性を保証するものではありません。

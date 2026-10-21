# Tesla T4 4 枚で Ornith 1.5 35B A3B の MTP ヘッド交換と 512k/1M コンテキストを検証

- **作成者**: MG8853
- **作成日**: 2026-10-18

## 概要

HPE ProLiant DL380 Gen10 + Tesla T4 × 4（15 GB / 枚）で、Ornith 1.5 35B A3B（Q4_K_M）の投機的デコード用 MTP ヘッドを比較しました。標準の純正ヘッドと、コミュニティ製の外部ドラフト [EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF](https://huggingface.co/EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF) の `mtpdraft-Q8_0.gguf` を `-md` で読み込ませた構成を、同一モデル・同一ビルド（llama.cpp build 11709）・同一条件で測っています。

結果は、**外部ドラフトはロードできて動作もするが、純正ヘッドより遅い**というものでした。254k 深さの decode は純正 43.5 t/s に対して外部 37.2 t/s（-14%）、32k では 89.5 対 75.6 t/s（-16%）です。純正ヘッドの採択率は実プロンプトで 0.486〜0.594、254k の合成テキストで 0.907 あり、「純正ヘッドは未学習で採択率 13%」という前提はこの環境では成立しませんでした。

MTP 自体の効果は深さで変わります。合成テキストのラダーを使った 254k では MTP あり 43.5 t/s / なし 24.8 t/s（+75%）ですが、深さ 0 の実プロンプトでは MTP なし 83.7〜84.1 t/s と MTP あり（79.8〜88.1 t/s）がほぼ同等でした。実プロンプトの 254k は未計測です。

長コンテキストは、ネイティブの 262,144 を超える 524,288（YaRN factor 2.0）と 1,048,576（factor 4.0）のロードを確認しました。512k は 514k 深さで prefill 296.2 t/s / decode 24.5 t/s（フルプロンプトの TTFT 約 28.9 分）、VRAM は 8.7〜9.4 GiB / 枚です。1M は VRAM 12.0〜13.2 GiB / 枚でロードと短い生成だけを確認し、速度と品質は未計測です。

## ハードウェア

| 項目 | 内容 |
|------|------|
| コンピュータ / マザーボード | HPE ProLiant DL380 Gen10 |
| GPU | Tesla T4 × 4（VRAM 15,360 MiB / 枚、電力上限 70 W / 枚） |
| GPU 接続 | PCIe 3.0 x16（`nvidia-smi` の current はアイドル時 Gen1 x16、max は Gen3 x16）。NVLink なし。`nvidia-smi topo -m` では GPU0-GPU1 が NODE、GPU2-GPU3 が NODE、この 2 組の間は SYS（NUMA ノードをまたぐ） |
| CPU | Intel Xeon Gold 6254 × 2（18 コア / 36 スレッド × 2。この環境でオンラインなのは 36 論理 CPU） |
| メモリ | 232 GiB |
| 電源 | 不明 |

## ソフトウェア環境

| 項目 | 内容 |
|------|------|
| OS | Ubuntu 26.04.1 LTS / Linux 7.0.14-14-pve |
| GPU ドライバ | 615.71.09（CUDA UMD 13.4） |
| llama.cpp | 0.7.0-dev（build 11709、commit `62318cd57`）、CUDA バックエンド。NCCL 2.30.4 がリンクされています（`ldd build/bin/libggml-cuda.so` に `libnccl.so.2`） |
| ベンチマーク | llama-split-bench 7af72d4 |

## ベンチマーク

### 条件

| 項目 | 内容 |
|------|------|
| ツール | llama-split-bench 7af72d4 |
| モデル | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) の `Ornith-1.5-35B-Q4_K_M.gguf`（21,713,463,040 バイト、総 36.0B / アクティブ 3B の MoE、アーキテクチャは `qwen35moe`、純正の nextn 1 層を内蔵） |
| 外部ドラフト | [EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF](https://huggingface.co/EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF) の `mtpdraft-Q8_0.gguf`（1,990,649,440 バイト、`qwen35moe` の nextn 1 層 + 出力層/埋め込み） |
| 測定モード | tensor（CUDA0〜CUDA3 の 4 枚、TP=4） |
| ctx / stages | 262144 / 0,32000,258000（262k の比較）、32768 / 0,32000（32k の比較）、524288 / 0,512000（512k、YaRN factor 2.0） |
| KV キャッシュ | q8_0 / q8_0 |
| 投機的デコード | `--spec-type draft-mtp --spec-draft-n-max 2`（純正ヘッド、外部ヘッドとも）。n-max 4 と MTP なしも比較 |
| その他 | `-fa on`、`-ngl all`、`-t 16`、`--parallel 1`、`--jinja`、`--cache-ram 8192 --cache-idle-slots --cache-reuse 256`（`--cache-reuse` はこのビルドでは未対応のため無効化される）、生成 1000 トークン / 段（512k のみ 200）、PORT 18081、`LAUNCH_PREFIX` なし |

外部ドラフトの実行では、`bench.local.conf` の既定 `SPEC_ARGS` と `EXTRA_ARGS` の両方に `--spec-type draft-mtp --spec-draft-n-max 2` が入り、サーバの起動引数で重複しました。サーバログに「`--spec-draft-n-max` specified multiple times ... only last value will be used」と警告が出ていますが、最後の値は 2 で、純正ヘッド側と同一条件です（実引数は `argv-tensor.txt` に記録）。

### 結果（262k / 32k、合成テキストのラダー）

単位は t/s です。depth 0 の prefill は新規プロンプト（pp2048）の値です。ラダー初段（11 トークン）の prefill は計測上のアーティファクトなので表に載せていません。

| 構成 | 32k prefill | 32k decode | 32k 採択率 | 254k prefill | 254k decode | 254k 採択率 |
|------|------:|------:|------:|------:|------:|------:|
| 純正 MTP（n-max 2） | 1082.2 | 89.5 | 0.906 | 468.1 | 43.5 | 0.907 |
| 外部 MTP（EryriLabs、n-max 2） | 1153.4 | 75.6 | 0.660 | 469.0 | 37.2 | 0.710 |
| MTP なし | 1273.8 | 65.5 | — | 543.9 | 24.8 | — |

### 結果（実プロンプト、深さ 0）

実プロンプト 3 本（design / review / qa、temp 0.7 / top_p 0.9、1200 トークン生成）の decode です。end-to-end は `生成トークン数 / wall_s`（TTFT を含むクライアント側の実効速度）です。

| 構成 | design | review | qa | 採択率 | end-to-end |
|------|------:|------:|------:|------:|------:|
| 純正 MTP（n-max 2） | 88.1 | 87.2 | 79.8 | 0.486〜0.594 | 77.9〜86.3 |
| 外部 MTP（EryriLabs、n-max 2） | 85.1 | 77.1 | 83.3 | 0.460〜0.571 | 75.5〜83.3 |
| 純正 MTP（n-max 4） | 74.6 | 67.3 | 76.9 | 0.278〜0.363 | 65.9〜75.5 |
| 外部 MTP（EryriLabs、n-max 4） | 77.2 | 74.1 | 71.6 | 0.330〜0.375 | 70.2〜75.9 |
| MTP なし | 83.7 | 83.9 | 83.9 | — | 82.2〜82.8 |

n-max 4 の実行は ctx 32768、n-max 2 と MTP なしの実行は ctx 262144 で起動しています（いずれも深さ 0 の実プロンプト）。

### 結果（n-max の比較、32k）

| 構成 | 合成 decode（32k） | 実プロンプト decode の平均 | 実プロンプト採択率 |
|------|------:|------:|------:|
| 純正 MTP、n-max 2 | 89.5 | 85.0 | 0.486〜0.594 |
| 純正 MTP、n-max 4 | 101.8 | 72.9 | 0.278〜0.363 |
| 外部 MTP、n-max 2 | 75.6 | 81.8 | 0.460〜0.571 |
| 外部 MTP、n-max 4 | 67.1 | 74.3 | 0.330〜0.375 |

合成テキストのラダーでは n-max 4 のほうが速く見えますが、実プロンプトでは n-max 2 のほうが速いという結果です。深いドラフトは合成テキストでは当たるが実プロンプトでは外れる、という傾向が両ヘッドで一致しました。

### 結果（512k、YaRN factor 2.0）

`--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` を追加し、524,288 コンテキストで測定しました。ラダーの目標 512,000 トークンに対して、実際には 514,374 トークン（96% フィル）で計測されています。

| depth | prefill | decode | 採択率 |
|------:|------:|------:|------:|
| 514k | 296.2 | 24.5 | 0.797 |

- 514k のフルプロンプトを読ませた場合の TTFT は約 28.9 分（514,374 / 296.2）です。
- 254k と比べると、prefill は 468.1 → 296.2 t/s（-37%）、decode は 43.5 → 24.5 t/s（-44%）でした。深さが倍になると attention のコストが増え、速度は約6割に落ちます。
- 深さ 0 の実プロンプトは design 91.2 / review 76.2 / qa 81.5 t/s（採択率 0.468〜0.681、end-to-end 74.5〜89.6）で、YaRN を有効にしても短い入力の生成速度は 254k のときと同程度でした。
- 起動時の VRAM は 8,869 / 9,637 / 8,869 / 9,637 MiB です。


### 結果（1M のロード確認）

ネイティブの 262,144 を超える 1,048,576 コンテキストを、YaRN factor 4.0（`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`）でロードしました。モデルは起動し、短いプロンプト（6 トークン）に対して 42.3 t/s で 16 トークンを生成しました（採択 8/14）。

| 項目 | 512k（factor 2.0） | 1M（factor 4.0） |
|------|------:|------:|
| 起動時の VRAM（CUDA0 / CUDA1 / CUDA2 / CUDA3） | 8,869 / 9,637 / 8,869 / 9,637 MiB | 12,279 / 13,559 / 12,279 / 13,559 MiB |
| ロード | 成功 | 成功 |
| 短いプロンプトの生成 | 42.4 t/s | 42.3 t/s |
| 長いプロンプトの実測 | 上の節 | 未実施 |

1M は最も大きい GPU で 13,559 MiB を使用し、T4 の 15,360 MiB に対して余裕は約 1.8 GiB です。512k / 1M とも、モデルカードが案内する YaRN の設定（factor 2.0 / 4.0）に合わせています。

### 所感

- **外部ドラフトは「動くが速くならない」。** EryriLabs の `mtpdraft-Q8_0.gguf` は標準 Ornith-1.5 の GGUF に `-md` でロードでき、生成も正常でした。ただし decode は 254k で 43.5 → 37.2 t/s（-14%）、32k で 89.5 → 75.6 t/s（-16%）と、純正ヘッドのほうが速い結果です。EryriLabs が案内する n-max 4 にすると、外部ヘッドは 32k で 67.1 t/s まで落ちました。
- **「純正ヘッドは未学習で採択率 13%」はこの環境では成立しません。** 純正ヘッドの採択率は実プロンプトで 0.486〜0.594、254k の合成テキストで 0.907 でした。Ornith-1.5 の上流 `config.json` には `mtp_num_hidden_layers: 1` があり、GGUF にも nextn 層が含まれています（公式モデルカードに MTP の記載はありません）。EryriLabs の 13% という数値は Ornith-1.0 世代（MTP 層が未学習だった）や別のランタイム設定を指している可能性があります（未検証）。
- **旧レポートとはビルドが違うため直接比較できません。** 2026-09-30 の [Ornith 1.5 の split-mode 比較](2026-09-30_105157_comparing_split_modes_of_ornith_1.5_35b_a3b_on_4x_tesla_t4.md) は llama.cpp build 11195 で、258k の tensor decode は 32.24 t/s、合成採択率 0.436〜0.528、実プロンプト 0.140〜0.326 でした。今回の build 11709 では同じモデルが 43.5 t/s、0.907、0.486〜0.594 です。MTP の実装が更新された可能性がありますが、ビルド間の差分は確認していません。**他モデル（Nemotron など）の旧ビルドの数値と、このレポートの数値を直接並べてはいけません。**
- **MTP の効果はコンテキスト深さに依存します（以下は合成テキストのラダー）。** 254k では MTP あり 43.5 t/s / なし 24.8 t/s（+75%）ですが、深さ 0 の実プロンプトでは MTP なし 83.7〜84.1 t/s に対し MTP あり 79.8〜88.1 t/s で、明確な差はありません。短いプロンプトの連続（チャット）では MTP の利得は小さく、長いコンテキストで生成するエージェント用途では大きい可能性がありますが、実プロンプトの長文は未計測です。
- **prefill は MTP なしのほうが速い。** 254k で 543.9 t/s（なし）/ 468.1 t/s（あり）でした。長文を読ませて少しだけ生成する用途では MTP を切る選択肢があります。一方、合成ラダーの 254k では decode が 75% 速いため、長いコンテキストで大量に生成する用途では有利な可能性があります（実プロンプトの長文では未検証）。
- **合成テキストの採択率だけで n-max を決めてはいけません。** 32k の合成ラダーでは n-max 4 が 101.8 t/s と n-max 2（89.5）より速く見えましたが、実プロンプトでは 72.9 対 85.0 t/s と逆転しました。実プロンプトの採択率も n-max 4 で 0.278〜0.363 まで落ちています。
- **end-to-end と decode の差は小さい。** 実プロンプトの `生成トークン数 / wall_s` は decode の 1〜2% 低いだけでした（例: 88.1 対 86.3）。ツールが算出する補正係数 0.770 は保守的ですが、実測の end-to-end と矛盾しません。今後は end-to-end を直接記録するほうが説明しやすい、という指摘は妥当です。
- **512k / 1M はロードできます。** 512k（YaRN 2.0）は VRAM 8.7〜9.4 GiB / 枚、1M（YaRN 4.0）は 12.0〜13.2 GiB / 枚で起動しました。1M は最も大きい GPU の余裕が約 1.8 GiB しかなく、長いプロンプトを入れたときの追加確保で足りなくなる可能性は未検証です。また YaRN は全リクエストに一律で適用されるため、通常長の入力での品質影響も未検証です。
- 計測中の GPU 最高温度は 84℃（純正 MTP、262k と 512k）でした。電力上限は既定の 70 W / 枚ですが、サンプラ（2 秒間隔）には 70 W を超える瞬間値（最大約 118 W）が記録されています。温度ガードは作動しませんでした。

## ChatGPT 提案のファクトチェック

2026-10-18 時点で、提案に含まれる Hugging Face リポジトリの実在・ファイルサイズ・主張を確認しました。「判定」は、この環境で実測またはファイル一覧で確認できた範囲です。

| 主張 | 確認結果 | 判定 |
|------|----------|------|
| EryriLabs BigBang-MTP-GGUF が存在し、`mtpdraft-Q8_0.gguf`（約 2 GB）を標準 Ornith に `-md` で追加できる | リポジトリは実在。`mtpdraft-Q8_0.gguf` は 1.99 GB。README に「stock Ornith-1.5 GGUFs でも動く」と記載。実際にロード・生成できた | 正しい |
| 標準 Ornith の MTP 採択率は約 13% | 本環境の Ornith-1.5 純正ヘッドは実プロンプト 0.486〜0.594、254k 合成 0.907。13% ではない | 再現せず（世代・ビルド依存） |
| 改造モデルの採択率は 55〜75% | 外部ヘッドは実プロンプト 0.460〜0.571、32k 合成 0.660、254k 合成 0.710 | 再現せず |
| RTX 3090 で 131 → 169 tok/s（+29%） | T4×4 では 254k で 43.5 → 37.2 t/s、32k で 89.5 → 75.6 t/s | 再現せず（逆効果） |
| shisa-ai Ornith-1.5-MTP-ONLY は vLLM 向けで GGUF 版がない | リポジトリは実在するが、ファイルは BF16 の `model-mtp.safetensors`（1.69 GB）のみ。GGUF はなし | 正しい（llama.cpp では未検証） |
| offmonreal MaxQuality-MTP の Q4_K_M は 20.22 GiB、`--spec-draft-n-max 2` が最速 | リポジトリは実在。`Ornith-1.5-35B_Q4_K_M_imatrix_MTP.gguf` は 21.71 GB（20.22 GiB）。n-max 2 > 4 は本計測でも両ヘッドで一致 | 概ね正しい（112k コードでの検証は未実施） |
| Nemotron 3.5 Lightning CRACK は Q4_K_M 25.4 GB、MMLU 78.3 → 74.2% | リポジトリは実在。Q4_K_M は 25.43 GB。MMLU の数値は README の記載で、当方は未検証 | サイズは正しい |
| Qwen3.8 27B Uncensored は Q5_K_M / Q6_K が 19.5〜22.4 GB、MTP を保持 | 実在。Q5_K_M 19.54 GB、Q6_K 22.43 GB。MTP 内蔵版と、noMTP 本体 + 別ドラフト（1.68 / 3.16 GB）の両方がある | 正しい |
| GLM-4.7-Flash Heretic の Q6_K は 22.92 GB | 実在するが、Q6_K は 24.61 GB | サイズが誤り |
| Qwen3.8 Flash Next Gyro-M TQ2_0 は 64k で約 37 GiB、専用 fork が必要 | ファイルは 91.97 GB（TQ2_0）。専用ランタイムが必要という点は正しいが、37 GiB はファイルサイズではなく実行時 VRAM の話とみられる。T4×4 の 60 GiB には収まらない | 不正確 |
| GLM-5.3-Flash Huihui UD-IQ3_XXS は RAM オフロード向け | 実在。UD-IQ3_XXS は約 120 GB（既存レポートの 112.1 GiB と整合） | 正しい |
| Nemotron 3.5 Super VL 120B A12B の AD-IQ3_XXS は 65.3 GB、RAM 併用 | 実在。43 分割で約 65.3 GB。60 GiB の VRAM には収まらない | 正しい |
| DeepSeek V4.1 Flash の最小 Q2_K は 271 GB、T4×4 では優先度が低い | 実在。Q2_K は約 269 GB | 正しい |


## 256k / 512k / 1M のコーディング用途への結論

- **256k は Ornith 1.5 35B A3B Q4_K_M + 純正 MTP（n-max 2）+ tensor が第一候補**です。254k で prefill 468.1 t/s / decode 43.5 t/s、実プロンプトの end-to-end は 77.9〜86.3 t/s。262k のフルプロンプトを一度読ませる場合の TTFT は約 9.3 分（262,144 / 468.1）で、プロンプトキャッシュが効く2ターン目以降は短くなります。
- **512k は YaRN factor 2.0 で動きますが、速度は 256k の約6割**です。514k で prefill 296.2 t/s / decode 24.5 t/s、TTFT は約 28.9 分でした。256k が対話的なら、512k はバッチ／長時間ジョブ向けの条件です。ネイティブ長ではないため、品質への影響は未評価です。
- **1M はロードできますが、速度は未計測**です。VRAM は 13,559 MiB / 枚で余裕が約 1.8 GiB です。prefill 速度は深さとともに低下するため（254k 468.1 → 514k 296.2 t/s）、1M の TTFT は 512k の 2 倍以上になる見込みです。1M を常用するなら、まず速度と品質を測ってから判断してください。
- **MTP ヘッドの交換は不要**です。外部ドラフトはロードできるものの、254k で -14%、32k で -16% 遅くなります。純正ヘッド + n-max 2 が最も速い構成でした。
- **短いプロンプト中心なら MTP を切る選択肢もあります**（実プロンプト 83.9 t/s 対 79.8〜88.1 t/s）。合成ラダーの 254k では +75% の差がありますが、実プロンプトの長文では未計測です。
- このレポートは**速度とロード可否の検証であり、生成コードの品質（正答率）は評価していません**。ChatGPT 提案の比較対象（Nemotron CRACK、Qwen3.8 27B Uncensored、GLM-4.7 Heretic など）も今回は未計測です。これらを比較する場合は、同一ビルド・同一量子化・同一プロンプトで測り直す必要があります。

## 添付

各実行の図（日英）、`run-info.json`、段ごとの結果 JSON、サーバ起動引数を格納しています。モデルファイルは含みません。

- 純正 MTP、262k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-262k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-262k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-262k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-262k/results-real.json)
- 外部 MTP（EryriLabs）、262k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-262k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-262k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-262k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-262k/results-real.json) / [argv-tensor.txt](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-262k/argv-tensor.txt)
- 純正 MTP、n-max 4、32k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-nmax4-32k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-nmax4-32k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-nmax4-32k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-nmax4-32k/results-real.json)
- 外部 MTP、n-max 4、32k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-nmax4-32k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-nmax4-32k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-nmax4-32k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/eryri-mtp-nmax4-32k/results-real.json)
- MTP なし、262k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-262k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-262k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-262k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-262k/results-real.json)
- MTP なし、32k: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-32k/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-32k/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-32k/results-tensor.json) / [results-real.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/nomtp-32k/results-real.json)
- 純正 MTP、512k（YaRN 2.0）: [split-bench-ja.png](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-512k-yarn2/split-bench-ja.png) / [run-info.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-512k-yarn2/run-info.json) / [results-tensor.json](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-512k-yarn2/results-tensor.json) / [argv-tensor.txt](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/stock-mtp-512k-yarn2/argv-tensor.txt)
- 1M のロード確認: [1m-load-check.txt](attachment/2026-10-18_091752_validating_mtp_head_swap_and_512k_1m_context_of_ornith_1.5_35b_a3b_on_4x_tesla_t4/1m-load-check.txt)

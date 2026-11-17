# GPUEnsemble

**複数のGPUを束ねて、1枚では載らない大きなLLMを動かす推論エンジン。**

*Pools the VRAM of several GPUs, in one PC or across PCs, so that models too large for any single card run locally.* **[English README](README.en.md)**

> **状態: pre-alpha。** 推論の実行は llama.cpp の層分割（パイプライン）に任せ、GPUEnsemble は配置プランナとして、この PC に合うモデルを選んでダウンロードし、層の割り振りを決めて llama.cpp を動かし、動いている間の速さと各カードの様子を見せます。別の PC の GPU を足すこともできます（試験段階）。独自のパイプライン実行器は MVP の後の課題です。下の「MVP の完了条件」を満たしたので、これを最初の公開版 v0.1.0 とします。

## 試す（Windows、NVIDIA GPU）

初めて使う方は、手順をまとめた [使い方ガイド](docs/GUIDE.md) から読んでください。

**zip で:** zip は 2 種類あり、ウィンドウのアプリ（`GPUEnsemble.exe`）付きの `GPUEnsemble-<版>-windows-x64-app.zip` と、アプリなしでコマンドだけの `GPUEnsemble-<版>-windows-x64-cli.zip` です。どちらかを、空き容量の多いドライブの英数字だけのフォルダ（例: `D:\GPUEnsemble`）に展開し、`GPUEnsemble.exe`（アプリ付きのとき）か `Start GPUEnsemble.cmd` をダブルクリックします。Python も含まれているので、ほかに入れるものはありません。アプリは日本語と英語で表示でき、左のメニューの下の「English」/「日本語」で切り替えられます。GPU のある別の PC があれば、そちらにも展開して `Start GPUEnsemble worker.cmd` をダブルクリックすると、その GPU も使います（試験段階。初回は worker のウィンドウで `y` を押して許可します。別の場所の PC も、両方が同じアカウントの Tailscale に入っていれば同じ手順でつながり、`y` も要りません。[2 拠点での計測](benchmarks/phase2b-two-sites-tailscale.md)）。モデルと llama.cpp はそのフォルダの `data` に入ります。zip は GitHub Actions の [windows-zip](.github/workflows/windows-zip.yml) が作ります（リリース時と、必要なときに手動で実行。手元では `python packaging/windows/build_zip.py`）。

**Python がある場合:** 3.10 以上なら、追加のライブラリは要りません。

```powershell
pip install .\planner
gpuensemble up
```

`up` は次を順に行います。

1. GPU（VRAM と PCIe リンク）、RAM、CPU スレッド数を調べる
2. カタログのモデルをこの PC で配置して速度を見積もり、読む速さ（5 トークン/秒）以上で動くうち最も賢いものを選ぶ
3. そのモデルを Hugging Face からダウンロードする（中断しても続きから、SHA-256 を照合）
4. この PC に合う llama.cpp（CUDA 12.4 / Vulkan / CPU）をダウンロードし、GPU が見えるか確かめる
5. 実際のファイルを読んで層と MoE エキスパートを GPU に割り振り、llama-server を起動する

起動したら `http://127.0.0.1:8080/` をブラウザで開くとチャットでき、`http://127.0.0.1:8080/v1` は OpenAI 互換 API です。`gpuensemble recommend` で候補の一覧と予想速度、`gpuensemble up --model qwen3-8b` でモデルの指定ができます。動いている間は `gpuensemble monitor --every 2` で、実際の生成速度、カードごとのメモリ・温度・クロック・負荷、いちばん時間のかかる部分が見られます（アプリではホームに出ます）。詳しくは [planner/README.md](planner/README.md)。

開発機（RTX 2080 Ti + RTX 2080 SUPER、64 GB RAM）では gpt-oss-120b が選ばれ、毎秒約 6.5 トークン（生成を続けたときの安定した値）で動きます。別の拠点にある RTX 4090 を Tailscale 経由で足すと、同じモデルが毎秒 10.7 トークンになりました。

## コンセプト

- モデルの層をカードごとに連続した範囲で分け（パイプライン分割）、重みも KV キャッシュも重複させずに VRAM を足し合わせる
- カード間で受け渡すのは活性値だけ（1トークンあたり数十KB）なので、NVLink も P2P も不要。PCIe x4 や x1 のスロットでも成り立つ
- 8GB 級の小さいカードも捨てずに使う。速いカード・遅いカード・VRAM の大小が混ざっていても、配置プランナが実測値から最適な分割を自動で決める
- 載りきらない MoE エキスパートは RAM に置き、必要な分を GPU へ流し込む
- 計算カーネルは [ggml](https://github.com/ggml-org/ggml)（llama.cpp の土台）を使い、モデルは既存の GGUF、API は OpenAI 互換

詳しくは [docs/DESIGN.md](docs/DESIGN.md) を参照してください。

## 対象ハードウェア（MVP）

| 項目 | 内容 |
| --- | --- |
| GPU | NVIDIA Turing 世代（RTX 20 系）を複数枚。開発機は RTX 2080 Ti ×1 + RTX 2080 SUPER ×2 |
| バックエンド | CUDA 12.x（将来 Pascal などを足せるよう CUDA 13 は使わない） |
| OS | Windows 10 / 11 |
| CPU | AVX2 なしの CPU（Xeon E5 v1 など）も対象 |

## ロードマップ

| フェーズ | 内容 |
| --- | --- |
| 0 | 既存の llama.cpp で基準値を計測（[docs/PHASE0_BENCHMARK.md](docs/PHASE0_BENCHMARK.md)） |
| 1 | MVP: 単一PC・複数GPU、自動配置プランナ、実行は llama.cpp の層分割（llama-server の OpenAI 互換 API）、モニタ（`monitor` とアプリのホーム） |
| 1c | 独自のパイプライン実行器（`engine/`）。llama.cpp の層分割より速くできる見込みが立ったら着手する |
| 1b | MoE エキスパートの RAM 退避と GPU への流し込み |
| 2 | Vulkan による AMD 旧型との混在、KV キャッシュ量子化、Pascal など旧世代の追加 |
| 3 | 複数PC（段を TCP でつなぐ）、マイクロバッチ。最初の形として、llama.cpp の RPC で別の PC の GPU に層を置く（`worker` と `--remote`、LAN で自動的に見つけて貸す側で一度許可する `worker --lan` / `--remote auto`、zip の `Start GPUEnsemble worker.cmd`。別の PC 1 台に収まるモデルは、その PC で丸ごと動かして API を中継する `worker --serve`。試験段階。[同じ PC の中での計測](benchmarks/phase2a-rpc-loopback.md)、[2 拠点での計測](benchmarks/phase2b-two-sites-tailscale.md)） |

**MVP の完了条件:** 1枚に入らない 70B 級 Q4 モデルが分割指定の手書きなしで動き、生成速度がフェーズ0で測った llama.cpp の層分割と同等（差 10% 以内）であること。2026-11-17 に開発機で確認しました（Llama 3.3 70B Q4_K_M、プランナの配置 1.22 tok/s、手で詰めた層分割 1.23 tok/s。[計測](benchmarks/mvp-70b-e5.md)）。

## リポジトリ構成

| ディレクトリ | 役割 |
| --- | --- |
| `engine/` | 独自のパイプライン実行器の置き場（MVP の後。今は空） |
| `planner/` | Python。デバイス検出と層の配置プランナ |
| `server/` | 独自の API サーバの置き場（今は空。API は llama-server、モニタは `planner` の `monitor`） |
| `benchmarks/` | 計測結果のまとめ |
| `docs/` | 設計書と手順書 |
| `third_party/` | 版を固定した ggml などの外部コード |

## 安全について

別の PC に GPU を貸す機能は llama.cpp の RPC サーバを使います。RPC サーバにはパスワードも暗号化もないので、GPUEnsemble は許可した PC のアドレスだけを通す門番を前に置きます。家庭の LAN、2 台を直結したケーブル、Tailscale のような信頼できるネットワークだけで使ってください。詳しくは [SECURITY.md](SECURITY.md) と [planner/README.md](planner/README.md)。

## 謝辞

[Strata](https://github.com/Niko1221/Strata) の層分割と MoE エキスパート退避の考え方を参考にしています。

## ライセンス

[MIT](LICENSE)。zip に含まれる Python、実行時にダウンロードする llama.cpp とモデルは、それぞれのライセンスに従います（[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

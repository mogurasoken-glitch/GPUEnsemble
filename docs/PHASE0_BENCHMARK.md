# フェーズ0: llama.cpp での基準計測

自作エンジンを作る前に、既存の llama.cpp が開発機（Xeon E5-2680 + RTX 2080 Ti ×1 + RTX 2080 SUPER ×2、Windows）でどれだけ出るかを測り、「超えるべき基準値」を残します。結果は `benchmarks/phase0.md` にまとめます。

## 1. 組む前の確認

- [ ] 電源が 1000W 前後あり、PCIe 補助電源のコネクタが足りている
- [ ] 2スロット幅のカードが 3枚挿せる（詰まっていればライザーケーブル）
- [ ] BIOS で Above 4G Decoding を有効にする（項目が無いボードもある）
- [ ] NVIDIA の最新ドライバを入れ、`nvidia-smi` で 3枚とも見える
- [ ] 各カードの PCIe リンクを記録（計測中に実行すること）:
  `nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv`

## 2. llama.cpp とモデル

1. [llama.cpp の Releases](https://github.com/ggml-org/llama.cpp/releases) から Windows 向け CUDA 12 版（`llama-b…-bin-win-cuda-12.x-x64.zip`）と `cudart-llama-bin-win-cuda-12.x-x64.zip` を取得し、同じフォルダに展開する。
2. `llama-bench.exe --list-devices` で CUDA0〜CUDA2 の番号とカード名の対応を確認する。
3. 起動しない、または「illegal instruction」で落ちる場合は、AVX2 が無いことが原因の可能性が高い。エラー文を残しておく（AVX2 なしでのビルド手順を別途用意する）。
4. モデル（空き容量は合計 90GB 程度）:

| 役割 | モデル | 大きさ（概算） |
| --- | --- | --- |
| 密モデルの基準 | Qwen3-32B Q4_K_M（GGUF） | 約 20GB |
| 大型 MoE の基準 | gpt-oss-120b（GGUF, MXFP4） | 約 61GB |

## 3. 計測

`pp512` はプロンプト処理、`tg128` は生成の速度（トークン/秒）。`-ts` の順番は CUDA0, CUDA1, CUDA2 で、CUDA0 が 2080 Ti でなければ入れ替える。

**A. 密モデル**

```powershell
.\llama-bench.exe -m <32Bのファイル> -ngl 99 -sm layer -ts 11/8/8 -p 512 -n 128   # A1 層分割
.\llama-bench.exe -m <32Bのファイル> -ngl 99 -sm row   -ts 11/8/8 -p 512 -n 128   # A2 行分割
.\llama-bench.exe -m <32Bのファイル> -ngl 99 -sm layer -ts 13/7/7 -p 512 -n 128   # A3
.\llama-bench.exe -m <32Bのファイル> -ngl 99 -sm layer -ts 9/9/9  -p 512 -n 128   # A4
```

**B. 大型 MoE**

```powershell
# B1: エキスパートを CPU 側に置く層の数を振り、動いた最小値を探す
.\llama-bench.exe -m <120bの最初のファイル> -ngl 99 -ts 11/8/8 -ncmoe 36,30,24,18 -p 512 -n 128
# B2/B3: その値でスレッド数を比べる
.\llama-bench.exe -m <120bの最初のファイル> -ngl 99 -ts 11/8/8 -ncmoe <B1の値> -t 8,16 -p 512 -n 128
```

実行中は別の PowerShell で `nvidia-smi -l 1` を動かし、VRAM と温度を見ておく。

## 4. 結果

| 計測 | 設定 | pp512 | tg128 | メモ |
| --- | --- | --- | --- | --- |
| A1 | 32B、層分割、-ts 11/8/8 | | | |
| A2 | 32B、行分割、-ts 11/8/8 | | | |
| A3 | 32B、層分割、-ts 13/7/7 | | | |
| A4 | 32B、層分割、-ts 9/9/9 | | | |
| B1 | 120b、動いた最小の -ncmoe = | | | |
| B2 | 120b、同上、-t 8 | | | |
| B3 | 120b、同上、-t 16 | | | |

# server/

独自の API サーバの置き場です（今は空）。

- **OpenAI 互換 API:** `gpuensemble up` が起動する llama-server が `http://127.0.0.1:8080/v1`（`/v1/chat/completions` など）を提供します。
- **モニタ:** `gpuensemble monitor`（[planner/src/gpuensemble_planner/monitor.py](../planner/src/gpuensemble_planner/monitor.py)）が、
  実際の生成速度、カードごとのメモリ・温度・クロック・負荷、段ごとの時間（プランの見積もりを実測の速さに合わせたもの）と
  いちばん時間のかかる部分を出します。Windows のアプリはこれをホームに表示します。

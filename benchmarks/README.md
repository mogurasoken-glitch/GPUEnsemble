# benchmarks/

計測結果のまとめを置きます。生ログは `benchmarks/raw/`（git 管理外）に保存してください。

- [`phase0-e5-turing.md`](phase0-e5-turing.md): E5 機（2080 Ti + 2080 SUPER）での llama.cpp 基準値（2026-11-14）（手順は [docs/PHASE0_BENCHMARK.md](../docs/PHASE0_BENCHMARK.md)）
- [`install-and-go-e5.md`](install-and-go-e5.md): `pull` / `setup` / `up` を空の状態から開発機で実行した結果（何も指定せずに gpt-oss-120b が選ばれ、起動してチャットできた）
- [`phase1a-e5-llamacpp.md`](phase1a-e5-llamacpp.md): プランナの配置を llama.cpp に渡した結果（手調整より生成 36% 速い。長い文脈では q8_0 の KV キャッシュで 1.3〜1.6 倍）。末尾に `recommend` の確認（開発機では gpt-oss-120b を推奨）
- [`phase2a-rpc-loopback.md`](phase2a-rpc-loopback.md): llama.cpp の RPC を同じ PC 内（127.0.0.1）で使ったときのコスト
- [`phase2b-two-sites-tailscale.md`](phase2b-two-sites-tailscale.md): 別の拠点の RTX 4090 を Tailscale 越しに借りた結果（gpt-oss-120b の生成が 6.45 → 10.7 tok/s、往復 7 ms）。翌日、ダブルクリックと同じ起動だけでつながること、貸す側が自分の分の重みを 133 秒でダウンロードすること（初回の起動が 804 秒 → 502 秒）を確認
- [`phase2c-three-cards.md`](phase2c-three-cards.md): 同じ 3 枚でほかのモデル（Qwen3-32B は 4090 だけに載って 4.8 → 25.9 tok/s、Llama 3.3 70B は 1.2 → 4.2 tok/s）と、別の PC に置いた段の通信の内訳（往復は 1 トークンに 1 回、出力の頭を別の PC に置くと約 5 ms、待つ間にメインの PC のカードがクロックを下げる、会話のキャッシュの読み戻し）
- [`clock-hold.md`](clock-hold.md): 待つ間のクロック低下の原因（NVIDIA の「パフォーマンス最大化を優先」は CUDA だけでは効かず、Vulkan などで開いているカードにだけ効く）と、`up` がカードを Vulkan で開いておく変更の確認
- [`cpu-rates-held.md`](cpu-rates-held.md): 予想速度が実測より約 20% 低かった原因（CPU の速さを、カードが待つ間にクロックを下げていた頃の計測で決めていた）と、クロックを保った状態で測り直した値（CPU の層 19.6 → 25 GB/s、エキスパート 17 → 23.5 GB/s。gpt-oss-120b の 3 枚は予想 12.2 → 約 15.1、実測 15.3〜15.5）
- [`mvp-70b-e5.md`](mvp-70b-e5.md): MVP の完了条件の確認（2026-11-17）。開発機だけで Llama 3.3 70B Q4_K_M を動かし、プランナの配置が 1.22 tok/s、手で詰めた llama.cpp の層分割が 1.23 tok/s（比 0.99）。モニタを実機で動かして見つかった点も
- [`no-gpu-main-e5.md`](no-gpu-main-e5.md): GPU のない主 PC の確認（2026-11-17）。E5 を GPU なしと見なし、llama.cpp の CPU 版から 4090 だけで Qwen3.8-27B を動かして 11〜23 tok/s（予想 25。4090 の PC で丸ごと動かすと 46）、質問ごとに 3〜5 秒の固定の待ち。CPU だけでは 1.7 tok/s。Vulkan 版でのカードの見分けも

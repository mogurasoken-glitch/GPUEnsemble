# planner/

GPUEnsemble の配置プランナ（試作）。モデルのどの層をどの GPU に置くかを決め、生成速度の上限を見積もります。Python 3.10 以上、外部ライブラリ不要です。

## 使い方

```powershell
cd planner
pip install -e .

# 実際の GGUF を読んで配置を計算（GPU と PCIe リンクは nvidia-smi、RAM とスレッド数は OS から自動検出）
gpuensemble-plan plan C:\models\Qwen3-32B-Q4_K_M.gguf --ctx 8192

# GPU と RAM を JSON で指定する（開発機の例）
gpuensemble-plan plan C:\models\Qwen3-32B-Q4_K_M.gguf --rig examples\rig-e5-turing.json

# 分割されたモデルは、どのパートを指定しても全パートを読む
gpuensemble-plan plan C:\models\gpt-oss-120b-00001-of-00003.gguf --rig examples\rig-e5-turing.json

# モデルの層ごとのサイズを JSON に書き出す（重いファイルを渡さずに共有できる）
gpuensemble-plan describe C:\models\Qwen3-32B-Q4_K_M.gguf > qwen3-32b.json
```

モデルをまだ持っていなくても、カタログの名前（`models` が一覧にする id）か `examples/` の見積もり用プロファイルで試せます。カタログの名前を指定すると、`pull` 済みならそのファイルを読み、まだならカタログにあるモデルの説明から配置を計算します（`plan`、`args`、`describe`。`run` はファイルが要るので、`pull` 済みのときだけ）。

```powershell
python -m gpuensemble_planner plan gpt-oss-120b --rig examples\rig-e5-turing.json
python -m gpuensemble_planner plan examples\gpt-oss-120b.json --rig examples\rig-e5-turing.json
```

`plan` が出す配置と速さは、`up` が実際に動かすものと同じです。MoE モデルのエキスパートは、llama.cpp にできるとおり層ごとにまとめて GPU か RAM に置きます。層の中でエキスパートを GPU と RAM に分ける（PCIe で流し込むほうが速ければそうする）独自の実行器向けの配置は `plan --split-experts` で見られますが、llama.cpp では動かせません。

## llama.cpp で配置どおりに動かす（フェーズ1a）

エンジン本体ができるまでは、プランナの配置を llama.cpp の引数（`-ngl`、`-dev`、`-ts`、`-ot`）に変換して動かします。`-ts` や `-ncmoe` を手で探す必要はありません。

```powershell
# 配置と、それを実行する llama-server のコマンドを表示
gpuensemble-plan args C:\llm\models\gpt-oss-120b-MXFP4.gguf --rig examples\rig-e5-turing.json --windows --exe C:\llm\bin\cuda\llama-server.exe

# llama-bench 用の書式（-ts と -dev を / 区切り）
gpuensemble-plan args C:\llm\models\Qwen3-32B-Q4_K_M.gguf --rig examples\rig-e5-turing.json --tool bench --exe llama-bench --windows

# 計画してそのまま起動（-- の後ろは llama.cpp にそのまま渡す）。llama-server なら読み込み完了で ready を表示。
# RAM の予算より大きいモデルは mmap せず RAM に読み込む（初回から速いが読み込みは数分）
# 起動した llama.cpp は、Ctrl+C でも、この Python を強制終了しても一緒に終わる（Windows はジョブオブジェクト、Linux は PR_SET_PDEATHSIG）
gpuensemble-plan run C:\llm\models\gpt-oss-120b-MXFP4.gguf --exe C:\llm\bin\cuda\llama-server.exe -- --port 8080
```

対応関係: CPU に残す先頭の層は `-ngl`、GPU の並びと担当層数は `-dev` と `-ts`、RAM に残す MoE エキスパートは層単位で `-ot "blk\.(…)\.ffn_.*_exps\.=CPU"`。カード番号がずれないよう `CUDA_DEVICE_ORDER=PCI_BUS_ID` を付けます。llama.cpp はエキスパートを PCIe で流せないので、このコマンドでは CPU 計算を前提に計画します。

KV キャッシュの型は `--kv auto`（既定）で選びます。f16 で CPU に残る層があるときは q8_0 も試し、見積もりで 2% より速ければ q8_0 にして `-ctk q8_0 -ctv q8_0` を付けます。開発機の Qwen3-32B（ctx 8192）では CPU に残る層が 13 → 10 に減り、文脈が埋まった状態の生成が 1.3〜1.6 倍になりました。回答は f16 と一致しています。`--kv f16` などで固定もできます。

## この PC で動かせるモデルを選ぶ（recommend）

PC の構成（GPU・RAM・CPU スレッド・PCIe）を自動で調べ、カタログ（`src/gpuensemble_planner/catalog.json`）の各モデルについて、入るかどうか・どう配置するか・予想速度を一覧にします。文脈が半分埋まった状態で 5 トークン/秒（読む速さ程度）以上出るモデルのうち、品質順位がいちばん高いものを推奨します。

**モデルの一覧を新しくする（`catalog --update`）:** 一覧は三つから作ります。(1) パッケージに入っている、手で確かめた一覧（`catalog.json`）、(2) その一覧の最新版をリポジトリから取ってきたもの（`updated` の日付が新しいほうを使います）、(3) Hugging Face で最近人気の GGUF（ggml-org、unsloth、bartowski、lmstudio-community の、作られて 180 日以内、2000 ダウンロード以上、`Q4_K_M` があるもの。最大 12 件）。(3) は Hugging Face がテキストのモデルとしているもので、ヘッダーに KV キャッシュか状態を持つ層があり、確かめた一覧にある種類（`catalogs.KNOWN_ARCHITECTURES`）のものだけです。手で確かめていないので `new: true` が付き、確かめたものの後ろに並びます。一覧のモデルに `profile`（層ごとの大きさ）が書かれていなければ、Hugging Face のファイルの先頭（ヘッダー、数 MB）だけを読んで作り、`<home>/profiles/` に残します。配置の計算はネットワークを使わないので、まだ読めていないモデルは一覧に出ません。Windows アプリは起動のたびにこれを裏で実行します。

**一覧にないモデルを足す（`catalog --add`）:** Hugging Face にある GGUF なら、どのリポジトリのモデルでもその PC の一覧に足せます（ファインチューン版など、どの一覧にも載らないもの）。リポジトリ名（`owner/name`）か、そのページやファイルのリンクを渡します。ファイルのリンクならその量子化、リポジトリだけなら `Q4_K_M`（`--quant` で変えられます）を使います。ファイル一覧とファイルの先頭だけを読み、チャット用のモデルなら `<home>/catalog-mine.json` に足します。手で確かめた一覧の後ろ、Hugging Face で見つけたものの前に並び、`mine: true` が付きます。足したモデルはほかの一覧のモデルと同じく `pull`、`up --model ID` で使えます。`catalog --remove ID` で外します（ダウンロード済みのファイルは消しません）。その PC の一覧にだけ加わるので、ほかの PC（`worker --serve`）にそのモデルだけを動かしてもらうには、その PC でも足します。Windows アプリでは、モデルの画面の「モデルを追加…」から同じことができます。

```powershell
gpuensemble-plan recommend                       # この PC を自動検出
gpuensemble-plan recommend --rig examples\rig-e5-turing.json --ctx 8192
gpuensemble-plan recommend --min-speed 20        # もっと速さを優先する
```

開発機（2080 Ti + 2080 SUPER、64 GB）では gpt-oss-120b（約 7.7 トークン/秒）を推奨します。`--min-speed 20` なら Qwen3.8-27B（2 枚に全部載って約 23、実測 23.0〜23.5）、`--min-speed 30` なら Qwen3-30B-A3B です。カタログの品質順位は 2025 年時点（Qwen3.8-27B は 2026 年）の公開ベンチマークからの大まかな順位で、自由に書き換えたり追加したりできます。モデルの大きさは各モデルの構造とファイルサイズからの見積もりで、ダウンロード後に `describe` すると正確な値になります。推奨されたモデルがまだ無ければ、`pull` のコマンドも一緒に表示します。

## モデルをダウンロードする（pull）

カタログの各モデルには Hugging Face の配布元（`hf_repo`）と量子化の名前（`hf_quant`）が書いてあり、`pull` で取得できます。ファイル名や分割数（`-00001-of-00003.gguf`）はリポジトリのファイル一覧から決めるので、カタログに書く必要はありません。

```powershell
gpuensemble-plan models                 # カタログと、ダウンロード済みかどうか
gpuensemble-plan pull qwen3-8b         # 終わると先頭ファイルのパスを表示
gpuensemble-plan pull gpt-oss-120b --dir D:\models
```

- 保存先は `%LOCALAPPDATA%\GPUEnsemble\models`（Linux は `~/.cache/gpuensemble/models`）。環境変数 `GPUENSEMBLE_HOME` か `--dir` で変えられます
- 途中で止まっても、同じコマンドでその続きから再開します。回線が切れたときは自動で数回つなぎ直します
- 受け取りながら SHA-256 を計算し、Hugging Face の公表値と照合します。合わないときはファイルを消して知らせます
- 空き容量が足りないときは、始める前に止めます
- ログインが必要なリポジトリは、環境変数 `HF_TOKEN` にトークンを入れておきます（Hugging Face 以外のサーバには送りません）

## llama.cpp を用意する（setup）

```powershell
gpuensemble-plan setup                  # この PC に合う llama.cpp をダウンロードして、GPU が見えるか確かめる
gpuensemble-plan run <file.gguf>        # --exe を省くと setup で入れた llama-server を使う
```

- NVIDIA のドライバが CUDA 12.4 に対応していれば（Windows は 551.61 以降）CUDA 12.4 版と CUDA ランタイム、古ければ Vulkan 版、NVIDIA の GPU が無ければ CPU 版を選びます。Linux は CUDA 版が配布されていないので Vulkan 版です。`--backend` で指定もできます
- 版は開発機で計測した `b12375` に固定しています。`--tag latest` で最新版になります
- 入れた後に `llama-server --list-devices` を実行し、GPU が見えなければエラーにします
- Windows の Visual C++ ランタイムが古い（14.40 未満）と、今の llama.cpp は起動直後に落ちます（開発機は 14.28 でした）。そのときは、ほかのアプリ（Microsoft Edge など）が持っている新しいランタイムを llama.cpp の隣にコピーして直します。見つからなければ、[Visual C++ 再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe) を入れるよう案内します
- 置き場所は `%LOCALAPPDATA%\GPUEnsemble\bin\llama.cpp`（`GPUENSEMBLE_HOME` で変更可）

## まとめて動かす（up）

```powershell
gpuensemble up                          # 検出 → 推奨 → ダウンロード → llama.cpp の用意 → 配置 → llama-server 起動
gpuensemble up --model qwen3-8b        # カタログのモデルを指定
gpuensemble up --model C:\models\x.gguf  # 手元の GGUF を指定（ダウンロードしない）
gpuensemble up --host 0.0.0.0 --port 8080 -- --jinja   # LAN に公開し、-- の後ろは llama-server へ
```

`gpuensemble` は `gpuensemble-plan` の別名です。ダウンロードの前に大きさを表示して確認します（`--yes` で確認なし）。配置は、カタログの見積もり用プロファイルではなく、ダウンロードした実際のファイルから計算します。llama-server には `-a <モデル名>` を付けるので、API の model 名はカタログの id になります。

初めて起動したときは、GPU 用の命令をドライバがその場でコンパイルするため、要求の長さが変わるたびに数十秒かかります。`up` は読み込みが終わると長さの違う短い要求をいくつか自分で送ってから ready を表示するので、この待ち時間は起動中に済みます（コンパイル結果は保存され、次からは速くなります。`--no-warmup` で省略）。`--open` を付けると、ready になった時点でブラウザでチャット画面を開きます（Windows の zip のダブルクリック起動では常に付きます）。llama-server の `/health` は読み込みが終わった時点で OK を返すので、スクリプトから使うときは `/health` ではなく ready の行を待ってください。ウォームアップ中に送った要求は、ウォームアップの要求と取り合いになって遅くなります。Qwen3 や gpt-oss は答える前に考える（推論トークンを出す）ので、`max_tokens` が小さいと考えている途中で打ち切られて本文が空になります。多めに取るか、Qwen3 なら `chat_template_kwargs: {"enable_thinking": false}` を送ってください。開発機での実行結果は [benchmarks/install-and-go-e5.md](../benchmarks/install-and-go-e5.md) にあります。

## 動いている様子を見る（monitor）

```powershell
gpuensemble monitor                 # 一度だけ
gpuensemble monitor --every 2       # 2 秒ごと（Ctrl+C で終了）
gpuensemble monitor --every 2 --json  # 1 行に 1 つの JSON（アプリが読む形）
```

`up` は起動したモデルの API とプラン（段ごとの層と見積もり時間）を `GPUENSEMBLE_HOME\run\up.json` に書き、ready になった時刻とその時点の llama-server のカウンタを足し、止まると消します。`up` が強制終了されてファイルが残っても、`monitor` はその `up` のプロセスがもう無いことを確かめてファイルを消します。`monitor` はそれと nvidia-smi、llama-server の `/metrics`（`up` は `--metrics` を付けて起動します）から次を出します。

- 実際の生成速度とプロンプト処理の速度（ready の後に終わった回答から計算。`up` のウォームアップの要求は数えません。最初の回答が終わるまでは出ません）と、見積もりの速度。10 トークン/秒未満は小数 2 桁で出します
- いまの状態（読み込み・ウォームアップ中、最初の回答中、回答中、待機中）
- この PC のカードごとのメモリ・温度・SM クロック・負荷・電力・P-state と、そのカードが持つ層
- 段ごとの時間とその中でいちばん時間のかかる部分。llama.cpp は段ごとの時間を出さないので、プランの見積もりを実測の速さに合わせて割り振ったものです。gpt-oss のような MoE モデルでは、メモリに残したエキスパートを CPU が計算する時間を、その層を持つカードの段に数えます（その段には「experts on the CPU」と出ます）
- モデルの処理中なのに P3〜P8 の省電力の状態（P-state が読めないときは最高クロックの半分未満）にあるカード（ドライバがクロックを下げている。[benchmarks/clock-hold.md](../benchmarks/clock-hold.md)）。1 トークンの時間の 1 割未満しか持たないカードは、ほとんどの時間を待っているだけなので数えません。ほかに 83 ℃ 以上のカード、見積もりの 8 割に届かない速さ

Windows では `up` と `monitor` は Ctrl+C のほか Ctrl+Break でも同じように止まり、`up` は llama-server を止めて `up.json` を消します。

別の PC のカードは、層の範囲と見積もり時間だけが出ます（温度やクロックはその PC の `nvidia-smi` で見てください）。

## 別の PC の GPU も使う（worker、試験段階）

llama.cpp の RPC（ネットワーク越しに GPU を使う仕組み）を使います。GPU を貸す側の PC で `worker` を起動しておくと、メインの PC の `up` がその GPU にも層を配置し、`--rpc` 付きで llama-server を起動します。

いちばん簡単なのは、同じ LAN で自動的に見つける方法です。

```powershell
# GPU を貸す PC：この PC の NVIDIA GPU を全部、LAN のメインの PC に貸す
gpuensemble worker --lan

# メインの PC：LAN と Tailscale の worker を探して、その GPU も使う
gpuensemble find                      # 見つかった worker と GPU の一覧
gpuensemble up --remote auto          # 自動で見つけて配置・起動
```

Windows の zip なら、貸す PC で `Start GPUEnsemble worker.cmd` をダブルクリックするだけです（`worker --lan --cache --serve` を起動します。受け取った重みを保存するので、2 回目からはネットワークで送り直しません。`--serve` は、この PC の GPU だけに収まるモデルをこの PC で丸ごと動かす依頼を受けます。下の「収まる PC で丸ごと動かす」を参照）。メインの PC の `Start GPUEnsemble.cmd` は `up --remote auto` で起動するので、LAN に worker があればその GPU も使い、なければこの PC の GPU だけで動きます。同じ PC で worker も動いているときは、その GPU を二重に数えないよう、ブロードキャストで見つけたこの PC 自身の worker は使いません（`--remote auto:127.0.0.1` のようにアドレスを指定したときは使います）。

worker は、メインの PC が尋ねるたびにカードの空きメモリを読み直して伝えます。worker を起動したあとで、貸す PC のほかのプログラム（別の LLM や画像生成など）がそのカードを使い始めても、メインの PC は残りに収まる配置を作ります（Windows では、入りきらない分をドライバが共有メモリに逃がすことがあり、そうなると止まらずに極端に遅くなるためです）。空きが起動時より 1.5 GiB 以内だけ少ないときは、worker 自身の RPC サーバが待機中に持つ分とみなして、起動時の値のまま伝えます。値が変わったときは worker のウィンドウに `note: main PCs are now told 4.0 GiB for ...` と出ます。

初めてつなぐときは、貸す側のウィンドウに「PC 'E5' at 192.168.10.1 asks to use these GPUs」と出ます。そこで `y` と Enter を押すとその PC を通し、次からは覚えています（GPUEnsemble のフォルダの `worker.json`）。メインの PC の `up` は、コンソールから起動したときは最大 2 分、この許可を待ちます（その間に貸す側で `y` を押せば、起動し直さずにその GPU も使って続けます。Ctrl+C で待たずに進みます）。待つのは、worker がそのウィンドウで初めて尋ねたときだけです（一度無視された PC は、worker を起動し直すまで待たずに進みます）。`--yes` 付きやスクリプトからの起動では待たず、許可されていない worker の GPU は使いません。許可はメインの PC の IP アドレスで覚えるので、家庭の LAN 向けです。知らない人もつながるネットワークでは、2 台を直結したケーブルを使ってください。見つける仕組みは UDP の 50053 番（worker の番号 + 1）を使います。Windows が Python の通信を許可するか聞いてきたら、プライベート ネットワークだけを許可してください。`find --at 192.168.10.2` や `--remote auto:192.168.10.2` で、ブロードキャストの代わりにそのアドレスだけに尋ねられます。別の拠点の PC は Tailscale でつなげます。ブロードキャストは Tailscale を越えないので、`--remote auto` と `find` は LAN に加えて、この PC の tailnet でオンラインの PC（`tailscale status --json` で読みます。最大 64 台）にも 1 台ずつ問い合わせます。`worker --lan` は、自分と同じアカウントで Tailscale にログインしている PC を、`y` を押さなくても通します（別の拠点の worker の前には誰もいないことが多いためです。Taildrop が自分の端末からだけ受け取るのと同じ線引きで、タグ付きの端末や他のアカウントから共有された端末は対象外です。この許可は覚えず、起動のたびに Tailscale に確かめます。これらも `y` で許可したいときは `worker --lan --ask-tailscale`）。同じ worker が LAN と Tailscale の両方で答えたときは、通してくれるほう、どちらも通すなら LAN のアドレスを使います。Tailscale 以外の VPN では、`--remote auto:10.x.y.z` のように worker のアドレスを指定してください。`--remote auto` は使う worker との往復時間を測り（`remote: ... (round trip 12.0 ms)`）、1 トークンあたり 1 往復分として見積もりに入れます（llama.cpp は入力と計算の指示を待たずに送り、結果の読み出しを 1 回だけ待ちます。2 拠点で通信を記録して確かめました）。往復が長くて速くならないときは、プランナはその PC の GPU を使わない配置を選びます。メインの PC から見つからないときは、貸す側の worker のウィンドウを見てください。Windows では、worker がネットワークの種類（パブリック／プライベート）とこの Python のファイアウォール規則を読み、ほかの PC から届かない理由があれば `note:` で知らせます（家庭の LAN がパブリック扱いになっている、最初の問い合わせでキャンセルしたためブロックされている、など。設定は読むだけで変えません）。

アドレスを自分で指定する方法もあります。

```powershell
# GPU を貸す PC（例: 192.168.10.2）。llama.cpp がなければ setup と同じものを入れる
gpuensemble worker --host 192.168.10.2 --allow 192.168.10.1   # この PC の NVIDIA GPU を全部、メインの PC（192.168.10.1）だけに貸す
gpuensemble worker --host 192.168.10.2 --allow 192.168.10.1 --device CUDA1 --cache   # 1枚だけ。--cache で受け取った重みをこの PC（GPUEnsemble のフォルダの rpc-cache）に保存する
# 標準出力に {"gpus": [...]} が出るので、メインの PC に pc2.json として保存する

# メインの PC
gpuensemble recommend --remote pc2.json             # この PC の GPU と合わせて候補を選ぶ
gpuensemble up --remote pc2.json                    # 配置して起動（--rpc 192.168.10.2:50052 が付く）
```

- **安全のため:** RPC サーバにはパスワードがありません。そのアドレスに届く人は誰でも GPU を使え、GPU のメモリを読み書きできます。そこで `worker` は、指定しなければ 127.0.0.1（その PC の中だけ）で待ち受け、LAN のアドレスを `--host` に指定するときは `--allow` でメインの PC のアドレス（または `192.168.10.0/24` のようなネットワーク）を求めます。このとき RPC サーバは 127.0.0.1 にとどまり、`--host` のアドレスでは小さな門番（gate）が待ち受けて、`--allow` のアドレスからの接続だけを RPC サーバにつなぎます（中身はそのまま中継します）。誰からでも受けるときだけ `--allow-any` を使います。Windows が Python の通信を許可するか聞いてきたら、プライベート ネットワークだけを許可してください。門番は 1 トークンあたり約 1.1 ms かかります（開発機の gpt-oss-20b で約 9% 遅くなる。1 トークンに時間のかかる大きなモデルほど割合は小さい）。プランナはこの分も見積もりに入れます。2 台を直結したケーブルのように、ほかの機器が届かないネットワークなら `--allow-any` で門番を省けます
- **メインの PC が急に消えたとき:** 電源が落ちた、回線が切れた、というとき、メインの PC は接続を閉じる知らせを送れません。そのままだと貸す側の RPC サーバは、その PC が読み込んだモデルを GPU のメモリに持ち続けます。そこで門番は、静かになった接続を OS に確かめさせます（TCP キープアライブ: 60 秒静かなら 6 秒おきに 10 回）。約 2 分返事がなければ接続を閉じ、RPC サーバはモデルをメモリから下ろします（worker の画面に `gate: the PC at … stopped answering` と出ます）。待っているだけのメインの PC は確認に答えるので、何時間でもつながったままです。2 分を超えて回線が切れていたメインの PC は、回線が戻っても接続は閉じているので、`up` をやり直します。`--allow-any` で門番を省いたときはこの確認がないので、消えたメインの PC のモデルを下ろすには worker を起動し直します
- **速さの見積もり:** 別の PC にある段は、同じ GPU がローカルにある場合の時間に、1トークンごとの往復の時間を足して見積もります。同じ PC の中でつないだ（ループバック）実測は 1 段あたり約 0.8 ms でした（[benchmarks/phase2a-rpc-loopback.md](../benchmarks/phase2a-rpc-loopback.md)）。LAN では 2 ms と仮に置いています。実測した値は `--remote` のファイルや rig ファイルの GPU に `rpc_overhead_ms` で書けます。段の間を流れるのは各トークンの隠れ状態だけです。ただし出力の頭を持つ段は、1 トークンごとに語彙の数 × 4 バイト（Qwen3 で 0.6 MB、gpt-oss で 0.8 MB）を返します。別の PC からだと 1 Gb/s で 5〜7 ms かかるので、プランナはこの時間も足し、層を分けて置くときは頭をメインの PC のカードに残します（`network_gbs` を書くとその速さで計算します）
- **読み込み:** 別の PC に置く重みは、起動のたびにネットワークで送ります。1 Gb/s のイーサネットでは 1 GiB あたり約 9 秒かかります（`network_gbs` を書くとその速さで見積もります）。`worker --cache` にすると、貸す側の PC が受け取った重みを保存し、2 回目からは同じ重みを送らずに済みます（`--lan` で見つけた worker が `--cache` なら、メインの PC の note もそう表示します。zip のダブルクリックの worker は最初から `--cache` です）
- **別の PC のカードは余裕を多めに:** 別の PC のカードには、この PC のカードより 1 GiB 多く空きを残して配置します。その PC の画面やほかのプログラムもカードを使っていて、入りきらない分を Windows がエラーにせず共有メモリ（RAM）へ回すため、読み込めても生成が半分の速さになるからです。実測では、別の拠点の RTX 4090 に Qwen3.8-27B Q8_0 を 54 層載せると 20.4 tok/s、56 層（llama.cpp の計算で空き 0）では 10.5 tok/s でした（[benchmarks/phase2c-three-cards.md](../benchmarks/phase2c-three-cards.md)）
- **別の拠点の worker は自分でダウンロード:** 別の拠点の PC に重みを送ると、メインの PC の回線の上り（家庭の回線ではたいてい遅い方向）が上限になります。そこで `up` は、往復 3 ms 以上の Windows の worker（`--cache` 付き）には、その PC に置く層のテンソルが Hugging Face のファイルのどこにあるか（URL とバイト範囲）だけを伝え、worker が自分でその範囲をダウンロードして `rpc-cache` に入れます（`remote: the PC at ... downloads its share of the weights itself`。進み具合も出ます）。2 拠点の実測では、20.8 GiB を 133 秒で取り終えました（メインの PC から送ったときは毎秒 46 MB で約 8 分。[計測](../benchmarks/phase2b-two-sites-tailscale.md)）。llama.cpp の RPC サーバは、受け取った重みを内容のハッシュ（FNV-1a 64 ビット）を名前にして保存するので、同じ名前で置いておけば、読み込みのときメインの PC はハッシュを送るだけで済みます。対象は `pull` で入れたカタログのモデルで、10 MiB を超えるテンソルです（それ以下は llama.cpp がそのまま送ります）。途中で止まっても、取り終えたテンソルは次の起動で取り直しません。前にメインの PC から送られて `rpc-cache` に残っているテンソルは、先頭と末尾の 64 KiB ずつを Hugging Face のものと比べて見分け、取り直しません（大きさだけでは見分けられません。層ごとのテンソルは同じ大きさのことが多いためです）。worker は取り始める前に、Hugging Face のファイルの大きさがメインの PC のファイルと同じかを確かめます（別のリポジトリの同じモデルなど、中身が違うファイルを取っても使えないため。違えば取りません）。ダウンロードできなかったとき（worker が古い、回線が切れた、空き容量が足りない、など）は、これまでどおりメインの PC が送ります。worker が取りに行く先は huggingface.co だけで、頼めるのは worker が通している PC だけです（TCP の 50053 番。見つける仕組みの UDP と同じ番号）。同じ LAN では送るほうが速いので使いません。Windows 以外の worker は、ハッシュの計算が遅く（Python で約 10 MB/s）送るほうが速いので、これを受け付けません。`up --weights fetch` で LAN の worker にも使い、`up --weights send` で常に送ります
- **別の PC 1 台に収まるモデル:** モデル全体が別の PC 1 台の GPU だけに載るときは、その PC 自身で起動した場合の見積もりも出します。メインの PC から起動すると、その PC のカードを使う段はトークンごとにネットワークの往復を待ち、出力の層もそこにあれば結果（語彙の数だけのロジット）の受け取りも待ちます。その PC で起動すれば、どちらもありません。1 割以上速くなるときは、`recommend` の表に `34.6 if started on the other PC` のように出て、`plan` と `up` も `note:` で知らせます。メインの PC のカードに収まる小さいモデルでも、別の PC のカードのほうが速ければ同じように出ます。その PC で直接 `gpuensemble up` するか、次の項目のように worker に頼めば、その速さで動きます。実測では、別の拠点の RTX 4090 に全部載る Qwen3-32B が、メインの PC から起動すると 25.2〜25.9 tok/s、4090 の PC で起動すると 41.5 tok/s でした（[benchmarks/phase2c-three-cards.md](../benchmarks/phase2c-three-cards.md)）
- **収まる PC で丸ごと動かす（`worker --serve`）:** 貸す側の worker を `--serve` 付きで起動しておくと（zip のダブルクリックはこれを含みます）、メインの PC の `up` は、上のように別の PC 1 台に収まって 1 割以上速くなるモデルを、その PC に頼んで丸ごと動かします。頼まれた PC は、モデルがなければ自分でダウンロードし、自分のカードで配置を決めて llama-server を起動します。メインの PC はいつもの `127.0.0.1:8080` で接続を受けて、その PC へ中継します。チャットの画面も API もそのまま使え、トークンごとにネットワークを待つことがなくなります。メインの PC にはモデルをダウンロードしません。`recommend` の表では `runs on the other PC` と出て、速さはその PC での見積もりです。`up` を Ctrl+C で止めるか、メインの PC との接続が切れると、その PC の llama-server も止まり、GPU が空きます。`up --run here` を付けるとこれまでどおりメインの PC で起動します。`--run there` は、頼めないときにエラーで止まります（既定の `auto` は、断られたらメインの PC で起動します）。`--exe`、`--kv`、`--catalog`、llama-server への追加の引数を付けたときは、その PC へ渡せないのでメインの PC で起動します。頼めるのはカタログにあるモデルの ID とコンテキストの長さだけで、ファイル名、ダウンロード元、llama-server の引数は指定できません。ポートは worker のポート + 1（依頼。重みのダウンロードの依頼と同じ）と + 2（モデルの API。頼んだ PC だけを通す門番つき）を使います。1 台の worker が同時に動かすモデルは 1 つです
- **MoE:** 別の PC の GPU に入りきらないエキスパートをメインの PC の CPU で計算すると、その層ごとに往復が増えるので、プランナはその分も時間に足します
- **会話のキャッシュ:** llama-server は終わった会話を RAM に残します（`--cache-ram`、既定 8 GiB）。そのために会話の KV キャッシュをカードから読み戻すので、別の PC のカードからだとダウンロードになります（64 層を置いた 422 トークンの会話で約 0.1 GB、次の会話が始まるまで 2 秒）。そこで `up` は、別の PC に層を置くときは `--cache-ram 0` を付けます。変えたいときは `up ... -- --cache-ram 4096` のように自分で指定してください
- **待っている間のクロック低下:** メインの PC のカードは、別の PC の計算を待つ時間が長いと、ドライバが軽い負荷とみなしてクロックを下げます（RTX 2080 Ti / 2080 SUPER、Windows の既定の電源設定で実測。メインの PC の分が 1 トークン 29 ms から 54 ms に）。この分は見積もりに入っていないので、メインの PC と別の PC に分けた配置は見積もりより 2〜3 割遅くなることがあります（そういう配置には `plan` と `up` が `note:` でそう知らせます）。NVIDIA コントロールパネルの「電源管理モード」を「パフォーマンス最大化を優先」にしても、CUDA で計算しているだけのカードには効きません。効くのは Vulkan などのグラフィックスの API が開いているカードだけです。そこで Windows の `up` は、モデルを分けて置くとき、モデルが動いている間だけ使う手元のカードを Vulkan で開いておきます（1 枚あたり VRAM 7 MiB、待機中の消費電力は約 40 W 増えます）。電源管理モードがほかの設定のときは `note:` でそう知らせます。`--no-hold-clocks` で止められます（[benchmarks/clock-hold.md](../benchmarks/clock-hold.md)）
- **2 台の PC での実測:** 別の拠点の RTX 4090 を Tailscale 経由で足した結果は [benchmarks/phase2b-two-sites-tailscale.md](../benchmarks/phase2b-two-sites-tailscale.md) に、ほかのモデルでの結果と通信の内訳は [benchmarks/phase2c-three-cards.md](../benchmarks/phase2c-three-cards.md) にあります

rig ファイルに直接書く場合は、GPU に `"rpc": "192.168.10.2:50052"` と、その PC での名前 `"llama_device": "CUDA0"` を付けます（`examples/rig-e5-rpc-loopback.json` は開発機の 2080 SUPER を同じ PC の RPC 経由で使う例です）。同じ RPC サーバの GPU は、そのサーバの `-d` の順に並べます。llama.cpp は `--rpc` に並べたサーバの順、各サーバの中では `-d` の順に RPC0、RPC1、… と名前を付けるためです。

## 開発機での見積もりと実測（フェーズ0で補正済み）

`examples/rig-e5-turing.json`（2080 Ti + 2080 SUPER、CUDA 12.4、ドライバ 560）。帯域は実測に合わせた実効値です（[benchmarks/phase0-e5-turing.md](../benchmarks/phase0-e5-turing.md)）。

| モデル | 置き方 | 推定 | プランナの引数で実測 | 手で調整した最良（フェーズ0） |
| --- | --- | --- | --- | --- |
| Qwen3-32B Q4_K_M | 5層を CPU、Ti に 36層、SUPER に 23層 + 出力 | 7.5 | **7.89** | 5.66 |
| Qwen3-32B Q4_K_M（ctx 8192、文脈 7552 まで埋まった状態） | 9層を CPU、KV は q8_0 | 3.6 | **3.42** | |
| gpt-oss-120b | 密部分は全層 GPU、27〜28層分のエキスパートを CPU | 6.5 | **7.80**（RAM に読み込むと 6.55） | 5.14（`-ncmoe 28`） |
| gpt-oss-20b | 2枚に全部 | 91 | **90.1** | 71.6（Vulkan） |

単位はトークン/秒（生成）。手で探した最良と比べて、Qwen3-32B は 39%、gpt-oss-120b は 52% 速くなりました（[benchmarks/phase1a-e5-llamacpp.md](../benchmarks/phase1a-e5-llamacpp.md)）。rig ファイルは、この実測と、文脈の深さを変えた計測、VRAM が足りなくなる境目の計測に合わせて補正してあります。見積もりは実測のおおむね ±12% 以内です（`-t 32` で計測）。

手で探した最良との比較は、どちらも llama-bench がファイルをメモリマップした状態での値です。`up` はモデルを RAM に読み込んで動かし、その方法で計り直すと gpt-oss-120b は 6.55 で、MoE の見積もりはこちらに合わせてあります。メモリマップのままだと、短い計測ほどページフォールトで遅く出ます（[フェーズ1a の再較正](../benchmarks/phase1a-e5-llamacpp.md#calibrating-with-the-weights-read-into-ram)）。

## 仕組み

- **GGUF の読み取り:** ヘッダ（メタデータとテンソル表）だけを読み、重みは読まない。テンソルのサイズは隣り合うオフセットの差から求めるので、新しい量子化形式（MXFP4 など）にも対応表なしで追従できる
- **層の種類が混ざったモデル:** Qwen3.8-27B のように、文脈をすべて見返す層（KV キャッシュを持つ）と、決まった大きさの状態だけを持つ層（再帰層）が混ざったモデルは、層ごとに数え分ける。KV キャッシュは見返す層の分だけで、再帰層の状態は文脈の長さによらず、llama-server が同時に扱う会話の数（4）だけ要る（Qwen3.8-27B では 48 層 × 4 で 598.5 MiB）。どの層が再帰層かは GGUF の `attention.recurrent_layers` から、その一覧が無いファイルでは `full_attention_interval` から読む。先のトークンを予測するための最後のブロック（`nextn_predict_layers`）は、llama.cpp が層として数えるが読み込まないので、大きさ 0 の層として扱う。カタログの見積もり用プロファイルでは、1 層あたりの平均を `state_bytes_per_layer` に書く
- **GPU の速さの表:** 検出した GPU の速さ（VRAM を読む帯域）は [`gpu_table.py`](src/gpuensemble_planner/gpu_table.py) の表から引きます。GTX 900 系から RTX 50 系までの GeForce（デスクトップとノート）、ワークステーション向け（Quadro、RTX A／Ada／PRO Blackwell）、データセンター向け（Tesla P40、T4、A100、H100 など）の 195 件です。ほとんどはカタログの帯域（バス幅 × メモリの速さ）の 75% と見積もった予測で、実測した GPU（いまは 2080 Ti、2080 SUPER、RTX 4090）は実測の値を使います。同じ名前でメモリの量が違う GPU（RTX 3060 の 12 GB と 8 GB など）は、検出した VRAM の量で選びます。ノート向けの GPU にデスクトップ版の値は使いません。表にない GPU は 300 GB/s と仮に置き、名前に `(bandwidth guessed)` と付けます。一覧は `gpus` コマンドか [GPUS.md](GPUS.md) で見られます。実測を足すときは、その GPU の行に `measured_gbs` と `measured_by` を書きます
- **時間のモデル:** 生成は段が順番に動くので、1トークンの時間は「各層の重みを、その層がある場所の帯域で読む時間」の合計。最後の段は出力ヘッドも読む
- **配置:** GPU の並び順ごとに、(段, 層) の動的計画法で「全部入って時間が最小」の連続分割を探す（7枚までは全順列）。入りきらない層は先頭から CPU に残す
- **文脈の深さ:** 1トークンごとに、それまでの文脈ぶんの KV キャッシュも読む。CPU に置いた層のアテンションは重みの読み出しよりずっと遅い（E5 機の f16 キャッシュで約 1.5 GB/s）ので、キャッシュの型ごとに別の速度（rig ファイルの `cpu_kv_gbs`）で見積もる。配置は文脈の半分まで埋まった時点で最速になるように選び、空のときと満杯のときの速度も表示する
- **MoE モード:** 各層の密な部分と KV キャッシュだけを GPU 必須とし、ルーティングされるエキスパートは残りの VRAM に詰め、溢れた分は RAM に置く。エキスパートは一様に使われる前提なので、よく使うものを優先して置く実際のキャッシュより悲観的な見積もりになる。CPU で計算するエキスパートは、RAM に置く量が多いほど 1 バイトあたり遅くなる（開発機では 1 GiB ごとに約 1.6%、rig ファイルの `expert_slowdown_per_gib`）。その量は分割が決まるまでわからないので、決まった量の速さで配置をやり直し、変わらなくなるまで繰り返す

## テスト

```powershell
cd planner
python -m unittest discover -s tests
```

## 既知の制限

- KV キャッシュは全層が通常のアテンションだと仮定している。スライディングウィンドウや MLA（DeepSeek）のモデルでは多めに見積もる
- 計算用バッファは1枚あたり固定 1GiB（`reserve_gib_per_gpu`）で見ている。実測で詰める
- MoE モードで RAM が足りるかは分割を決めた後に確認しているので、RAM が厳しいときは最速ではない分割を見落とすことがある
- 別の PC の GPU（RPC）は、プロンプト処理での遅れ（同じ PC の中でも 512 トークンごとに約 18 ms、LAN では隠れ状態の転送が加わる）を見積もりに入れていない

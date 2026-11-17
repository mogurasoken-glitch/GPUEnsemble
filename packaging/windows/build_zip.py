"""Build the Windows zips: unpack one anywhere and double-click GPUEnsemble.exe or Start GPUEnsemble.cmd.

Two zips come out, the same but for the window app: ``...-windows-x64-app.zip``
with GPUEnsemble.exe, and ``...-windows-x64-cli.zip`` without it (launchers and
the command line only). Without the .NET SDK only the second is made.

The zip holds python.org's embeddable Python (no installer, nothing written
outside the folder), the planner package, the example rig files and three
launchers (the third lends this PC's GPUs to another PC). It needs nothing else on the PC: `up` downloads llama.cpp and
the model itself, into the ``data`` folder next to the launchers.

    python packaging/windows/build_zip.py [--python-zip FILE] [--out DIR]

Without ``--python-zip`` the embeddable Python is downloaded from python.org
and checked against the SHA-256 pinned below (from python.org's
``windows-3.12.10.json``). Runs on any OS; only the result needs Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE = os.path.join(ROOT, "planner", "src", "gpuensemble_planner")
EXAMPLES = os.path.join(ROOT, "planner", "examples")
APP_PROJECT = os.path.join(ROOT, "app", "windows")
APP_EXE = "GPUEnsemble.exe"

PYTHON_VERSION = "3.12.10"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embeddable-amd64.zip"
PYTHON_SHA256 = "156c7eea90d58cd7e91a23f28a0056616b13e9f4cf4901b7b99b837b7848c6da"

# Launchers stay ASCII: cmd.exe reads .cmd files in the console's code page
# (932 on Japanese Windows), so UTF-8 text would print garbled.
CLI_CMD = r"""@echo off
rem GPUEnsemble command line: gpuensemble up, gpuensemble recommend, ...
setlocal
if not defined GPUENSEMBLE_HOME set "GPUENSEMBLE_HOME=%~dp0data"
set "GPUENSEMBLE_PROG=gpuensemble.cmd"
"%~dp0python\python.exe" -m gpuensemble_planner %*
exit /b %ERRORLEVEL%
"""

START_CMD = r"""@echo off
rem Double-click to pick, download and start the best model for this PC.
rem It runs in a window of its own; press Ctrl+C there or close it to stop.
rem (Started with "start" so that no batch file is running in that window,
rem which would ask "Terminate batch job (Y/N)?" on Ctrl+C.)
setlocal
if not defined GPUENSEMBLE_HOME set "GPUENSEMBLE_HOME=%~dp0data"
set "GPUENSEMBLE_PROG=gpuensemble.cmd"
start "GPUEnsemble" "%~dp0python\python.exe" -m gpuensemble_planner.launch %*
"""

WORKER_CMD = r"""@echo off
rem Double-click to lend this PC's GPUs to GPUEnsemble on another PC of the LAN.
rem That PC finds this one by itself; the first time, type y here to let it in.
rem It runs in a window of its own; press Ctrl+C there or close it to stop.
setlocal
if not defined GPUENSEMBLE_HOME set "GPUENSEMBLE_HOME=%~dp0data"
set "GPUENSEMBLE_PROG=gpuensemble.cmd"
start "GPUEnsemble worker" "%~dp0python\python.exe" -m gpuensemble_planner.launch worker %*
"""

README = """GPUEnsemble {version} for Windows
================================
{app}
使い方
------
1. この zip を、空き容量の多いドライブの、英数字だけのフォルダに展開します
   （例: D:\\GPUEnsemble）。モデルは data フォルダに保存され、大きいものは 60 GB あります。
2. Start GPUEnsemble.cmd をダブルクリックします。"GPUEnsemble" というウィンドウが開き、
   PC の GPU・RAM を調べて、動かせるうち最も良いモデルを選び、大きさを表示して
   確認したうえで、モデルと llama.cpp をダウンロードして起動します。
3. "ready" と表示されると、ブラウザでチャット画面（http://127.0.0.1:8080/）が開きます。
   http://127.0.0.1:8080/v1 は OpenAI 互換 API です。初回はその前に GPU 用の命令の
   準備（ウォームアップ）があり、数分かかることがあります。
4. 止めるときは、そのウィンドウで Ctrl+C を押すか、ウィンドウを閉じます。

コマンドで使う場合は、このフォルダで gpuensemble.cmd を使います。
  gpuensemble.cmd recommend          候補のモデルと予想速度の一覧
  gpuensemble.cmd up --model gemma-4-12b  モデルを指定して起動
  gpuensemble.cmd --help             すべてのコマンド

別の PC の GPU も使う（試験段階）
--------------------------------
GPU のある別の PC にもこの zip を展開し、Start GPUEnsemble worker.cmd をダブルクリック
します（Windows に Python の通信を許可するか聞かれたら、プライベート ネットワークだけを
許可します）。次にこの PC で Start GPUEnsemble.cmd を起動すると、同じ LAN の worker を
見つけて問い合わせます。初回は worker のウィンドウに「PC '...' at ... asks to use these
GPUs」と出るので、y と Enter を押して許可します。この PC の GPUEnsemble はそれを最大
2 分待ち、許可されるとそのまま両方の GPU で続けます（待たずに進めるときは Ctrl+C）。
許可は覚えておくので、2 回目からはすぐに両方の GPU を使います。
見つからないときは worker のウィンドウの note: を見てください（ネットワークがパブリック
扱いになっている、ファイアウォールが Python を止めている、などを知らせます）。
worker は受け取った重みを data\\rpc-cache に保存し、2 回目からは送り直さずに済ませます
（容量が必要なときはこのフォルダを消してかまいません）。
worker にはパスワードがなく、許可した PC の IP アドレスだけを通します。知らない人と
共有しているネットワークでは使わないでください。

別の場所にある PC は、両方に Tailscale を入れて同じアカウントでログインしておけば、
同じようにダブルクリックだけでつながります。この PC は LAN に加えて Tailscale の相手にも
問い合わせ、worker は自分と同じアカウントの PC を、y を押さなくても通します（貸す側に
人がいなくてもつながります）。別のアカウントの PC は、LAN と同じく y で許可します。
初回は、worker が自分の受け持つ分の重みを Hugging Face から直接ダウンロードします
（この PC から送るより速いことが多いためです。できなかったときはこの PC が送ります）。

モデルを別の場所に置くときは、環境変数 GPUENSEMBLE_MODELS にフォルダを指定します
（アプリではモデル画面の「モデルの保存先」で選べます。GPUENSEMBLE_HOME は data フォルダごと移します）。
ダウンロードしたファイルを開くと Windows の警告が出ることがあります。そのときは
「詳細情報」から「実行」を選ぶか、展開する前に zip のプロパティで「許可する」に
チェックを入れてください。

Usage
-----
1. Unpack this zip into a folder with plenty of free space and an ASCII-only path
   (for example D:\\GPUEnsemble). Models are stored in its data folder; the largest is 60 GB.
2. Double-click Start GPUEnsemble.cmd. A window named "GPUEnsemble" opens; it inspects the GPUs
   and RAM, picks the best model this PC runs well, shows the download size and asks, then
   downloads the model and llama.cpp and starts the server.
3. When it says "ready", the chat page (http://127.0.0.1:8080/) opens in the browser.
   http://127.0.0.1:8080/v1 is an OpenAI-compatible API. The first start warms up the GPU
   kernels before "ready", which can take a few minutes.
4. Press Ctrl+C in that window, or close it, to stop.

GPUs on another PC (experimental): unpack this zip there too and double-click
Start GPUEnsemble worker.cmd (if Windows asks whether Python may communicate, allow private
networks only). Start GPUEnsemble.cmd on this PC then finds it on the LAN. The first time, the
worker window shows "PC '...' at ... asks to use these GPUs": type y and Enter there.
GPUEnsemble on this PC waits up to 2 minutes for that and then goes on with both PCs' GPUs
(Ctrl+C goes on without them). The approval is remembered. If this PC does not find the
worker, look for "note:" lines in the worker window (a network set to public, or a firewall
rule that blocks Python). The worker keeps the weights it receives in data\\rpc-cache, so later
starts do not send them again (delete that folder to free the space). The worker has no
password and lets in only the IP addresses approved; do not use it on a network shared with
people you do not trust.

A PC at another place links the same way, by double-clicks alone, if both PCs run Tailscale
signed in with the same account: this PC also asks the PCs of its Tailscale network, and the
worker lets in a PC of its own account without the y, so nobody needs to be there. PCs of
other accounts are approved with y, as on the LAN. The first time, the worker downloads its
share of the weights from Hugging Face itself, which is usually faster than this PC sending
them (if it cannot, this PC sends them).

From a terminal in this folder, use gpuensemble.cmd (recommend, up --model ..., --help).
Set GPUENSEMBLE_MODELS (or, in the app, the Models page) to keep the models somewhere else;
GPUENSEMBLE_HOME moves the whole data folder.

Included: Python {python} (embeddable package, python.org, PSF license in python\\LICENSE.txt).
GPUEnsemble is MIT licensed (LICENSE); what it bundles or downloads is listed in
THIRD_PARTY_NOTICES.md. https://github.com/mogurasoken-glitch/GPUEnsemble
"""


def package_version() -> str:
    with open(os.path.join(PACKAGE, "__init__.py"), encoding="utf-8") as f:
        m = re.search(r'^__version__ = "([^"]+)"', f.read(), re.MULTILINE)
    if not m:
        raise SystemExit("no __version__ in gpuensemble_planner/__init__.py")
    return m.group(1)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_python(cache: str) -> str:
    """The embeddable Python zip, downloaded once into ``cache`` and checked."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(PYTHON_URL))
    if not os.path.exists(path):
        print(f"downloading {PYTHON_URL}", file=sys.stderr)
        part = path + ".part"
        req = urllib.request.Request(PYTHON_URL, headers={"User-Agent": "gpuensemble-build"})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(part, path)
    got = sha256_of(path)
    if got != PYTHON_SHA256:
        os.remove(path)
        raise SystemExit(f"{path}: SHA-256 {got} does not match the pinned {PYTHON_SHA256}")
    return path


# What README.txt says first in the zip with the window app.
README_APP = """
アプリ（GPUEnsemble.exe）
------------------------
GPUEnsemble.exe をダブルクリックすると、ウィンドウで GPU・モデル・つながる PC を見て、
起動や停止、別の PC に GPU を貸すことができます。閉じても通知領域のアイコンから戻れます。
下の .cmd を使う方法も、これまでどおり使えます。

The app: double-click GPUEnsemble.exe to see the GPUs, models and other PCs in a window and to
start, stop or lend GPUs from there. The .cmd launchers below work as before.
"""


def _write_text(path: str, text: str, bom: bool = False) -> None:
    # CRLF for cmd.exe and Notepad.
    with open(path, "w", encoding="utf-8-sig" if bom else "ascii", newline="\r\n") as f:
        f.write(text)


def _pth_name(python_dir: str) -> str:
    names = [n for n in os.listdir(python_dir) if n.endswith("._pth")]
    if len(names) != 1:
        raise SystemExit(f"expected one ._pth file in the embeddable Python, found {names}")
    return names[0]


def build_app(out_dir: str, dotnet: str | None = None) -> str | None:
    """GPUEnsemble.exe built from app/windows with the .NET SDK, or None without one.

    The exe targets .NET Framework 4.8, which Windows 10 and 11 already have, so
    the zip carries no .NET runtime. The SDK builds it on any OS.
    """
    dotnet = dotnet or shutil.which("dotnet")
    if dotnet is None:
        return None
    # The window shows its version: the package's, whatever app/windows says.
    subprocess.run([dotnet, "build", APP_PROJECT, "-c", "Release", "-o", out_dir, "--nologo", "-v", "quiet",
                    f"-p:Version={package_version()}"], check=True)
    exe = os.path.join(out_dir, APP_EXE)
    if not os.path.exists(exe):
        raise SystemExit(f"dotnet build did not make {exe}")
    return exe


def add_app(top: str, exe: str) -> None:
    """The window app and its .config (which tells Windows the .NET Framework version it runs on)."""
    shutil.copy(exe, os.path.join(top, APP_EXE))
    config = exe + ".config"
    if os.path.exists(config):
        shutil.copy(config, os.path.join(top, APP_EXE + ".config"))


def assemble(python_zip: str, stage: str, version: str) -> str:
    """Lay out the folder that goes into the zip; returns its path."""
    top = os.path.join(stage, f"GPUEnsemble-{version}")
    python_dir = os.path.join(top, "python")
    os.makedirs(python_dir)
    with zipfile.ZipFile(python_zip) as z:
        z.extractall(python_dir)
    if not os.path.exists(os.path.join(python_dir, "python.exe")):
        raise SystemExit(f"{python_zip} has no python.exe")
    pth = _pth_name(python_dir)
    stdlib = pth[: -len("._pth")] + ".zip"
    # The ._pth file is the whole module search path of an embeddable Python
    # (it also makes it ignore PYTHONPATH and the registry): the standard
    # library, its own folder, and the app folder next to it.
    _write_text(os.path.join(python_dir, pth), f"{stdlib}\n.\n..\\app\n")

    shutil.copytree(PACKAGE, os.path.join(top, "app", "gpuensemble_planner"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(EXAMPLES, os.path.join(top, "examples"))
    # Windows file names ignore case, so the two launchers must differ in more than that.
    _write_text(os.path.join(top, "Start GPUEnsemble.cmd"), START_CMD)
    _write_text(os.path.join(top, "Start GPUEnsemble worker.cmd"), WORKER_CMD)
    _write_text(os.path.join(top, "gpuensemble.cmd"), CLI_CMD)
    write_readme(top, version, app=False)
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(top, "LICENSE"))
    shutil.copy(os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), os.path.join(top, "THIRD_PARTY_NOTICES.md"))
    return top


def write_readme(top: str, version: str, app: bool) -> None:
    text = README.format(version=version, python=PYTHON_VERSION, app=README_APP if app else "")
    _write_text(os.path.join(top, "README.txt"), text, bom=True)


def make_zip(top: str, out_dir: str, kind: str) -> str:
    """``kind`` is "app" (with GPUEnsemble.exe) or "cli" (without): the end of the zip's name."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(top)
    out = os.path.join(out_dir, f"{name}-windows-x64-{kind}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, dirs, files in os.walk(top):
            dirs.sort()
            for n in sorted(files):
                full = os.path.join(root, n)
                rel = os.path.relpath(full, os.path.dirname(top)).replace(os.sep, "/")
                z.write(full, rel)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python-zip", help="embeddable Python zip to use instead of downloading it")
    parser.add_argument("--out", default=os.path.join(ROOT, "dist"), help="output folder (default: dist)")
    parser.add_argument("--cache", default=os.path.join(ROOT, "dist", "cache"),
                        help="where the downloaded Python zip is kept")
    parser.add_argument("--app-exe", help=f"a built {APP_EXE} to put in (default: build app/windows with dotnet "
                        "if it is installed, else leave the app out)")
    parser.add_argument("--require-app", action="store_true",
                        help=f"fail instead of making only the zip without {APP_EXE} when it cannot be built (releases)")
    args = parser.parse_args(argv)
    version = package_version()
    python_zip = args.python_zip or fetch_python(args.cache)
    outs = []
    with tempfile.TemporaryDirectory() as stage:
        top = assemble(python_zip, stage, version)
        exe = args.app_exe or build_app(os.path.join(stage, "app-build"))
        if exe is None and args.require_app:
            raise SystemExit(f"{APP_EXE} cannot be built here: install the .NET SDK or pass --app-exe")
        outs.append(make_zip(top, args.out, "cli"))
        if exe is not None:
            add_app(top, exe)
            write_readme(top, version, app=True)
            outs.append(make_zip(top, args.out, "app"))
        else:
            print(f"note: only the zip without the app was made (no .NET SDK here for {APP_EXE}; "
                  "pass --app-exe or install it)", file=sys.stderr)
    for out in outs:
        print(f"{out}\n  {os.path.getsize(out) / 1024**2:.1f} MiB, SHA-256 {sha256_of(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

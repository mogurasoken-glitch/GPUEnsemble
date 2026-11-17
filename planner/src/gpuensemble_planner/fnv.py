"""The hash llama.cpp's RPC server names its cached tensors by: FNV-1a, 64 bits.

``ggml-rpc-server -c`` keeps each tensor above 10 MiB that it receives in a
file named by the hash of its bytes, as 16 hex digits; the main PC sends the
hash first and the bytes only when the server has no such file
(ggml-rpc.cpp, b12375). A worker that puts a tensor's bytes into that folder
under that name therefore spares the main PC sending them.

FNV-1a takes one multiplication per byte, each depending on the last, so it
cannot be handed to a library call in Python, where it runs at about 10 MB/s:
half an hour for a 20 GiB share. On Windows the files are hashed by a few
lines of C# that PowerShell compiles when it starts (about 1 GB/s); nothing
is installed. Elsewhere, and if PowerShell fails, the Python loop is used;
a worker offers to download its weights itself only where the hash is fast.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from typing import Callable

OFFSET = 0xCBF29CE484222325
PRIME = 0x100000001B3
MASK = (1 << 64) - 1

SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.IO;
public static class GpuEnsembleFnv {
    public static string File(string path) {
        ulong h = 0xcbf29ce484222325UL;
        byte[] buf = new byte[1 << 22];
        using (FileStream f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20,
                                             FileOptions.SequentialScan)) {
            int n;
            while ((n = f.Read(buf, 0, buf.Length)) > 0) {
                unchecked {
                    for (int i = 0; i < n; i++) { h = (h ^ buf[i]) * 0x100000001b3UL; }
                }
            }
        }
        return h.ToString("x16");
    }
}
'@
[Console]::Out.WriteLine('ready')
while (($line = [Console]::In.ReadLine()) -ne $null) {
    try { [Console]::Out.WriteLine('ok ' + [GpuEnsembleFnv]::File($line)) }
    catch { [Console]::Out.WriteLine('error ' + $_.Exception.Message) }
}
"""


def fast() -> bool:
    """Whether files are hashed by compiled code here.

    Where they are not, a worker does not offer to download its weights
    itself: hashing them in Python takes longer than the main PC sending them.
    """
    return sys.platform == "win32"


def fnv1a64(data: bytes, h: int = OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * PRIME) & MASK
    return h


def name_of(h: int) -> str:
    """The cache file's name for a hash, as llama.cpp prints it (%016x)."""
    return f"{h:016x}"


def hash_file_python(path: str) -> str:
    h = OFFSET
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h = fnv1a64(chunk, h)
    return name_of(h)


class FileHasher:
    """Hashes files in one folder, by name; use as a context manager.

    The names must be plain ASCII: they go to PowerShell on its standard
    input, while the folder (which may hold any characters) is its working
    directory.
    """

    def __init__(self, folder: str, native: bool | None = None,
                 popen: Callable = subprocess.Popen, log: Callable[[str], None] = lambda s: None) -> None:
        self.folder = folder
        self.log = log
        self.proc = None
        if native is None:
            native = sys.platform == "win32"
        if native:
            try:
                encoded = base64.b64encode(SCRIPT.encode("utf-16-le")).decode("ascii")
                self.proc = popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                   "-EncodedCommand", encoded],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  cwd=folder, text=True, encoding="ascii", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if self.proc.stdout.readline().strip() != "ready":
                    raise OSError("PowerShell could not compile the hash")
            except (OSError, ValueError) as e:
                self.log(f"note: hashing in Python, which is slow ({e})")
                self.close()

    @property
    def native(self) -> bool:
        return self.proc is not None

    def hash(self, name: str) -> str:
        """The cache name (16 hex digits) for the file ``name`` in the folder."""
        if self.proc is not None:
            try:
                self.proc.stdin.write(name + "\n")
                self.proc.stdin.flush()
                answer = self.proc.stdout.readline().strip()
                if answer.startswith("ok ") and len(answer) == 19:
                    return answer[3:]
                raise OSError(answer or "PowerShell stopped")
            except (OSError, ValueError) as e:
                self.log(f"note: hashing in Python from here on, which is slow ({e})")
                self.close()
        return hash_file_python(os.path.join(self.folder, name))

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.SubprocessError):
            try:
                proc.kill()
            except OSError:
                pass

    def __enter__(self) -> "FileHasher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

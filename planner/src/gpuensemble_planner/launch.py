"""What the Windows zip's double-click launchers run, then wait for Enter.

``Start GPUEnsemble.cmd`` runs ``up --open --remote auto``: it picks, downloads
and serves the best model for this PC, together with the GPUs that other PCs
on the LAN lend it. ``Start GPUEnsemble worker.cmd`` runs ``worker --lan --cache
--serve``: it lends this PC's GPUs to the main PC that asks, once its user
types ``y``, keeps the weights it receives so that later starts load faster,
and runs a model that fits on its own GPUs itself when the main PC asks.

Each launcher starts this in its own console window with ``start``, so no
batch file is running there: Ctrl+C stops the server without cmd.exe's
"Terminate batch job (Y/N)?" question, and the window stays open afterwards
so that the last messages, errors included, can be read.
"""

from __future__ import annotations

import queue
import sys
import threading

from . import cli


class Lines:
    """What is typed in the window, read by one thread and handed out in turn.

    The worker reads typed lines in a thread of its own (``y`` lets a main PC
    in), and that thread is still waiting when the worker stops. If the
    closing prompt read the console too, the two would take turns and the
    first Enter would go to the worker. Reading through one ``Lines``,
    ``hand_over()`` sends the lines typed from then on to the next reader.
    """

    def __init__(self, stream) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._ended = threading.Event()
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._queue.put(line)
        except (OSError, ValueError):
            pass  # the window's input is gone
        self._ended.set()
        self._queue.put("")

    def readline(self) -> str:
        """The next typed line, or "" when nothing more can be typed."""
        mine = self._queue
        while True:
            try:
                # In short steps: on Windows Ctrl+C does not interrupt an endless wait.
                return mine.get(timeout=0.2)
            except queue.Empty:
                if self._ended.is_set() and mine is not self._queue:
                    return ""

    def hand_over(self) -> None:
        """Give the lines typed from now on to whoever calls readline() next."""
        self._queue = queue.Queue()
        if self._ended.is_set():
            self._queue.put("")


def _ask_from(lines: Lines):
    def ask() -> str:
        lines.hand_over()
        line = lines.readline()
        if not line:
            raise EOFError
        return line
    return ask


def main(argv: list[str] | None = None, ask=None, stdin=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    lend = bool(args) and args[0] == "worker"
    lines = None
    saved = sys.stdin
    if lend:
        lines = Lines(stdin if stdin is not None else sys.stdin)
        sys.stdin = lines  # the worker's approval prompt reads it
        # Keeping the weights it receives saves sending them again over the LAN at every start;
        # a model that fits on this PC's cards alone runs here from end to end, which is faster.
        command = ["worker", "--lan", "--cache", "--serve", *args[1:]]
        name = "GPUEnsemble worker"
    else:
        # Opening the chat page is what a double-click user wants once it is ready;
        # GPUs that other PCs on the LAN lend this one join in.
        command = ["up", "--open", "--remote", "auto", *args]
        name = "GPUEnsemble"
    if ask is None:
        ask = _ask_from(lines) if lines is not None else input
    code = 1
    try:
        code = cli.main(command)
    except KeyboardInterrupt:
        code = 130
    except SystemExit as e:  # argparse rejected the arguments
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.stdin = saved
    print(f"\n{name} stopped (exit code {code}). Press Enter to close this window.", flush=True)
    try:
        ask()
    except (EOFError, KeyboardInterrupt):
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())

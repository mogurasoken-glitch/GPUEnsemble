"""Command line: ``python -m gpuensemble_planner ...``

    describe MODEL.gguf              print the per-layer profile as JSON
    plan MODEL --rig RIG.json        print the placement and the speed estimate
    args MODEL --rig RIG.json        print the llama.cpp command that carries out the plan
    run MODEL.gguf --exe llama-server [--rig RIG.json] [-- extra llama.cpp args]
                                     plan, then start llama.cpp with that placement
    recommend [--rig RIG.json]       list which catalog models this PC can run, how fast,
                                     and the best one
    pull MODEL_ID [--dir DIR]        download a catalog model from Hugging Face (resumable)
    setup [--backend B] [--tag T]    download llama.cpp for this PC and check that it sees the GPUs
    up [--model ID|FILE]             all of it: pick the best model for this PC (or the one
                                     given), download it and llama.cpp if needed, place it on
                                     the GPUs and start llama-server
    models                           list the catalog and which models are downloaded
    catalog --update                 bring the list of models up to date (repository, Hugging Face)
    catalog --add REPO [--quant Q]   put any GGUF model on Hugging Face on this PC's own list
    worker [--host ADDRESS]          lend this PC's GPUs to the main PC (llama.cpp RPC server);
                                     prints them as rig entries for --remote on the main PC
    worker --lan                     lend them to main PCs on the LAN that find them
                                     (up --remote auto), each approved here once
    worker --serve                   also run a whole model on this PC when the main PC asks
                                     for one that fits on these GPUs alone (up --run)
    find                             list the workers on the LAN
    state [--remote auto]            what the Windows app shows (GPUs, other PCs, models) as JSON
    monitor [--every S] [--json]     the running model's speed and each card's memory, heat and clocks

plan, args, run, recommend and up take --remote FILE (repeatable): GPUs on
other PCs, as ``worker`` prints them, added to this PC's GPUs or the rig file's.
--remote auto finds the workers started with --lan instead. A model that one
other PC holds on its cards alone is started on that PC by ``up`` when its
worker allows it (--serve), with the API passed on to this PC (``serving``).

MODEL is a .gguf file (any part of a split model), a profile .json written
by ``describe`` or by hand (see planner/examples/), or a catalog id as
``models`` lists them: the downloaded file if there is one, else the
catalog's own description of the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser

from .devices import GIB, detect_host, detect_nvidia, is_loopback, load_gpus, load_rig
from .discovery import discovery_port, find_workers, is_this_pc, measure_rtt, one_address_each
from .fetch import Piece, ask_to_fetch, stage_tensors
from .gguf import GGUFError, read_model
from .hub import find_local, home_dir, models_dir, pull, source_url
from .llamabin import DEFAULT_TAG, installed, setup
from .model import ModelProfile, profile_from_gguf
from .llamacpp import READY_TIMEOUT, llama_command, run
from .plan import Plan, PlanError, make_plan, remote_load_seconds, serving_pc
from .recommend import MIN_SPEED, evaluate, format_table, load_catalog, pick
from .remote import profile_of
from . import __version__
from . import appstate, cardhold, catalogs, monitor, serving, tailscale, worker

# How the user starts this program, for the commands it suggests. The Windows
# zip's launcher sets it to gpuensemble.cmd.
PROG = os.environ.get("GPUENSEMBLE_PROG") or "gpuensemble-plan"


def _load_model(path: str) -> ModelProfile:
    if path.lower().endswith(".json"):
        return ModelProfile.from_json_file(path)
    return profile_from_gguf(read_model(path))


def _named_model(name: str, needs_file: bool = False) -> tuple[ModelProfile, str | None]:
    """The model a command was given, and its .gguf file if there is one.

    ``name`` is a file (.gguf or a profile .json), or a catalog id: then the
    downloaded file, else the catalog's profile, which is enough to plan with.
    """
    if os.path.isfile(name) or name.lower().endswith((".gguf", ".json")):
        return _load_model(name), None if name.lower().endswith(".json") else name
    try:
        entry = _find_entry(name, None)
    except LookupError as e:
        raise ValueError(f"there is no file {name}, and {e}") from None
    path = find_local(entry)
    if path is not None:
        model = _load_model(path)
        model.name = entry.get("name", model.name)  # the file's own name can be anything ("Abl 27b")
        return model, path
    if needs_file:
        raise ValueError(f"{name} is not downloaded yet: `{PROG} pull {name}` downloads it "
                         f"(or `{PROG} up --model {name}` downloads and starts it)")
    try:
        model = profile_of(entry)
    except (LookupError, OSError) as e:
        raise ValueError(f"{name}: its description could not be read ({e})") from None
    print(f"{entry['name']}: not downloaded; using the catalog's description of it", file=sys.stderr)
    model.name = entry.get("name", model.name)
    return model, None


def _gib(n: float) -> str:
    return f"{n / GIB:5.1f}"


def format_plan(plan: Plan) -> str:
    # Wide enough for the longest card name (one on another PC adds "on <address>").
    w = max([34] + [len(s.device) for s in plan.stages])
    lines = [
        f"model:   {plan.model}",
        f"mode:    {plan.mode}   context: {plan.context} tokens   KV cache: {plan.kv_type}",
        "",
        f"{'stage':<{w}} {'layers':>11} {'weights':>8} {'KV':>6} {'head':>6} {'used/cap GiB':>14} {'ms/token':>9}",
    ]
    if plan.cpu_layers:
        lines.append(f"{'CPU (RAM)':<{w}} {f'0-{plan.cpu_layers - 1}':>11}")
    for s in plan.stages:
        layer_range = f"{s.first_layer}-{s.last_layer}" if s.n_layers else "-"
        used = f"{_gib(s.used_bytes).strip()}/{_gib(s.capacity_bytes).strip()}"
        extra = ""
        if plan.mode == "moe":
            extra = f"   experts on GPU: {s.expert_fraction_on_gpu:.0%}"
            if s.expert_path:
                extra += f", rest via {s.expert_path}"
        lines.append(
            f"{s.device:<{w}} {layer_range:>11} {_gib(s.weight_bytes):>8} {_gib(s.kv_bytes):>6} "
            f"{_gib(s.head_bytes):>6} {used:>14} {s.seconds_per_token * 1000:9.1f}{extra}"
        )
    lines += [
        "",
        f"weights in RAM: {plan.cpu_bytes / GIB:.1f} GiB",
        f"estimated generation speed: {plan.tokens_per_second:.1f} tokens/s with {plan.depth} tokens in the "
        f"context ({plan.seconds_per_token * 1000:.1f} ms per token); "
        f"{plan.tokens_per_second_empty:.1f} with an empty context, {plan.tokens_per_second_full:.1f} with a full one",
        "  (within ~10% on a rig file fitted to measurements; without one, the CPU part is a rough guess)",
    ]
    for note in plan.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _hardware(args: argparse.Namespace, wait: float = 0.0):
    if args.rig:
        devices, host = load_rig(args.rig)
        if args.ram_gib is not None:
            host.ram_bytes = int(args.ram_gib * GIB)
    else:
        devices = detect_nvidia()
        host = detect_host(args.ram_gib)
        print(f"detected: {len(devices)} GPU(s), {host.ram_bytes / GIB:.0f} GiB of RAM for weights, "
              f"{host.threads} CPU threads (a rig file gives better speed estimates)", file=sys.stderr)
    for path in getattr(args, "remote", None) or []:
        if path == "auto" or path.startswith("auto:"):
            remote = found_remote(path, wait)
        else:
            remote = load_remote(path)
            print(f"remote: {len(remote)} GPU(s) from {path}: " +
                  ", ".join(f"{d.name} ({d.rpc})" for d in remote), file=sys.stderr)
        devices = devices + remote
    names = [d.name for d in devices]
    if len(set(names)) != len(names):
        raise ValueError("two GPUs have the same name; rename one in the rig or --remote file")
    return devices, host


def _find(spec: str, port: int = worker.DEFAULT_PORT):
    """The workers that answer: at the addresses of auto:ADDR,..., else on the LAN and this PC's tailnet."""
    if ":" in spec:
        at = [a.strip() for a in spec.split(":", 1)[1].split(",") if a.strip()]
        return find_workers(at=at, port=discovery_port(port))
    # A broadcast does not cross Tailscale: ask each PC of the tailnet that is online.
    also = [p.address for p in tailscale.peers(tailscale.read_status())]
    workers = find_workers(port=discovery_port(port), also=also)
    return one_address_each(workers, tailscale.is_tailnet_address)


# How long `up` waits for a worker's user to type y the first time (Ctrl+C goes on without it).
APPROVAL_WAIT = 120.0


def found_remote(spec: str, wait: float = 0.0, sleep=time.sleep, workers=None):
    """GPUs of the workers that answer on the LAN or the tailnet (--remote auto) and let this PC in.

    Asked by broadcast, this PC's own worker answers too; it is left out, as
    its GPUs are used here directly. Named addresses (auto:ADDR) are used as given.
    With ``wait``, a worker that has not let this PC in yet is asked again
    for up to that many seconds, so that its user can type y there.
    """
    if workers is None:
        workers = _find(spec)
    used = []
    for w in workers:
        if spec == "auto" and is_this_pc(w):
            print(f"remote: left out the worker on this PC itself ({w.name} at {w.endpoint}); "
                  "its GPUs are used here directly", file=sys.stderr)
        else:
            used.append(w)
    if wait > 0 and any(w.asking for w in used):
        used = _wait_for_approval(used, wait, sleep)
    devices = []
    for w in used:
        if w.allowed:
            far = ""
            if not is_loopback(w.endpoint):
                # The time per token on that PC depends on it: a few ms on a LAN, far more between sites.
                w.rtt = measure_rtt(w)
                if w.rtt is not None:
                    far = f" (round trip {w.rtt * 1000:.1f} ms)"
            gpus = load_gpus(w.rig_entries())
            devices += gpus
            print(f"remote: {w.name} at {w.endpoint}{far}: " + ", ".join(d.name for d in gpus), file=sys.stderr)
        else:
            print(f"remote: {w.name} at {w.endpoint} has not let this PC in yet; type y in its worker window "
                  "and start again", file=sys.stderr)
    if not workers:
        print(f"remote: no other PC lends its GPUs on this LAN or this PC's Tailscale network (to add one, "
              f"start `{PROG} worker --lan` there, or Start GPUEnsemble worker.cmd from the Windows zip, "
              "and let Python through its firewall on private networks)", file=sys.stderr)
    return devices


def fetch_on_workers(plan: Plan, path: str, entry: dict, base: str | None, devices: list, cpu_experts: list[int],
                     mode: str = "auto", log=lambda s: None, ask=ask_to_fetch, every: float = 10.0) -> None:
    """Let the workers at other sites download their share of the weights themselves (see ``fetch``).

    ``mode`` auto asks the workers that are far (``devices.FAR_RTT``), fetch asks
    every worker that can, send asks none. A worker that cannot do it is
    sent the weights by llama.cpp as before.
    """
    if mode == "send":
        return
    by_name = {d.name: d for d in devices}
    stages = [s for s in plan.stages if s.n_layers]
    wanted: dict[str, list[Piece]] = {}
    model = urls = None
    sizes: dict[str, int] = {}
    for s in stages:
        dev = by_name[s.device]
        if not (dev.is_remote and dev.rpc_fetch and dev.rpc_cache) or is_loopback(dev.rpc):
            continue
        if mode == "auto" and not dev.is_far:
            continue
        if model is None:
            model = read_model(path)
            urls = [source_url(entry, part, base) for part in model.parts]
            if not all(urls):
                return  # not where pull put it: its source is not known
            # For the worker to see that the file on Hugging Face is this one, before it downloads from it.
            sizes = {url: os.path.getsize(part) for url, part in zip(urls, model.parts)}
        tensors = stage_tensors(model, s.first_layer, s.last_layer, cpu_experts, head=s is stages[-1])
        wanted.setdefault(dev.rpc, []).extend(Piece(urls[t.part], t.start, t.size) for t in tensors)
    for endpoint, pieces in wanted.items():
        if not pieces:
            continue
        host, _, port = endpoint.rpartition(":")
        size = sum(p.size for p in pieces) / GIB
        started = time.monotonic()
        told = [0.0]

        def progress(done: int, total: int, host: str = host) -> None:
            now = time.monotonic()
            if not total:
                return  # it has them all
            if not told[0]:
                log(f"remote: the PC at {host} downloads its share of the weights itself "
                    f"({total / GIB:.1f} GiB, from Hugging Face), so that this PC need not send them")
            elif now - told[0] >= every:
                log(f"remote:   {done / GIB:.1f} / {total / GIB:.1f} GiB")
            else:
                return
            told[0] = now

        try:
            result = ask(host.strip("[]"), discovery_port(int(port)), pieces, progress, sizes=sizes)
        except OSError as e:
            log(f"remote: the PC at {host} could not download its share of the weights itself ({e}); "
                "this PC sends them as the model loads")
            continue
        if result.get("fetched"):
            log(f"remote: the PC at {host} has them now ({result.get('bytes', 0) / GIB:.1f} GiB "
                f"in {time.monotonic() - started:.0f} s)")
        else:
            log(f"remote: the PC at {host} has its share of the weights already ({len(pieces)} tensors, "
                f"{size:.1f} GiB)")


def _wait_for_approval(workers, wait: float, sleep) -> list:
    """The workers, asked again until those whose user was just asked let this PC in, or ``wait`` seconds pass.

    A worker that asked its user on an earlier start and still does not let
    this PC in was ignored there; it is not waited for.
    """
    workers = list(workers)
    pending = [w for w in workers if w.asking]
    names = ", ".join(f"{w.name} at {w.address}" for w in pending)
    print(f"remote: {names} can lend GPUs to this PC once you type y in its worker window; "
          f"waiting up to {wait:.0f} s for that (Ctrl+C goes on without it)", file=sys.stderr, flush=True)
    deadline = time.monotonic() + wait
    try:
        while pending and time.monotonic() < deadline:
            sleep(1.0)
            for w in pending:
                for again in find_workers(at=[w.address], port=discovery_port(w.port), timeout=1.0):
                    if again.port == w.port and again.allowed:
                        workers[workers.index(w)] = again
                        print(f"remote: {again.name} at {again.endpoint} let this PC in",
                              file=sys.stderr, flush=True)
            pending = [w for w in workers if w.asking and not w.allowed]
    except KeyboardInterrupt:
        print("remote: going on without the PCs that have not let this one in", file=sys.stderr, flush=True)
    return workers


def _find_cmd(args: argparse.Namespace) -> int:
    workers = _find("auto:" + args.at if args.at else "auto", args.port)
    if not workers:
        print("no worker answered on the LAN or this PC's Tailscale network "
              "(start `worker --lan` on the other PC)", file=sys.stderr)
        return 1
    for w in workers:
        state = "lets this PC in" if w.allowed else "has not let this PC in yet: type y in its worker window"
        if is_this_pc(w):
            state += " (it runs on this PC itself, so --remote auto leaves it out)"
        print(f"{w.name} at {w.endpoint}: {state}")
        for g in w.gpus:
            print(f"  {g.get('llama_device', '?')}: {g.get('name', '?')}, {g.get('vram_gib', '?')} GiB")
    return 0


def load_remote(path: str):
    """GPUs on other PCs from a --remote file: {"gpus": [...]} as ``worker`` prints it, or the list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("gpus") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not all(isinstance(e, dict) and e.get("rpc") for e in entries):
        raise ValueError(f"{path}: every GPU in a --remote file needs \"rpc\": \"host:port\"")
    return load_gpus(entries)


def _recommend(args: argparse.Namespace) -> int:
    try:
        devices, host = _hardware(args)
        host.expert_path = "cpu"  # the plan is carried out by llama.cpp
        entries = load_catalog(args.catalog)
        candidates = evaluate(entries, devices, host, args.ctx)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    best = pick(candidates, args.min_speed)
    print(format_table(candidates, best, args.ctx, args.min_speed))
    unread = unread_note(entries, candidates)
    if unread:
        print(unread)
    if best is not None:
        rig = f" --rig {args.rig}" if args.rig else ""
        rig += "".join(f" --remote {r}" for r in getattr(args, "remote", None) or [])
        local = find_local(best.entry)
        how = f"it runs on the other PC, which downloads {best.entry['file_gib']:.0f} GiB first if it does not " \
              "have it" if best.there else "already downloaded" if local else \
            f"downloads {best.entry['file_gib']:.0f} GiB first, and llama.cpp if needed"
        print(f"to start it ({how}):")
        print(f"  {PROG} up --model {best.entry['id']}{rig} --ctx {args.ctx}")
    return 0 if best is not None else 1


def unread_note(entries: list[dict], candidates: list) -> str:
    """A line saying how many models of the list were left out because their files have not been read yet."""
    left = len(entries) - len(candidates)
    if left <= 0:
        return ""
    return (f"{left} more model(s) in the list are not read yet and were left out; "
            f"`{PROG} catalog --update` reads them (the app does this when it starts)")


def _find_entry(model_id: str, catalog: str | None) -> dict:
    entries = load_catalog(catalog)
    for e in entries:
        if e["id"] == model_id:
            return e
    raise LookupError(f"{model_id} is not in the catalog; choose one of: " + ", ".join(e["id"] for e in entries))


def _pull(args: argparse.Namespace) -> int:
    try:
        path = pull(_find_entry(args.model_id, args.catalog), base=args.dir)
    except (LookupError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped; run the same command again to resume", file=sys.stderr)
        return 130
    print(path)
    return 0


def _setup(args: argparse.Namespace) -> int:
    try:
        inst = setup(tag=args.tag, backend=args.backend, force=args.force)
    except (LookupError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped; run setup again to resume the download", file=sys.stderr)
        return 130
    for d in inst.devices:
        print(f"  {d}", file=sys.stderr)
    print(inst.server)
    return 0


def _default_exe(tool: str) -> str | None:
    inst = installed()
    if inst is None:
        return None
    return inst.tool("llama-bench" if tool == "bench" else "llama-server")


def _confirm(question: str, yes: bool) -> bool:
    if yes or not sys.stdin.isatty():
        return True
    answer = input(f"{question} [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def _start_there(args: argparse.Namespace, entry: dict, devices, host, log) -> int | None:
    """Start a catalog model on the other PC that holds all of it; None when it is to start on this PC.

    Decided from the catalog's description of the model, before anything is
    downloaded here: this PC needs no copy of a model that the other PC runs.
    The other PC's worker downloads, plans and starts it (``serving``); this
    PC listens where the API was asked for and passes the connections on.
    """
    if args.run == "here":
        return None
    try:
        model = profile_of(entry)
    except (LookupError, OSError, ValueError):
        return None  # planned from the file once it is downloaded
    try:
        plan = make_plan(model, devices, host, context=args.ctx, kv_type=args.kv)
    except PlanError:
        return None  # said in full on the usual way
    there = serving_pc(plan, devices)
    if there is None:
        if args.run == "there":
            log(f"error: --run there: {entry['name']} is not a model that fits on one other PC that runs models "
                "itself (a worker started with --serve, or by double-click)")
            return 1
        return None
    # What the other PC decides for itself: its own llama-server, its own catalog, its own plan.
    kept = "--exe" if args.exe else "--kv" if args.kv != "auto" else "--catalog" if args.catalog else \
        "arguments for llama-server" if args.extra else ""
    if kept:
        if args.run == "there":
            log(f"error: --run there cannot pass {kept} on to the other PC")
            return 1
        log(f"note: {entry['name']} fits on the other PC alone and would be faster started there, but {kept} "
            "cannot be passed on to it; starting it on this PC")
        return None
    address, rpc_port = there
    log("")
    log(f"{entry['name']} fits on the other PC ({address}) alone: starting it there and passing its API on to this "
        f"PC, about {plan.tokens_per_second_on_other_pc:.0f} tokens/s instead of {plan.tokens_per_second:.0f} "
        f"(`--run here` starts it on this PC with that PC's cards, as before)")
    try:
        relay = serving.Relay((args.host, args.port), (address, serving.api_port(rpc_port)), log)
    except OSError as e:
        log(f"error: this PC cannot listen on {args.host}:{args.port} ({e}); choose another --port")
        return 1
    say = lambda line: log(f"[{address}] {line}")  # noqa: E731
    try:
        session = serving.ask(address, discovery_port(rpc_port), entry["id"], args.ctx, say)
    except KeyboardInterrupt:
        relay.close()
        log("\nstopped")
        return 130
    except OSError as e:
        relay.close()
        if args.run == "there":
            log(f"error: the other PC did not start it ({e})")
            return 1
        log(f"note: the other PC did not start it ({e}); starting it on this PC with that PC's cards")
        if entry.get("mine"):
            log(f"  {entry['name']} was added on this PC only: `{PROG} catalog --add {entry['hf_repo']} --quant "
                f"{entry['hf_quant']}` on the other PC too lets it run there by itself")
        return None
    try:
        relay.target = (address, session.port)
        relay.start()
        shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        log(f"ready: http://{shown}:{args.port}/v1 (the model runs on the PC at {address}; chat at "
            f"http://{shown}:{args.port}/, Ctrl+C stops it there)")
        if args.open:
            try:
                webbrowser.open(f"http://{shown}:{args.port}/")
            except Exception as e:  # noqa: BLE001 - a convenience must not stop the model
                log(f"note: {e}")
        return session.wait(say)
    except KeyboardInterrupt:
        log("\nstopped; the other PC frees its GPUs")
        return 130
    finally:
        relay.close()
        session.close()


def _up(args: argparse.Namespace) -> int:
    """detect -> recommend -> pull -> setup -> plan on the real file -> llama-server."""
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    try:
        # Someone at the console can walk over to a worker and type y there the first time.
        wait = APPROVAL_WAIT if not args.yes and sys.stdin.isatty() else 0.0
        devices, host = _hardware(args, wait)
        host.expert_path = "cpu"  # the plan is carried out by llama.cpp
        entry, path = None, None
        if args.model and args.model.lower().endswith(".gguf"):
            path = args.model
        elif args.model:
            entry = _find_entry(args.model, args.catalog)
        else:
            entries = load_catalog(args.catalog)
            candidates = evaluate(entries, devices, host, args.ctx)
            best = pick(candidates, args.min_speed)
            log(format_table(candidates, best, args.ctx, args.min_speed))
            unread = unread_note(entries, candidates)
            if unread:
                log(unread)
            if best is None:
                return 1
            entry = best.entry
        if entry is not None:
            code = _start_there(args, entry, devices, host, log)
            if code is not None:
                return code
            path = find_local(entry, args.dir)
            if path is None:
                size = entry.get("file_gib", 0)
                if not _confirm(f"download {entry['name']} ({size:.1f} GiB) from Hugging Face?", args.yes):
                    log("stopped; nothing was downloaded")
                    return 1
                path = pull(entry, base=args.dir)
        exe = args.exe
        if exe is None:
            inst = installed() or setup(log=log)
            exe = inst.server
        log(f"model: {path}")
        model = _load_model(path)
        plan = make_plan(model, devices, host, context=args.ctx, kv_type=args.kv)
        command = llama_command(plan, model, devices, tool="server", threads=host.threads, ram_bytes=host.ram_bytes)
    except KeyboardInterrupt:
        log("\nstopped; run up again to continue where it stopped")
        return 130
    except (GGUFError, PlanError, LookupError, RuntimeError, ValueError, OSError) as e:
        log(f"error: {e}")
        return 1
    log("")
    log(format_plan(plan))
    for note in command.notes:
        log(f"note: {note}")
    if not _rpc_servers_up(command, log):
        return 1
    if entry is not None:
        try:
            fetch_on_workers(plan, path, entry, args.dir, devices, command.cpu_expert_layers, args.weights, log)
        except KeyboardInterrupt:
            log("\nstopped")
            return 130
        except (GGUFError, OSError, ValueError) as e:
            log(f"remote: could not ask the other PCs to download their weights ({e}); this PC sends them")
    name = entry["id"] if entry else os.path.splitext(os.path.basename(path))[0]
    extra = ["--host", args.host, "--port", str(args.port), "-a", name]
    if any(not is_loopback(server) for server in command.rpc_servers) and not {"--cache-ram", "-cram"} & set(args.extra):
        # llama-server keeps finished conversations in RAM (8 GiB by default) to come back to them without
        # reading the prompt again. For that it copies their KV cache off the cards, and from a card on
        # another PC that is a download: 2 s and 0.1 GB for a 422-token chat with 64 layers there, before
        # the next chat starts (benchmarks/phase2c-three-cards.md). Putting it back would be the same upload.
        extra += ["--cache-ram", "0"]
        log("note: finished chats are not kept in RAM (--cache-ram 0): with layers on another PC that would "
            "download their cache over the network before each new chat")
    if "--metrics" not in args.extra:
        extra.append("--metrics")  # tokens and seconds for `monitor` and the app
    extra += args.extra
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    log(f"starting llama-server with {name}; when it says ready, chat at http://{shown}:{args.port}/ "
        f"or use the OpenAI-compatible API at http://{shown}:{args.port}/v1 (Ctrl+C stops it)")
    log("starting: " + command.shell(exe, model=path, extra=extra))
    hold = None if args.no_hold_clocks else cardhold.hold_for(plan, devices, log)
    run_path = None
    try:
        run_path = monitor.write_run(monitor.run_info(plan, devices, name, args.host, args.port))
    except OSError as e:
        log(f"note: monitor cannot see this model ({e})")

    def on_ready(host: str, port: int) -> None:
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        if run_path is not None:
            # The warm-up's requests are not answers: `monitor` counts the speed from here.
            monitor.mark_ready(monitor.read_metrics(shown, port), run_path)
        if args.open:
            url = f"http://{shown}:{port}/"
            log(f"opening {url} in the browser")
            webbrowser.open(url)
    try:
        return run(exe, command, path, extra, warmup=not args.no_warmup, on_ready=on_ready,
                   ready_timeout=READY_TIMEOUT + remote_load_seconds(plan, devices))
    finally:
        if run_path is not None:
            monitor.clear_run(run_path)
        if hold is not None:
            hold.close()


def _monitor(args: argparse.Namespace) -> int:
    """The running model and this PC's cards, once or every few seconds."""
    def emit(snap: dict) -> None:
        if args.json:
            # One line per reading; the app reads UTF-8 whatever the console's code page is.
            sys.stdout.buffer.write(json.dumps(snap, ensure_ascii=False).encode("utf-8") + b"\n")
        else:
            print(monitor.format_text(snap) + ("\n" if args.every else ""))
        sys.stdout.flush()

    count = args.count if args.count is not None else (0 if args.every else 1)
    try:
        monitor.watch(args.every or 0.0, count, emit)
    except KeyboardInterrupt:
        return 0
    except (BrokenPipeError, OSError):  # the app closed its end
        return 0
    return 0


def _worker(args: argparse.Namespace) -> int:
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    wanted = [w.strip() for w in args.device.split(",") if w.strip()] if args.device else None
    try:
        allow = args.allow.split(",") if args.allow else None
        host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
        return worker.start(host, args.port, wanted, args.cache, detect=detect_nvidia,
                            install=lambda: installed() or setup(log=log), exe=args.exe, log=log,
                            allow=allow, allow_any=args.allow_any, lan=args.lan,
                            tailnet=None if args.ask_tailscale else tailscale.read_status, serve=args.serve)
    except KeyboardInterrupt:
        log("\nstopped")
        return 130
    except (LookupError, RuntimeError, ValueError, OSError) as e:
        log(f"error: {e}")
        return 1


def _rpc_servers_up(command, log) -> bool:
    """Whether every --rpc server of the command answers; says what to do if not."""
    missing = worker.unreachable_servers(command.rpc_servers)
    for server in missing:
        log(f"error: no RPC server answers at {server}; start `{PROG} worker --host <its address>` on that PC "
            "(and let it through that PC's firewall), or leave its GPUs out of --remote / the rig file")
    return not missing


def _state(args: argparse.Namespace) -> int:
    """Everything the Windows app shows, as JSON on stdout (see appstate)."""
    notes: list[str] = []
    found = []
    try:
        if args.rig:
            devices, host = load_rig(args.rig)
            if args.ram_gib is not None:
                host.ram_bytes = int(args.ram_gib * GIB)
        else:
            try:
                devices = detect_nvidia()
            except RuntimeError as e:  # no NVIDIA driver: the app still shows the models and the other PCs
                notes.append(str(e))
                devices = []
            host = detect_host(args.ram_gib)
        for spec in args.remote or []:
            if spec == "auto" or spec.startswith("auto:"):
                workers = _find(spec)
                found += workers
                devices = devices + found_remote(spec, workers=workers)
            else:
                devices = devices + load_remote(spec)
        host.expert_path = "cpu"  # the plan is carried out by llama.cpp
        entries = load_catalog(args.catalog)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    candidates = evaluate(entries, devices, host, args.ctx)
    best = pick(candidates, args.min_speed)
    state = appstate.build(
        devices, host, candidates, best,
        local_paths={e["id"]: find_local(e, args.dir) for e in entries},
        workers=[(w, is_this_pc(w)) for w in found],
        context=args.ctx, home=home_dir(), models_dir=args.dir or models_dir(),
        llama_cpp=installed() is not None, notes=notes)
    out = json.dumps(state, ensure_ascii=False, indent=1)
    # The app reads UTF-8 whatever the console's code page is.
    sys.stdout.buffer.write(out.encode("utf-8") + b"\n")
    sys.stdout.flush()
    return 0


def _catalog(args: argparse.Namespace) -> int:
    if args.add:
        try:
            entry, notes = catalogs.add_mine(args.add, args.quant, args.name)
        except (LookupError, ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"added {entry['id']}: {entry['name']} from {entry['hf_repo']}, {entry['file_gib']:.1f} GiB")
        for n in notes:
            print(f"note: {n}")
        print(f"  `{PROG} up --model {entry['id']}` downloads and starts it")
        return 0
    if args.remove:
        if not catalogs.remove_mine(args.remove):
            print(f"error: {args.remove} is not a model added on this PC", file=sys.stderr)
            return 1
        print(f"removed {args.remove} from this PC's list (a downloaded file is kept)")
        return 0
    if args.update:
        return catalogs.update(discover=not args.no_discover)
    checked = catalogs.checked_list()
    found = sum(1 for e in catalogs.load_all() if e.get("new"))
    print(f"checked list: {len(checked['models'])} models, updated {checked.get('updated', '?')}")
    print(f"found on Hugging Face: {found} models ({catalogs.found_path()})")
    added = catalogs.mine()
    print(f"added on this PC: {len(added)} models ({catalogs.mine_path()})")
    for e in added:
        print(f"  {e['id']}: {e['hf_repo']} {e['hf_quant']}")
    return 0


def _models(args: argparse.Namespace) -> int:
    print(f"models folder: {args.dir or models_dir()}")
    entries = load_catalog(args.catalog)
    sources = [f"{e.get('hf_repo', '-')} {e.get('hf_quant', '')}".strip() for e in entries]
    width = max(len(s) for s in sources + ["source"])
    id_width = max(len(e["id"]) for e in entries + [{"id": "id"}])
    print(f"   {'id':<{id_width}} {'size GiB':>8}  {'source':<{width}}  downloaded")
    for e, source in zip(entries, sources):
        local = find_local(e, args.dir)
        size = f"{e['file_gib']:8.1f}" if e.get("file_gib") else f"{'?':>8}"
        print(f"   {e['id']:<{id_width}} {size}  {source:<{width}}  {local or '-'}")
    return 0


def _break_is_ctrl_c() -> None:
    """Ctrl+Break (and CTRL_BREAK_EVENT from a script) stops a command the way Ctrl+C does.

    Python ends at once on Ctrl+Break by default, with no clean-up: `up` then
    left its run file behind, and whatever else a `finally` would have done.
    """
    import signal

    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, signal.default_int_handler)
        except ValueError:  # not the main thread (tests)
            pass


def main(argv: list[str] | None = None) -> int:
    _break_is_ctrl_c()
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"GPUEnsemble {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gpus", help="list the GPU bandwidth table the planner uses (Markdown)")

    p_desc = sub.add_parser("describe", help="print a model's per-layer profile as JSON")
    p_desc.add_argument("model")

    p_plan = sub.add_parser("plan", help="place a model on the GPUs and estimate its speed")
    p_plan.add_argument("model")
    p_plan.add_argument("--rig", help="rig description JSON (default: detect GPUs with nvidia-smi)")
    p_plan.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    p_plan.add_argument("--mode", choices=["auto", "whole", "moe"], default="auto")
    p_plan.add_argument("--kv", choices=["auto", "f16", "q8_0", "q4_0"], default="auto",
                        help="KV cache type (default auto: q8_0 only when it is clearly faster)")
    p_plan.add_argument("--ram-gib", type=float, default=None,
                        help="RAM for weights (default: the rig file, or all but max(8 GiB, 20%%) of this PC's RAM)")
    p_plan.add_argument("--split-experts", action="store_true",
                        help="MoE models: plan for an executor that keeps part of a layer's experts on the GPU "
                             "and streams the rest over PCIe when that is faster (the rig file's expert_path). "
                             "llama.cpp cannot run that plan, so `up` and the default of `plan` do not make it")

    p_args = sub.add_parser("args", help="print the llama.cpp command for the plan")
    p_run = sub.add_parser("run", help="plan, then start llama.cpp with that placement")
    for p in (p_args, p_run):
        p.add_argument("model")
        p.add_argument("--rig", help="rig description JSON (default: detect GPUs with nvidia-smi)")
        p.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
        p.add_argument("--mode", choices=["auto", "whole", "moe"], default="auto")
        p.add_argument("--kv", choices=["auto", "f16", "q8_0", "q4_0"], default="auto")
        p.add_argument("--ram-gib", type=float, default=None)
        p.add_argument("--tool", choices=["server", "bench"], default="server",
                       help="llama-server/llama-cli (default) or llama-bench argument style")
    p_args.add_argument("--exe", default=None, help="program name to print (default llama-server, or llama-bench with --tool bench)")
    p_args.add_argument("--windows", action="store_true", help="print a PowerShell line")
    p_run.add_argument("--exe", default=None,
                       help="path to llama-server, llama-cli or llama-bench (default: the one `setup` installed)")
    p_run.add_argument("extra", nargs="*", help="extra arguments passed to llama.cpp (after --)")

    p_rec = sub.add_parser("recommend", help="which models this PC can run, and the best one")
    p_rec.add_argument("--rig", help="rig description JSON (default: detect this PC)")
    p_rec.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    p_rec.add_argument("--min-speed", type=float, default=MIN_SPEED,
                       help=f"tokens/s the recommended model must reach with half the context filled (default {MIN_SPEED:g})")
    p_rec.add_argument("--ram-gib", type=float, default=None)
    p_rec.add_argument("--catalog", help="model catalog JSON (default: the built-in one)")

    p_state = sub.add_parser("state", help="what the Windows app shows: GPUs, other PCs and models, as JSON")
    p_state.add_argument("--rig", help="rig description JSON (default: detect this PC)")
    p_state.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    p_state.add_argument("--min-speed", type=float, default=MIN_SPEED,
                         help=f"tokens/s the recommended model must reach (default {MIN_SPEED:g})")
    p_state.add_argument("--ram-gib", type=float, default=None)
    p_state.add_argument("--catalog", help="model catalog JSON (default: the built-in one)")
    p_state.add_argument("--dir", default=None, help="models folder (default: the GPUEnsemble home)")

    p_cat = sub.add_parser("catalog", help="update the list of models: the newest checked list from the "
                           "repository, new models found on Hugging Face, and the start of each one's file")
    p_cat.add_argument("--update", action="store_true", help="fetch and read (without it: show where the list is)")
    p_cat.add_argument("--no-discover", action="store_true", help="do not look for new models on Hugging Face")
    p_cat.add_argument("--add", metavar="REPO", help="put a model from Hugging Face on this PC's own list: "
                       "owner/name, or a link to the repository or to one of its .gguf files")
    p_cat.add_argument("--quant", help="the quantization to take with --add (default: the one in the file "
                       "link, else Q4_K_M)")
    p_cat.add_argument("--name", help="the name shown for a model added with --add")
    p_cat.add_argument("--remove", metavar="ID", help="take a model added with --add off the list "
                       "(its downloaded file is kept)")

    p_pull = sub.add_parser("pull", help="download a catalog model from Hugging Face")
    p_pull.add_argument("model_id", help="catalog id, as `models` or `recommend` lists it")
    p_list = sub.add_parser("models", help="list the catalog and what is downloaded")
    for p in (p_pull, p_list):
        p.add_argument("--dir", default=None,
                       help="models folder (default: GPUENSEMBLE_HOME\\models, else %%LOCALAPPDATA%%\\GPUEnsemble\\models "
                            "or ~/.cache/gpuensemble/models)")
        p.add_argument("--catalog", help="model catalog JSON (default: the built-in one)")

    p_setup = sub.add_parser("setup", help="download llama.cpp for this PC")
    p_setup.add_argument("--backend", choices=["auto", "cuda", "vulkan", "cpu"], default="auto",
                         help="default auto: CUDA 12.4 with a recent NVIDIA driver, else Vulkan, else CPU")
    p_setup.add_argument("--tag", default=DEFAULT_TAG,
                         help=f"llama.cpp release (default {DEFAULT_TAG}, the one measured; `latest` for the newest)")
    p_setup.add_argument("--force", action="store_true", help="download again even if it is installed")

    p_up = sub.add_parser("up", help="pick, download and start the best model for this PC")
    p_up.add_argument("--model", help="catalog id or a .gguf file (default: the recommended model)")
    p_up.add_argument("--rig", help="rig description JSON (default: detect this PC)")
    p_up.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    p_up.add_argument("--min-speed", type=float, default=MIN_SPEED,
                      help=f"tokens/s the recommended model must reach (default {MIN_SPEED:g})")
    p_up.add_argument("--kv", choices=["auto", "f16", "q8_0", "q4_0"], default="auto")
    p_up.add_argument("--ram-gib", type=float, default=None)
    p_up.add_argument("--dir", default=None, help="models folder (default: the GPUEnsemble home)")
    p_up.add_argument("--catalog", help="model catalog JSON (default: the built-in one)")
    p_up.add_argument("--exe", default=None, help="llama-server to use (default: the one `setup` installs)")
    p_up.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 for the LAN)")
    p_up.add_argument("--port", type=int, default=8080)
    p_up.add_argument("--yes", "-y", action="store_true", help="download without asking")
    p_up.add_argument("--open", action="store_true", help="open the chat page in the browser when ready")
    p_up.add_argument("--weights", choices=["auto", "send", "fetch"], default="auto",
                      help="how a PC that lends GPUs gets its share of the weights the first time: send them from "
                           "this PC, or let it fetch them from Hugging Face itself (default auto: fetch when that "
                           "PC is at another site, send on a LAN)")
    p_up.add_argument("--run", choices=["auto", "here", "there"], default="auto",
                      help="where llama-server runs when the whole model fits on one other PC: there, with its API "
                           "passed on to this PC, or here with that PC's cards over the network (default auto: "
                           "there when that PC's worker allows it and it is at least 10%% faster)")
    p_up.add_argument("--no-warmup", action="store_true",
                      help="say ready as soon as the model is loaded, without first sending a few short requests")
    p_up.add_argument("--no-hold-clocks", action="store_true",
                      help="do not hold a Vulkan device open on this PC's cards while the model runs (held when "
                           "the model is split, so that the cards keep their clocks while they wait; Windows)")
    p_up.add_argument("extra", nargs="*", help="extra arguments passed to llama-server (after --)")
    for p in (p_plan, p_args, p_run, p_rec, p_up, p_state):
        p.add_argument("--remote", action="append", metavar="FILE|auto",
                       help="GPUs on another PC: a file as `worker` prints it, or auto to find workers started "
                            "with --lan on the LAN and on this PC's Tailscale network (auto:ADDRESS,... asks "
                            "those addresses only); repeatable")

    p_mon = sub.add_parser("monitor", help="the running model's speed and each card's memory, temperature, "
                           "clocks and load (which part holds the others back)")
    p_mon.add_argument("--every", type=float, default=None, metavar="SECONDS",
                       help="read again every SECONDS until stopped (default: once)")
    p_mon.add_argument("--count", type=int, default=None, help="stop after this many readings")
    p_mon.add_argument("--json", action="store_true", help="one JSON object per reading (what the app reads)")

    p_find = sub.add_parser("find", help="list the workers on the LAN and this PC's Tailscale network")
    p_find.add_argument("--at", help="ask these addresses only, e.g. 192.168.10.2")
    p_find.add_argument("--port", type=int, default=worker.DEFAULT_PORT,
                        help=f"the workers' port (default {worker.DEFAULT_PORT}; they answer on the next one)")

    p_work = sub.add_parser("worker", help="lend this PC's GPUs to GPUEnsemble on another PC (llama.cpp RPC)")
    p_work.add_argument("--host", default=None,
                        help="address to listen on: this PC's address on the network to the main PC "
                             "(default 127.0.0.1, this PC only; with --lan, every network)")
    p_work.add_argument("--lan", action="store_true",
                        help="let main PCs on the LAN find these GPUs (up --remote auto); each one is approved "
                             "here once, by typing y, and remembered")
    p_work.add_argument("--port", type=int, default=worker.DEFAULT_PORT,
                        help=f"port to listen on (default {worker.DEFAULT_PORT})")
    p_work.add_argument("--device", help="GPUs to lend, e.g. CUDA0,CUDA1 (default: every NVIDIA GPU)")
    p_work.add_argument("--cache", action="store_true",
                        help="keep the weights the main PC sends on this PC, so later loads of the same model are fast")
    p_work.add_argument("--serve", action="store_true",
                        help="let the main PC start a whole model on this PC when it fits on these GPUs alone: this PC "
                             "then downloads the model and runs it itself, which is faster than lending the GPUs")
    p_work.add_argument("--exe", default=None, help="ggml-rpc-server to use (default: the one `setup` installs)")
    p_work.add_argument("--allow", help="addresses or networks that may use these GPUs, e.g. 192.168.10.1 or "
                                        "192.168.10.0/24 (needed with --host other than 127.0.0.1)")
    p_work.add_argument("--ask-tailscale", action="store_true",
                        help="with --lan: ask here about PCs signed in to Tailscale with this PC's account too "
                             "(default: they are let in without asking)")
    p_work.add_argument("--allow-any", action="store_true",
                        help="let anyone who can reach --host connect (the RPC server has no password)")

    args = parser.parse_args(argv)
    if args.command == "gpus":
        from .gpu_table import markdown_table

        print(markdown_table())
        return 0
    if args.command == "up":
        return _up(args)
    if args.command == "setup":
        return _setup(args)
    if args.command == "recommend":
        return _recommend(args)
    if args.command == "state":
        return _state(args)
    if args.command == "pull":
        return _pull(args)
    if args.command == "models":
        return _models(args)
    if args.command == "catalog":
        return _catalog(args)
    if args.command == "worker":
        return _worker(args)
    if args.command == "monitor":
        return _monitor(args)
    if args.command == "find":
        return _find_cmd(args)
    try:
        model, path = _named_model(args.model, needs_file=args.command == "run")
        if args.command == "describe":
            print(model.to_json())
            return 0
        devices, host = _hardware(args)
        if args.command in ("args", "run") or not args.split_experts:
            # llama.cpp keeps a layer's experts on the GPU or in RAM as a whole and cannot stream them
            # over PCIe, so plan for what it can do: `plan` then shows what `up` runs.
            host.expert_path = "cpu"
        plan = make_plan(model, devices, host, context=args.ctx, mode=args.mode, kv_type=args.kv)
        command = llama_command(plan, model, devices, tool=args.tool, threads=host.threads, ram_bytes=host.ram_bytes) if args.command in ("args", "run") else None
    except (GGUFError, PlanError, RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if command is None:
        print(format_plan(plan))
        return 0
    if args.command == "args":
        print(format_plan(plan))
        print()
        for note in command.notes:
            print(f"note: {note}")
        exe = args.exe or ("llama-bench" if args.tool == "bench" else "llama-server")
        print(command.shell(exe, windows=args.windows))
        return 0
    if path is None:
        print("error: run needs the .gguf file, not a profile", file=sys.stderr)
        return 1
    args.exe = args.exe or _default_exe(args.tool)
    if args.exe is None:
        print(f"error: no llama.cpp found; run `{PROG} setup` or pass --exe", file=sys.stderr)
        return 1
    print(format_plan(plan), file=sys.stderr)
    if not _rpc_servers_up(command, lambda s: print(s, file=sys.stderr, flush=True)):
        return 1
    print("starting: " + command.shell(args.exe, model=path, extra=args.extra), file=sys.stderr)
    return run(args.exe, command, path, args.extra,
               ready_timeout=READY_TIMEOUT + remote_load_seconds(plan, devices))

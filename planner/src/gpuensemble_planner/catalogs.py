"""Where the list of models comes from, and how it stays new.

Four sources, merged by ``load_catalog``:

1. ``catalog.json`` next to this file: the list checked by hand that ships in the package.
2. ``<home>/catalog.json``: the newest copy of that list, fetched from the repository by
   ``catalog --update``. Whichever of 1 and 2 has the later ``updated`` date is used, so
   models added to the repository reach every PC without a new install.
3. ``<home>/catalog-new.json``: models found on Hugging Face by ``catalog --update``:
   recent, popular GGUF repositories of the well-known quantizers that have a ``Q4_K_M``
   file. They are listed after the checked ones, marked ``"new": true`` (not checked by
   hand), ordered by downloads.
4. ``<home>/catalog-mine.json``: models the user added with ``catalog --add`` from any GGUF
   repository on Hugging Face (a fine-tune, say, that no list carries). Only that PC lists them;
   they come after the checked ones and before the found ones, marked ``"mine": true``.

``catalog --update`` also reads, for every model that has no profile written in the list,
the header of its file on Hugging Face (``remote.profile_of``), so planning needs no network.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .hub import HF, _request, home_dir, list_repo, select_gguf
from .remote import profile_of

BUNDLED = os.path.join(os.path.dirname(__file__), "catalog.json")
CATALOG_URL = ("https://raw.githubusercontent.com/mogurasoken-glitch/GPUEnsemble/main/"
               "planner/src/gpuensemble_planner/catalog.json")
# Quantizers whose GGUF files are looked for, in order of preference when several have one model.
QUANTIZERS = ("ggml-org", "unsloth", "bartowski", "lmstudio-community")
NEW_QUANT = "Q4_K_M"
NEW_MAX = 12  # models taken from Hugging Face at most
NEW_DAYS = 180  # only repositories created within this many days
NEW_MIN_DOWNLOADS = 2000
NEW_RANK = 1000  # found models rank after every checked one
MINE_RANK = 500  # models the user added rank after the checked ones and before the found ones
READ_WORKERS = 6  # files on Hugging Face read at the same time
# What Hugging Face says a chat model does (its "pipeline_tag"); a repository that says nothing is
# kept here and judged by its header (looks_like_llm).
TEXT_TASKS = (None, "", "text-generation", "image-text-to-text", "conversational")
# Not chat models, or changed by someone other than the maker.
_SKIP = re.compile(r"embed|rerank|draft|mtp|mmproj|abliterat|uncensor|heretic|lora|tts|whisper|ocr|"
                   r"guard|reward|base\b|-base-|-pt-|"
                   r"image|video|audio|speech|diffusion|flux|sdxl|music", re.IGNORECASE)  # not chat models


def fetched_path(home: str | None = None) -> str:
    return os.path.join(home or home_dir(), "catalog.json")


def found_path(home: str | None = None) -> str:
    return os.path.join(home or home_dir(), "catalog-new.json")


def mine_path(home: str | None = None) -> str:
    return os.path.join(home or home_dir(), "catalog-mine.json")


def _read(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        data = {"models": data}
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return None
    return data


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def checked_list(home: str | None = None) -> dict:
    """The checked list: the bundled one, or the fetched copy when that is newer."""
    bundled = _read(BUNDLED) or {"models": []}
    fetched = _read(fetched_path(home))
    if fetched and str(fetched.get("updated", "")) > str(bundled.get("updated", "")):
        return fetched
    return bundled


def model_key(repo: str) -> str:
    """One key for the same model from different quantizers: ``unsloth/gemma-4-31B-it-GGUF``,
    ``bartowski/google_gemma-4-31B-it-GGUF`` and ``ggml-org/gemma-4-31b-it-GGUF`` give the same."""
    name = repo.rsplit("/", 1)[-1].lower()
    if repo.lower().startswith("bartowski/") and "_" in name:
        name = name.split("_", 1)[1]  # bartowski puts the maker in front
    name = re.sub(r"-gguf$", "", name)
    name = re.sub(r"-(it|instruct|chat)(?=-|$)", "", name)
    return name


def _source(e: dict) -> tuple[str, str]:
    return str(e.get("hf_repo", "")).lower(), str(e.get("hf_quant", "")).upper()


def mine(home: str | None = None) -> list[dict]:
    """The models the user added on this PC (``catalog --add``)."""
    data = _read(mine_path(home)) or {"models": []}
    return [e for e in data["models"]
            if isinstance(e, dict) and e.get("id") and e.get("hf_repo") and e.get("hf_quant")]


def load_all(home: str | None = None) -> list[dict]:
    """Checked models, then the ones the user added, then the ones found on Hugging Face that are
    not among them."""
    entries = [dict(e) for e in checked_list(home)["models"]]
    ids = {e["id"] for e in entries}
    sources = {_source(e) for e in entries}
    for e in mine(home):
        if e["id"] in ids or _source(e) in sources:
            continue  # the checked list has it now
        ids.add(e["id"])
        entries.append({**e, "mine": True})
    known = {model_key(e["hf_repo"]) for e in entries if e.get("hf_repo")}
    found = _read(found_path(home)) or {"models": []}
    for e in found["models"]:
        if not isinstance(e, dict) or not e.get("hf_repo") or e.get("id") in ids:
            continue
        if model_key(e["hf_repo"]) in known:
            continue
        known.add(model_key(e["hf_repo"]))
        entries.append({**e, "new": True})
    return entries


def fetch_checked(home: str | None = None, url: str = CATALOG_URL,
                  opener: Callable = urllib.request.urlopen) -> str:
    """Fetch the newest checked list from the repository; say what happened."""
    try:
        with opener(_request(url), timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return f"the list in the repository could not be read (HTTP {e.code}); using the one installed"
    except (OSError, ValueError) as e:
        return f"the list in the repository could not be read ({e}); using the one installed"
    if not isinstance(data, dict) or not isinstance(data.get("models"), list) or not all(
            isinstance(m, dict) and m.get("id") for m in data["models"]):
        return "the list in the repository is not a model list; using the one installed"
    _write(fetched_path(home), data)
    now = checked_list(home)
    return f"list of {len(now['models'])} checked models, updated {now.get('updated', '?')}"


def search(author: str, opener: Callable = urllib.request.urlopen, limit: int = 100) -> list[dict]:
    """The newest GGUF repositories of one quantizer: id, downloads, likes, createdAt."""
    q = urllib.parse.urlencode({"author": author, "search": "GGUF", "sort": "createdAt",
                                "direction": "-1", "limit": str(limit)})
    with opener(_request(f"{HF}/api/models?{q}"), timeout=30) as r:
        return [m for m in json.load(r) if isinstance(m, dict) and m.get("id")]


def _name(repo: str) -> str:
    name = repo.rsplit("/", 1)[-1]
    if repo.lower().startswith("bartowski/") and "_" in name:
        name = name.split("_", 1)[1]
    return re.sub(r"-GGUF$", "", name, flags=re.IGNORECASE)


# Model types (GGUF general.architecture) of the checked list, whose files were read on 2026-11-16 and
# which the llama.cpp release GPUEnsemble installs (llamabin.DEFAULT_TAG) was built after, plus older
# common ones. A model found on Hugging Face of another type may need a newer llama.cpp, or the
# planner may misread its layers (as it did Qwen3.8's at first), so it is left out until a checked
# model of that type is added.
KNOWN_ARCHITECTURES = frozenset({
    "llama", "qwen2", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next", "gpt-oss",
    "gemma3", "gemma4", "mistral3", "mistral4", "glm4moe", "granite", "olmo2",
    "nemotron_h", "nemotron_h_moe", "phi3",
    "deepseek4", "glm5-next", "minimax-m3", "mimo2", "hy_v3", "qwen4exp", "step35", "llama4",
})


def looks_like_llm(profile) -> bool:
    """A model llama.cpp runs as a chat model: layers with a KV cache or a recurrent state.
    An image or video model's GGUF file has neither (and no blocks the reader knows)."""
    return profile.architecture != "unknown" and profile.n_layers > 0 and any(
        layer.kv_bytes_per_token > 0 or layer.state_bytes > 0 for layer in profile.layers)


def find_new(checked: list[dict], opener: Callable = urllib.request.urlopen,
             today: _dt.date | None = None, max_new: int = 2 * NEW_MAX) -> tuple[list[dict], list[str]]:
    """Recent popular GGUF models on Hugging Face that the checked list does not have."""
    today = today or _dt.date.today()
    since = (today - _dt.timedelta(days=NEW_DAYS)).isoformat()
    known = {model_key(e["hf_repo"]) for e in checked if e.get("hf_repo")}
    best: dict[str, tuple[int, int, dict]] = {}  # model key -> (quantizer order, -downloads, repo)
    notes = []
    for order, author in enumerate(QUANTIZERS):
        try:
            repos = search(author, opener)
        except (OSError, ValueError) as e:
            notes.append(f"{author}: not searched ({e})")
            continue
        for m in repos:
            repo = m["id"]
            if str(m.get("createdAt", ""))[:10] < since or _SKIP.search(repo):
                continue
            if int(m.get("downloads") or 0) < NEW_MIN_DOWNLOADS:
                continue
            if m.get("pipeline_tag") not in TEXT_TASKS:
                continue  # image, video or speech models also come as GGUF files
            key = model_key(repo)
            if key in known:
                continue
            item = (order, -int(m.get("downloads") or 0), m)
            if key not in best or item[:2] < best[key][:2]:
                best[key] = item
    # The most downloaded models first; each one's file list says whether it has the quantization.
    ranked = sorted(best.values(), key=lambda t: t[1])
    out = []
    for _, minus_downloads, m in ranked:
        if len(out) >= max_new:
            break
        repo = m["id"]
        try:
            parts = select_gguf(list_repo(repo, opener=opener), NEW_QUANT)
        except (LookupError, OSError, ValueError):
            continue
        name = _name(repo)
        out.append({
            "id": _slug(name),
            "name": f"{name} {NEW_QUANT}",
            "rank": NEW_RANK + len(out),
            "file_gib": round(sum(p.size for p in parts) / 1024**3, 2),
            "hf_repo": repo,
            "hf_quant": NEW_QUANT,
            "downloads": -minus_downloads,
            "created": str(m.get("createdAt", ""))[:10],
        })
    return out, notes


def update(home: str | None = None, opener: Callable = urllib.request.urlopen, url: str = CATALOG_URL,
           discover: bool = True, say: Callable[[str], None] = print, workers: int = READ_WORKERS) -> int:
    """``catalog --update``: newest checked list, models found on Hugging Face, and the profiles they need.

    The first run reads the start of some fifty files; they are read ``workers`` at a time, and a line
    ``read N/M`` is said after each one, so the app can show how far it is. Each profile is kept as soon
    as it is read, so a run that is stopped goes on from there the next time.
    """
    say(fetch_checked(home, url, opener))
    profiles = os.path.join(home or home_dir(), "profiles")
    errors: dict[str, str] = {}

    def read(e: dict):
        try:
            return profile_of(e, base=profiles, opener=opener)
        except Exception as err:  # one model's odd file must not stop the others
            errors[e["id"]] = f"{type(err).__name__}: {err}"
            return None

    found: list[dict] = []
    notes: list[str] = []
    if discover:
        found, notes = find_new(checked_list(home)["models"] + mine(home), opener)
        for n in notes:
            say(n)
    todo = found + [e for e in checked_list(home)["models"] + mine(home) if not e.get("profile") and e.get("hf_repo")]
    profiles_read: dict[int, object] = {}
    if todo:
        say(f"read 0/{len(todo)}")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(read, e): i for i, e in enumerate(todo)}
            for done, f in enumerate(as_completed(futures), 1):
                profiles_read[futures[f]] = f.result()
                say(f"read {done}/{len(todo)}")
    for i, e in enumerate(todo):
        if e["id"] in errors:
            say(f"{e['id']}: could not read its file on Hugging Face ({errors[e['id']]})")
    if discover:
        kept = []
        for i, e in enumerate(found):
            if len(kept) >= NEW_MAX:
                break
            prof = profiles_read.get(i)
            if prof is None:
                continue
            if not looks_like_llm(prof):
                say(f"{e['id']}: not a chat model llama.cpp runs ({prof.architecture}); left out")
                continue
            if prof.architecture not in KNOWN_ARCHITECTURES:
                say(f"{e['id']}: a model type not checked yet ({prof.architecture}); left out")
                continue
            kept.append({**e, "rank": NEW_RANK + len(kept)})
        if kept or not notes:
            _write(found_path(home), {"updated": _dt.date.today().isoformat(), "models": kept})
        say(f"{len(kept)} new models found on Hugging Face" + "".join(f"\n  {e['name']}" for e in kept))
    failed = sum(1 for i in range(len(found), len(todo)) if profiles_read.get(i) is None)
    return 1 if failed else 0


# A quantization's tag in a file name: Q4_K_M, IQ4_XS, UD-Q4_K_XL, Q8_0, MXFP4, BF16 ...
_QUANT_TAG = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?I?Q\d(?:_[A-Za-z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32)(?![A-Za-z0-9])",
                        re.IGNORECASE)
_REPO_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_source(text: str, quant: str | None = None) -> tuple[str, str]:
    """The repository and quantization in what the user gave: ``owner/name``, or a link to the
    repository or to one of its ``.gguf`` files, whose name then gives the quantization.
    Without ``quant`` or a file, ``Q4_K_M``."""
    s = re.sub(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", "", text.strip(), flags=re.IGNORECASE)
    s = re.split(r"[?#]", s, maxsplit=1)[0].strip("/")
    parts = s.split("/")
    if len(parts) < 2 or not all(_REPO_PART.fullmatch(p) for p in parts[:2]):
        raise ValueError(f"{text.strip()!r} is not a Hugging Face repository (owner/name) or a link to one")
    if not quant and len(parts) > 2 and parts[-1].lower().endswith(".gguf"):
        tags = _QUANT_TAG.findall(parts[-1][:-len(".gguf")])
        quant = tags[-1] if tags else None
    return f"{parts[0]}/{parts[1]}", (quant or NEW_QUANT).strip().upper()


def _maker(repo: str) -> str:
    owner, name = repo.split("/", 1)
    if owner.lower() == "bartowski" and "_" in name:
        return name.split("_", 1)[0]  # bartowski puts the maker in front
    return owner


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def add_mine(source: str, quant: str | None = None, name: str | None = None, home: str | None = None,
             token: str | None = None, opener: Callable = urllib.request.urlopen) -> tuple[dict, list[str]]:
    """``catalog --add``: put a model from a GGUF repository on Hugging Face on this PC's own list.

    Reads the repository's file list and the start of the file (a few MB), so the model can be
    planned and downloaded like a checked one. Adding the same repository and quantization again
    replaces it. Returns the entry and notes for the user; raises LookupError or ValueError (or
    OSError) when the model cannot be added.
    """
    repo, quant = parse_source(source, quant)
    token = token or os.environ.get("HF_TOKEN") or None
    checked = checked_list(home)["models"]
    for e in checked:
        if _source(e) == (repo.lower(), quant):
            raise ValueError(f"{repo} {quant} is already in the list, as {e['id']}")
    parts = select_gguf(list_repo(repo, token, opener), quant)
    others = [e for e in mine(home) if _source(e) != (repo.lower(), quant)]
    same = [e for e in mine(home) if _source(e) == (repo.lower(), quant)]
    label = _name(repo)
    if same:
        model_id = same[0]["id"]
    else:
        taken = {e["id"] for e in checked} | {e["id"] for e in others}
        tries = [_slug(label), _slug(f"{label}-{_maker(repo)}"), _slug(f"{label}-{_maker(repo)}-{quant}")]
        tries += [f"{tries[-1]}-{n}" for n in range(2, len(taken) + 3)]
        model_id = next(t for t in tries if t not in taken)
    entry = {
        "id": model_id,
        # The maker in the name: two people's versions of one model are told apart in the app.
        "name": name or f"{label} {quant} ({_maker(repo)})",
        "rank": MINE_RANK,
        "file_gib": round(sum(p.size for p in parts) / 1024**3, 2),
        "hf_repo": repo,
        "hf_quant": quant,
        "added": _dt.date.today().isoformat(),
    }
    try:
        prof = profile_of(entry, base=os.path.join(home or home_dir(), "profiles"), token=token, opener=opener)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise LookupError(f"{repo}: Hugging Face asks for a login to download it: accept its terms on "
                              f"its page and set HF_TOKEN to a token of that account (HTTP {e.code})") from e
        raise
    if not looks_like_llm(prof):
        raise ValueError(f"{repo} {quant}: not a chat model llama.cpp runs ({prof.architecture})")
    notes = []
    if prof.architecture not in KNOWN_ARCHITECTURES:
        notes.append(f"its model type ({prof.architecture}) has not been tried with GPUEnsemble; the "
                     f"llama.cpp it installs may not run it, and the estimates may be off")
    models = others + [entry]
    for i, e in enumerate(models):
        e["rank"] = MINE_RANK + i
    _write(mine_path(home), {"models": models})
    return entry, notes


def remove_mine(model_id: str, home: str | None = None) -> bool:
    """``catalog --remove``: take a model the user added off this PC's list. Its downloaded file is kept."""
    models = mine(home)
    kept = [e for e in models if e["id"] != model_id]
    if len(kept) == len(models):
        return False
    for i, e in enumerate(kept):
        e["rank"] = MINE_RANK + i
    _write(mine_path(home), {"models": kept})
    return True

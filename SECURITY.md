# Security

## What to know before lending GPUs to another PC

`gpuensemble worker` (and `Start GPUEnsemble worker.cmd`) runs llama.cpp's RPC server so that another PC can use this PC's GPUs.

- The RPC server has **no password and no encryption**. Whoever reaches it can use the GPUs and read or change their memory.
- GPUEnsemble therefore keeps the RPC server on 127.0.0.1 and puts a small gate in front of it. The gate lets in only the addresses you approve (`--allow`, or `y` in the worker's window with `--lan`). It tells PCs apart by IP address alone.
- With `--lan`, a PC that is signed in to Tailscale with the same account as the worker is let in without asking (`--ask-tailscale` turns that off).
- A PC that is let in can also ask a worker started with `--cache` to download model files into its cache folder, from huggingface.co only.
- A PC that is let in can also ask a worker started with `--serve` (the double-click worker is) to run a whole model itself. The request names a model of the worker's own catalog and a context size, and nothing else: no file, no address to download from, no argument for llama-server. The worker downloads the model from the place its catalog names if it does not have it (tens of GiB for the larger ones), runs llama-server on 127.0.0.1, and opens one more port (50054 by default) whose gate passes only the PC that asked on to it. One model runs at a time, and it stops when that PC closes its connection or stops answering.
- Use it only on networks you trust: a home LAN, a cable between two PCs, or a VPN such as Tailscale. Do not expose the worker's ports (50052 to 50054 by default) to the internet.
- If Windows asks whether Python may accept connections, allow private networks only.

The server that `up` starts (llama-server) listens on 127.0.0.1 unless you pass `--host`. When the model runs on another PC (see `--serve` above), `up` listens there in its place and passes each connection on to that PC; like the RPC, that link is not encrypted by GPUEnsemble, and the PC that runs the model reads the prompts and writes the answers, so the same advice about networks applies.

## Reporting a problem

Please report security problems privately, through **Security → Report a vulnerability** on the GitHub repository, rather than in a public issue. Problems in llama.cpp itself are best reported to [llama.cpp](https://github.com/ggml-org/llama.cpp/security).

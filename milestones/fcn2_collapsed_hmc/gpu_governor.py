#!/usr/bin/env python3
"""
Thermal governor for GPU runs: pauses (SIGSTOP) the watched processes on a GPU when it reaches --hot °C and resumes
them (SIGCONT) when it has cooled to --cool °C. Pausing a sampler only stretches wall time; the Markov chain itself
is unchanged. Each watched process is assigned to the GPU where it holds the most memory (a chain that drew its
data on another GPU keeps a small idle context there). Exits when every watched process has ended, and on any exit
(Ctrl-C, SIGTERM, error) resumes whatever it paused. Every pause/resume is logged with the temperatures.
Needs only nvidia-smi and permission to signal the processes (same user), no sudo.

Temperatures are read through NVML (libnvidia-ml via ctypes) every --poll seconds: under full load the die heats
~20 °C per second, so a slow poll overshoots the threshold by 10 °C. Processes already stopped at start (e.g. by a
previous governor) are adopted as paused. A status line with the peak temperature per GPU is logged every minute.

    python gpu_governor.py --pids 1234 5678                 # defaults: --hot 72 --cool 64 --poll 0.1
    python gpu_governor.py --pids 1234 --hot 70 --cool 62 --log runs/logs/governor.log
"""

import argparse
import os
import signal
import subprocess
import sys
import time


class NVML:
    """Temperatures straight from libnvidia-ml (well under 1 ms per reading; nvidia-smi takes ~100 ms), keyed by UUID."""

    def __init__(self):
        import ctypes
        self.c = ctypes
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        n = ctypes.c_uint()
        self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        self.handles = {}
        for i in range(n.value):
            h = ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
            buf = ctypes.create_string_buffer(96)
            self.lib.nvmlDeviceGetUUID(h, buf, 96)
            self.handles[buf.value.decode()] = h

    def temp(self, uuid):
        t = self.c.c_uint()
        self.lib.nvmlDeviceGetTemperature(self.handles[uuid], 0, self.c.byref(t))   # 0 = NVML_TEMPERATURE_GPU
        return int(t.value)


def query(args):
    out = subprocess.run(["nvidia-smi", f"--query-{args}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_state():
    """{uuid: (index, name, temperature)} and {pid: [(uuid, MiB), ...]}."""
    gpus = {u: (int(i), n, int(t)) for i, u, n, t in query("gpu=index,uuid,name,temperature.gpu")}
    procs = {}
    for pid, u, mem in query("compute-apps=pid,gpu_uuid,used_memory"):
        procs.setdefault(int(pid), []).append((u, float(mem) if mem.replace(".", "").isdigit() else 0.0))
    # a paused process can drop out of the compute-apps list on some drivers; keep the GPU it was last seen on
    return gpus, procs


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", type=int, nargs="+", required=True)
    ap.add_argument("--hot", type=float, default=72.0, help="pause at or above this temperature (°C)")
    ap.add_argument("--cool", type=float, default=64.0, help="resume at or below this temperature (°C)")
    ap.add_argument("--poll", type=float, default=0.1,
                    help="seconds between temperature checks (under load the die heats ~20 °C/s, so keep this short)")
    ap.add_argument("--log", default=None, help="also append the log to this file")
    args = ap.parse_args()
    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    home, paused = {}, set()                     # pid -> gpu uuid; pids currently stopped by us
    paused_time, t0 = {p: 0.0 for p in args.pids}, time.time()
    since = {}

    def resume_all(reason):
        for p in list(paused):
            if alive(p):
                os.kill(p, signal.SIGCONT)
            paused_time[p] += time.time() - since.pop(p, time.time())
            paused.discard(p)
            log(f"resumed pid {p} ({reason})")

    def on_signal(sig, _):
        resume_all(f"governor got signal {sig}")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        nvml = NVML()
    except Exception as e:                       # fall back to nvidia-smi for the temperatures (slow)
        nvml = None
        log(f"NVML unavailable ({e}); reading temperatures with nvidia-smi")
    for p in args.pids:                          # adopt processes that are already stopped (e.g. by a previous governor)
        if alive(p) and open(f"/proc/{p}/stat").read().split()[2] == "T":
            paused.add(p); since[p] = time.time()
            log(f"pid {p} is already stopped; treating it as paused")
    log(f"watching pids {args.pids}: pause at >= {args.hot} °C, resume at <= {args.cool} °C, poll {args.poll} s")
    gpus, last_map, peak, last_status = {}, 0.0, {}, time.time()
    try:
        while True:
            live = [p for p in args.pids if alive(p)]
            if not live:
                break
            if any(p not in home for p in live) or time.time() - last_map > 10:
                # process -> GPU map via nvidia-smi (slow, so not every poll). Re-evaluated every time, not only once:
                # a chain draws its data on cuda:0 before moving to its own GPU, so its largest allocation is on the
                # wrong GPU for the first seconds (on 2026-10-29 this left the 4080 unwatched for hours).
                gpus, procs = gpu_state()
                last_map = time.time()
                for p in live:
                    if procs.get(p):
                        u = max(procs[p], key=lambda x: x[1])[0]
                        if home.get(p) != u:
                            home[p] = u
                            i, n, t = gpus[u]
                            log(f"pid {p} -> GPU {i} ({n}), now {t} °C")
            elif nvml:
                gpus = {u: (i, n, nvml.temp(u)) for u, (i, n, _) in gpus.items()}
            else:
                gpus, _ = gpu_state()
            temps = " ".join(f"GPU{i}={t}°C" for i, _, t in sorted(gpus.values()))
            for i, _, t in gpus.values():
                peak[i] = max(peak.get(i, 0), t)
            if time.time() - last_status > 60:
                log("status: peak over the last minute " + " ".join(f"GPU{i}={t}°C" for i, t in sorted(peak.items()))
                    + f"; paused now: {sorted(paused)}")
                peak.clear(); last_status = time.time()
            for u, (i, n, t) in gpus.items():
                mine = [p for p in live if home.get(p) == u]
                if t >= args.hot and any(p not in paused for p in mine):
                    for p in mine:
                        if p not in paused:
                            os.kill(p, signal.SIGSTOP); paused.add(p); since[p] = time.time()
                    log(f"PAUSE  GPU {i} ({n}) at {t} °C: pids {mine}   [{temps}]")
                elif t <= args.cool and any(p in paused for p in mine):
                    for p in mine:
                        if p in paused:
                            os.kill(p, signal.SIGCONT); paused.discard(p)
                            paused_time[p] += time.time() - since.pop(p)
                    log(f"RESUME GPU {i} ({n}) at {t} °C: pids {mine}   [{temps}]")
            time.sleep(args.poll)
    finally:
        resume_all("governor exiting")
        wall = time.time() - t0
        log("done; time paused per pid: " + ", ".join(f"{p}: {paused_time[p] / 60:.1f} of {wall / 60:.1f} min"
                                                    for p in args.pids))


if __name__ == "__main__":
    main()

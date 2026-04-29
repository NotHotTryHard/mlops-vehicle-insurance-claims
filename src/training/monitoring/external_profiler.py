import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def profiler_settings(cfg):
    defaults = {
        "enabled": False,
        "output_dir": "session/reports/profiles",
        "time": True,
        "memory": True,
        "memray_native_traces": True,
        "time_sample_interval": 0.001,
    }
    merged = dict(defaults)
    merged.update(cfg["training"]["profiler"])
    return merged


def _memray_flamegraph(bin_path, html_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "memray",
            "flamegraph",
            "-o",
            str(html_path),
            str(bin_path),
        ],
        check=True,
    )


def _append_profiler_manifest(out_dir, project_root, entry):
    path = out_dir / "manifest.yaml"
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(entry)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)


def run_profiled(cfg, project_root, mode_label, fn, logger):
    settings = profiler_settings(cfg)
    if not settings["enabled"]:
        return fn()

    use_time = settings["time"]
    use_memory = settings["memory"]
    if not use_time and not use_memory:
        return fn()

    out_dir = project_root / settings["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{mode_label}_{stamp}"
    run_dir = out_dir / base
    run_dir.mkdir(parents=True, exist_ok=True)

    interval = float(settings["time_sample_interval"])
    native_traces = settings["memray_native_traces"]

    if use_time:
        from pyinstrument import Profiler

        ProfilerClass = Profiler
    else:
        ProfilerClass = None
    if use_memory:
        from memray import Tracker

        TrackerClass = Tracker
    else:
        TrackerClass = None

    pyi_html = run_dir / "time.html"
    bin_path = run_dir / "memray.bin"
    memray_html = run_dir / "memory_flamegraph.html"
    rel_run = str(run_dir.relative_to(project_root)).replace("\\", "/")

    def _manifest(time_ok, memory_ok, memory_bin_ok):
        _append_profiler_manifest(
            out_dir,
            project_root,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": mode_label,
                "run_dir": rel_run,
                "time_report": time_ok,
                "memory_flamegraph": memory_ok,
                "memory_capture": memory_bin_ok,
            },
        )

    # memray is outside, pyinstrument inside — otherwise conflict threading.setprofile on exit.
    if use_time and use_memory:
        with TrackerClass(str(bin_path), native_traces=native_traces):
            with ProfilerClass(interval=interval) as profiler:
                result = fn()
            profiler.write_html(str(pyi_html))
        logger.info("profiler time path=%s", pyi_html)
        print(f"[profiler] time -> {pyi_html}", flush=True)
        mem_ok = False
        try:
            _memray_flamegraph(bin_path, memray_html)
            mem_ok = True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("memray flamegraph failed: %s (binary at %s)", exc, bin_path)
            print(f"[profiler] memory flamegraph failed ({exc}); binary: {bin_path}", flush=True)
        else:
            logger.info("profiler memory path=%s", memray_html)
            print(f"[profiler] memory -> {memray_html}", flush=True)
            print(f"[profiler] memory capture -> {bin_path}", flush=True)
        _manifest(True, mem_ok, bin_path.is_file())
        print(f"[profiler] run dir -> {run_dir}", flush=True)
        return result

    if use_memory:
        with TrackerClass(str(bin_path), native_traces=native_traces):
            result = fn()
        mem_ok = False
        try:
            _memray_flamegraph(bin_path, memray_html)
            mem_ok = True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("memray flamegraph failed: %s (binary at %s)", exc, bin_path)
            print(f"[profiler] memory flamegraph failed ({exc}); binary: {bin_path}", flush=True)
        else:
            logger.info("profiler memory path=%s", memray_html)
            print(f"[profiler] memory -> {memray_html}", flush=True)
            print(f"[profiler] memory capture -> {bin_path}", flush=True)
        _manifest(False, mem_ok, bin_path.is_file())
        print(f"[profiler] run dir -> {run_dir}", flush=True)
        return result

    with ProfilerClass(interval=interval) as profiler:
        result = fn()
    profiler.write_html(str(pyi_html))
    logger.info("profiler time path=%s", pyi_html)
    print(f"[profiler] time -> {pyi_html}", flush=True)
    _manifest(True, False, False)
    print(f"[profiler] run dir -> {run_dir}", flush=True)
    return result

"""Codex CLI command line, session-record reading and the quota guard of the end-to-end experiment.

The command line is the one of the TACS SMAA batches (tacs_vendor/elicit.py: model gpt-6.1-sol, reasoning effort xhigh,
service tier "default", web search off, memories, hooks, multi-agent, plugins and goals disabled, read-only sandbox in an
empty working folder, output schema), plus the isolation of the October 2026 replication (experiments/replication_2026-10/
scripts/codex_turn.py): the user's config.toml is not loaded, and the environment, permission, apps and collaboration-mode
blocks and project documents are left out. `--cmdline tacs` gives the TACS command line unchanged.

Rules of the author (binding, 21 and 26 October 2026): only gpt-6.1-sol at xhigh on the default (normal-speed) tier,
never Fast/priority, never paid credits. The guard stops a batch when the credit balance changes, when the next call could
take the 5-hour window past 95 %, when the weekly window reaches 90 %, when a record reports a reached rate limit or spend
control or a change of has_credits, or when a session record shows another model, effort or tier (or none), a loaded
global AGENTS.md or a tool call. Before the first call, run_batch.py reads the newest session record (latest_reading)
for the credit baseline and the current 5-hour reading (protocol review of 26.10.2026, required change 9b).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

CODEX = "codex"  # path to the Codex command line tool
SESSIONS = Path.home() / ".codex" / "sessions"
GLOBAL_AGENTS = Path.home() / ".codex" / "AGENTS.md"
PAUSED_AGENTS = GLOBAL_AGENTS.with_name("AGENTS.md.paused-by-claude-endtoend-batch")

MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"
TIER = "default"
# Credit balance in every session record so far (TACS batches of 21.10.2026, replication of 25.10.2026). A batch does
# not start when the newest record shows another balance; the author decides (run_batch.py --expected-credits).
EXPECTED_CREDITS = "62437.5049350000"


def build_cmd(schema: Path, out_file: Path, workdir: Path, cmdline: str = "isolated") -> list[str]:
    cmd = [CODEX, "exec", "-m", MODEL,
           "-c", f'model_reasoning_effort="{EFFORT}"', "-c", f'service_tier="{TIER}"',
           "-c", 'web_search="disabled"', "-c", "notify=[]"]
    if cmdline == "isolated":
        cmd += ["-c", "include_environment_context=false", "-c", "include_permissions_instructions=false",
                "-c", "include_apps_instructions=false", "-c", "include_collaboration_mode_instructions=false",
                "-c", "project_doc_max_bytes=0"]
    elif cmdline != "tacs":
        raise ValueError(cmdline)
    cmd += ["--disable", "memories", "--disable", "hooks", "--disable", "multi_agent",
            "--disable", "multi_agent_v2", "--disable", "plugins", "--disable", "remote_plugin",
            "--disable", "goals"]
    if cmdline == "isolated":
        cmd += ["--ignore-user-config"]
    cmd += ["-s", "read-only", "--skip-git-repo-check", "-C", str(workdir),
            "--output-schema", str(schema), "-o", str(out_file), "--json", "-"]
    return cmd


def find_rollout(session_id: str) -> Path | None:
    hits = sorted(SESSIONS.rglob(f"*{session_id}*.jsonl"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def read_rollout(path: Path) -> dict:
    """Rate limits, settings, AGENTS.md and tool calls recorded in a session file (after codex_turn.read_rollout)."""
    info = {"rate_limits": None, "service_tier": None, "model": None, "effort": None, "personality": None,
            "agents_md_loaded": False, "tool_calls": [], "n_token_count": 0, "token_info": None}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("payload") or {}
        if p.get("type") == "token_count":
            info["n_token_count"] += 1
            if p.get("rate_limits"):
                info["rate_limits"] = p["rate_limits"]
            if p.get("info"):
                info["token_info"] = p["info"]
        if p.get("type") == "thread_settings_applied":
            ts = p.get("thread_settings") or {}
            info["service_tier"] = ts.get("service_tier")
            info["model"] = ts.get("model")
            info["effort"] = ts.get("reasoning_effort") or ts.get("effort") or info["effort"]
            info["personality"] = ts.get("personality")
        if r.get("type") == "turn_context":
            info["effort"] = p.get("effort") or info["effort"]
            info["model"] = p.get("model") or info["model"]
        if r.get("type") == "world_state":
            am = ((p.get("state") or {}).get("agents_md")) or {}
            if am.get("text"):
                info["agents_md_loaded"] = True
        if r.get("type") == "response_item":
            if p.get("type") == "message":
                for c in p.get("content") or []:
                    if "AGENTS.md instructions" in (c.get("text") or ""):
                        info["agents_md_loaded"] = True
            elif p.get("type") not in ("reasoning",):
                info["tool_calls"].append(p.get("type"))
        if r.get("type") == "event_msg" and str(p.get("type", "")).endswith(("_begin",)) and \
                any(k in p.get("type", "") for k in ("exec", "patch", "mcp", "web_search", "tool")):
            info["tool_calls"].append(p.get("type"))
    return info


def copy_rollout(session_id: str | None, dest: Path) -> dict:
    if not session_id:
        return {}
    src = find_rollout(session_id)
    if not src:
        return {}
    shutil.copy2(src, dest)
    return read_rollout(dest)


def last_rate_limits(path: Path) -> dict | None:
    """Rate limits of the last token_count event of one session record, or None."""
    rl = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"token_count"' not in line:
            continue
        try:
            p = json.loads(line).get("payload") or {}
        except json.JSONDecodeError:
            continue
        if p.get("type") == "token_count" and p.get("rate_limits"):
            rl = p["rate_limits"]
    return rl


def latest_reading(max_files: int = 50) -> dict | None:
    """The newest session record (by modification time) that holds rate limits: path, time and rate limits.
    Used before the first call of a batch, so the credit baseline and the 5-hour reading do not come from the batch's
    own first call."""
    files = sorted(SESSIONS.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:max_files]:
        rl = last_rate_limits(f)
        if rl:
            return {"file": str(f), "modified": datetime.fromtimestamp(f.stat().st_mtime).astimezone().isoformat(
                timespec="seconds"), "rate_limits": rl}
    return None


def preflight(reading: dict | None, expected_credits: str, max_primary: float, max_secondary: float,
              step: float, now: float | None = None) -> str | None:
    """A reason not to start the batch, from the newest session record, or None."""
    if not reading:
        return "no session record with rate limits found in ~/.codex/sessions"
    now = time.time() if now is None else now
    rl = reading["rate_limits"]
    credits = (rl.get("credits") or {}).get("balance")
    if credits != expected_credits:
        return (f"credit balance in the newest record is {credits}, expected {expected_credits}; ask the author "
                f"(run_batch.py --expected-credits after the author's decision)")
    prim, sec = rl.get("primary") or {}, rl.get("secondary") or {}
    live = lambda w: (w.get("resets_at") or 0) > now                                   # noqa: E731
    if prim.get("used_percent") is not None and live(prim) and prim["used_percent"] + 2 * step >= max_primary:
        return (f"5-hour window at {prim['used_percent']}% (+2 x {step:.1f} >= {max_primary}%) until "
                f"{datetime.fromtimestamp(prim['resets_at']).astimezone().isoformat(timespec='minutes')}")
    if sec.get("used_percent") is not None and live(sec) and sec["used_percent"] >= max_secondary:
        return f"weekly window at {sec['used_percent']}%"
    if (rl.get("rate_limit_reached_type") is not None or rl.get("spend_control_reached") is not None) and \
            (live(prim) or live(sec)):
        return (f"newest record reports rate_limit_reached_type {rl.get('rate_limit_reached_type')}, "
                f"spend_control_reached {rl.get('spend_control_reached')}")
    return None


def binary_info() -> dict:
    """SHA-256 and version string of the Codex binary (no model call)."""
    info = {"path": CODEX, "sha256": None, "version": None}
    p = Path(CODEX)
    if p.exists():
        info["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            out = subprocess.run([CODEX, "--version"], capture_output=True, text=True, timeout=60)
            info["version"] = (out.stdout or out.stderr).strip()
        except Exception as exc:  # noqa: BLE001
            info["version"] = f"error: {exc!r}"
    return info


def agents_md_state() -> dict:
    return {"global_present": GLOBAL_AGENTS.exists(), "paused_by_this_batch": PAUSED_AGENTS.exists(),
            "other_paused": sorted(p.name for p in GLOBAL_AGENTS.parent.glob("AGENTS.md.paused-by-*")
                                   if p.name != PAUSED_AGENTS.name)}


def settings_problem(info: dict) -> str | None:
    """A reason to stop the batch, read from one session record, or None."""
    if not info:
        return "no session record found"
    if info.get("model") != MODEL:
        return f"model recorded as {info.get('model')}"
    if info.get("effort") != EFFORT:                     # a missing value is a problem too (review 26.10.2026, 9a)
        return f"effort recorded as {info.get('effort')}"
    if info.get("service_tier") != TIER:
        return f"service tier recorded as {info.get('service_tier')}"
    if info.get("agents_md_loaded"):
        return "global AGENTS.md was loaded"
    return None


class QuotaGuard:
    """Stops the batch on a credit change, a projected 5-hour reading above max_primary, a weekly reading at
    max_secondary or a settings problem.

    Readings lag about one request behind (replication_2026-10, DESIGN.md), so after a call the window may already be one
    step above the reading, and the next call adds another step. The batch therefore stops when
    reading + 2 * step >= max_primary, where step is the median increase per call seen so far (at least min_step).
    """

    def __init__(self, max_primary: float = 95.0, max_secondary: float = 90.0, min_step: float = 1.0,
                 log: Path | None = None):
        self.max_primary, self.max_secondary, self.min_step = max_primary, max_secondary, min_step
        self.credits0 = None
        self.has_credits0 = None
        self.baseline = None
        self.readings: list[float] = []
        self.last = None
        self.stop_reason = None
        self.log = log

    def set_baseline(self, reading: dict) -> None:
        """Credit balance and has_credits of the newest session record before the first call."""
        rl = reading["rate_limits"]
        self.baseline = reading
        self.credits0 = (rl.get("credits") or {}).get("balance")
        self.has_credits0 = (rl.get("credits") or {}).get("has_credits")
        prim, sec = rl.get("primary") or {}, rl.get("secondary") or {}
        line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} baseline from {reading['file']} (modified "
                f"{reading['modified']}): 5h {prim.get('used_percent')}% (resets_at {prim.get('resets_at')}), week "
                f"{sec.get('used_percent')}% (resets_at {sec.get('resets_at')}), credits {self.credits0}, has_credits "
                f"{self.has_credits0}")
        print(line, flush=True)
        if self.log:
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def step(self) -> float:
        inc = [b - a for a, b in zip(self.readings, self.readings[1:]) if b >= a]
        return max(self.min_step, statistics.median(inc)) if inc else max(self.min_step, 1.0)

    def record(self, name: str, manifest: dict) -> None:
        rl = manifest.get("rate_limits_after") or {}
        credits = (rl.get("credits") or {}).get("balance")
        prim = (rl.get("primary") or {}).get("used_percent")
        sec = (rl.get("secondary") or {}).get("used_percent")
        if rl:
            self.last = rl
        if not rl:
            self.stop_reason = self.stop_reason or "no rate limits in the session record"
        if credits is not None:
            if self.credits0 is None:
                self.credits0 = credits
            elif credits != self.credits0:
                self.stop_reason = self.stop_reason or f"credit balance changed {self.credits0} -> {credits}"
        elif rl:
            self.stop_reason = self.stop_reason or "no credit balance in the session record"
        has = (rl.get("credits") or {}).get("has_credits") if rl else None
        if rl and self.has_credits0 is not None and has != self.has_credits0:
            self.stop_reason = self.stop_reason or f"has_credits changed {self.has_credits0} -> {has}"
        for key in ("rate_limit_reached_type", "spend_control_reached"):
            if rl and rl.get(key) is not None:
                self.stop_reason = self.stop_reason or f"{key} = {rl.get(key)}"
        if prim is not None:
            self.readings.append(float(prim))
            if prim + 2 * self.step() >= self.max_primary:
                self.stop_reason = self.stop_reason or (f"5-hour window at {prim}% (+2 x {self.step():.1f} per call "
                                                        f">= {self.max_primary}%)")
        if sec is not None and sec >= self.max_secondary:
            self.stop_reason = self.stop_reason or f"weekly window at {sec}%"
        prob = manifest.get("settings_problem")
        if prob:
            self.stop_reason = self.stop_reason or prob
        line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {name}: {manifest.get('wall_seconds')} s, rc "
                f"{manifest.get('returncode')}, 5h {prim}%, week {sec}%, credits {credits}, tier "
                f"{manifest.get('service_tier_recorded')}, tools {len(manifest.get('tool_calls') or [])}"
                + (f"  STOP: {self.stop_reason}" if self.stop_reason else ""))
        print(line, flush=True)
        if self.log:
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def ok(self) -> bool:
        return self.stop_reason is None

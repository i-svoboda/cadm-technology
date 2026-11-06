"""Run one turn of a virtual-expert conversation through the local Codex CLI and archive it.

A turn is either the first message of a new session (`codex exec`) or a follow-up
message in an existing session (`codex exec resume <session_id>`). Every turn gets
its own folder with the prompt, the command line, the JSONL event stream, stderr,
the final message, a copy of the session record (rollout) and a manifest with the
model settings, timing, token usage and the rate limits reported after the turn.

The command line follows scripts/elicit.py of the TACS SMAA project (same Codex
binary, read-only sandbox, web search, memories, hooks, plugins, multi-agent and
goals off, normal service tier). In addition, the user's config.toml is not loaded,
the Codex environment, permission, apps and collaboration-mode blocks are left out,
and the persona's custom instructions go into the developer message.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CODEX = "codex"  # path to the Codex command line tool
SESSIONS = Path.home() / ".codex" / "sessions"
GLOBAL_AGENTS = Path.home() / ".codex" / "AGENTS.md"

_lock = threading.Lock()


def toml_str(s: str) -> str:
    """A TOML basic string (JSON string escapes are valid TOML escapes)."""
    return json.dumps(s, ensure_ascii=False)


def build_cmd(*, model: str, effort: str, schema: Path | None, out_file: Path,
              resume_id: str | None, developer_instructions: str | None,
              workdir: Path) -> list[str]:
    common = [
        "-m", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "-c", 'service_tier="default"',
        "-c", 'web_search="disabled"',
        "-c", "notify=[]",
        "-c", "include_environment_context=false",
        "-c", "include_permissions_instructions=false",
        "-c", "include_apps_instructions=false",
        "-c", "include_collaboration_mode_instructions=false",
        "-c", "project_doc_max_bytes=0",
        "--disable", "memories", "--disable", "hooks", "--disable", "multi_agent",
        "--disable", "multi_agent_v2", "--disable", "plugins", "--disable", "remote_plugin",
        "--disable", "goals",
        "--ignore-user-config", "--skip-git-repo-check",
    ]
    if developer_instructions:
        common += ["-c", f"developer_instructions={toml_str(developer_instructions)}"]
    if schema is not None:
        common += ["--output-schema", str(schema)]
    common += ["-o", str(out_file), "--json"]
    if resume_id:
        # resume has no --cd/-s options; the process runs inside the session's workdir
        return [CODEX, "exec", "resume", *common, resume_id, "-"]
    return [CODEX, "exec", *common, "-s", "read-only", "-C", str(workdir), "-"]


def find_rollout(session_id: str) -> Path | None:
    hits = sorted(SESSIONS.rglob(f"*{session_id}*.jsonl"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def read_rollout(path: Path) -> dict:
    """Last rate limits, thread settings and turn context recorded in a session file."""
    info = {"rate_limits": None, "service_tier": None, "model": None, "effort": None,
            "agents_md_loaded": False, "n_token_count": 0}
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
        if r.get("type") == "turn_context":
            info["effort"] = p.get("effort") or info["effort"]
            info["model"] = p.get("model") or info["model"]
        if r.get("type") == "world_state":
            am = ((p.get("state") or {}).get("agents_md")) or {}
            if am.get("text"):
                info["agents_md_loaded"] = True
        if r.get("type") == "response_item" and p.get("type") == "message":
            for c in p.get("content") or []:
                if "AGENTS.md instructions" in (c.get("text") or ""):
                    info["agents_md_loaded"] = True
    return info


def run_turn(turn_dir: Path, prompt: str, *, model: str, effort: str, workdir: Path,
             schema: Path | None = None, resume_id: str | None = None,
             developer_instructions: str | None = None, timeout: int = 3600,
             meta: dict | None = None) -> dict:
    turn_dir = Path(turn_dir)
    turn_dir.mkdir(parents=True, exist_ok=False)
    workdir.mkdir(parents=True, exist_ok=True)
    out_file = turn_dir / "last_message.txt"
    cmd = build_cmd(model=model, effort=effort, schema=schema, out_file=out_file,
                    resume_id=resume_id, developer_instructions=developer_instructions,
                    workdir=workdir)
    (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if developer_instructions:
        (turn_dir / "developer_instructions.txt").write_text(developer_instructions, encoding="utf-8")
    if schema is not None:
        shutil.copy2(schema, turn_dir / "output_schema.json")
    (turn_dir / "command.json").write_text(json.dumps(cmd[1:], indent=1, ensure_ascii=False), encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rc = None
    err = None
    with open(turn_dir / "events.jsonl", "w", encoding="utf-8") as ev, \
            open(turn_dir / "stderr.txt", "w", encoding="utf-8") as er:
        try:
            proc = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", stdout=ev,
                                  stderr=er, timeout=timeout, cwd=str(workdir))
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            err = "timeout"
    wall = time.time() - t0
    events = []
    for line in (turn_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    session_id = resume_id or next((e.get("thread_id") or e.get("session_id") for e in events
                                    if e.get("thread_id") or e.get("session_id")), None)
    usage = [e.get("usage") for e in events if "usage" in e]
    errors = [e for e in events if e.get("type") in ("error", "turn.failed")]
    info = {}
    if session_id:
        with _lock:
            rollout = find_rollout(session_id)
            if rollout:
                shutil.copy2(rollout, turn_dir / "session_rollout.jsonl")
                info = read_rollout(turn_dir / "session_rollout.jsonl")
    text = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
    answer = None
    if schema is not None and text.strip():
        try:
            answer = json.loads(text)
        except json.JSONDecodeError as exc:
            answer = {"parse_error": str(exc)}
    manifest = {
        **(meta or {}),
        "model_requested": model, "effort_requested": effort, "service_tier_requested": "default",
        "model_recorded": info.get("model"), "effort_recorded": info.get("effort"),
        "service_tier_recorded": info.get("service_tier"),
        "global_agents_md_loaded": info.get("agents_md_loaded"),
        "session_id": session_id, "resumed": bool(resume_id),
        "started_utc": started, "wall_seconds": round(wall, 1), "returncode": rc, "error": err,
        "event_errors": errors, "usage": usage, "rate_limits_after": info.get("rate_limits"),
        "token_info_after": info.get("token_info"),
    }
    (turn_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    if answer is not None:
        (turn_dir / "answer.json").write_text(json.dumps(answer, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "text": text, "answer": answer, "session_id": session_id}


class QuotaGuard:
    """Stops the batch when the credit balance moves or a quota threshold is reached."""

    def __init__(self, max_primary: float = 95.0, max_secondary: float = 90.0, log: Path | None = None):
        self.max_primary = max_primary
        self.max_secondary = max_secondary
        self.credits0 = None
        self.last = None
        self.stop_reason = None
        self.log = log
        self._lock = threading.Lock()

    def record(self, name: str, manifest: dict) -> None:
        rl = manifest.get("rate_limits_after") or {}
        with self._lock:
            credits = (rl.get("credits") or {}).get("balance")
            prim = (rl.get("primary") or {}).get("used_percent")
            sec = (rl.get("secondary") or {}).get("used_percent")
            if rl:
                self.last = rl
            if credits is not None:
                if self.credits0 is None:
                    self.credits0 = credits
                elif credits != self.credits0:
                    self.stop_reason = f"credit balance changed {self.credits0} -> {credits}"
            if prim is not None and prim >= self.max_primary:
                self.stop_reason = self.stop_reason or f"5-hour quota at {prim}%"
            if sec is not None and sec >= self.max_secondary:
                self.stop_reason = self.stop_reason or f"weekly quota at {sec}%"
            tier = manifest.get("service_tier_recorded")
            if tier not in (None, "default"):
                self.stop_reason = self.stop_reason or f"service tier recorded as {tier}"
            line = (f"{datetime.now().strftime('%H:%M:%S')} {name}: {manifest.get('wall_seconds')} s, rc "
                    f"{manifest.get('returncode')}, 5h {prim}%, week {sec}%, credits {credits}, tier {tier}"
                    + (f"  STOP: {self.stop_reason}" if self.stop_reason else ""))
            print(line, flush=True)
            if self.log:
                with open(self.log, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def ok(self) -> bool:
        return self.stop_reason is None

    def primary_reset_at(self) -> float | None:
        return ((self.last or {}).get("primary") or {}).get("resets_at")

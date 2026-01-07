

import os
import re
import asyncio
import sys
import tempfile
import subprocess
import shutil
import uuid
import json
import importlib
from dataclasses import dataclass
import html
from typing import Optional, Dict, Any, Tuple, List
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------
# PIP install command (convenience)
# ---------------------------
PIP_INSTALL_CMD = """
pip install \
  requests beautifulsoup4 selenium webdriver-manager playwright \
  pyautogui pywinauto keyboard pynput Pillow opencv-python pandas numpy \
  matplotlib moviepy pygame psutil watchdog pyperclip \
  pytesseract mss openpyxl PyPDF2 yagmail apscheduler \
  edge-tts pedalboard pyfxr gensound
"""


# ---------------------------
# Requested libs (pyppeteer removed)
# ---------------------------
REQUESTED_LIBS = [
    # stdlib
    "os", "sys", "time", "re", "json", "subprocess", "threading",
    "multiprocessing", "tempfile", "math", "datetime", "pathlib",
    "logging", "shutil", "inspect", "queue", "traceback", "functools",
    "itertools", "base64", "hashlib", "typing", "signal",
    # audio / synthesis
    "edge_tts",
    "pedalboard",
    "pyfxr",
    "gensound",

    # local LLM wrapper - Use full path or relative import
    
    "TTT",  # module

    # web / scraping / browser automation
    "requests",
    "bs4.BeautifulSoup",
    "selenium",
    "webdriver_manager",
    "playwright",

    # desktop automation & input
    "pyautogui",
    "pywinauto",
    "keyboard",
    "pynput",

    # imaging / OCR / screenshots
    "PIL.Image",
    "cv2",
    "pytesseract",
    "mss",

    # data / files / office
    "pandas",
    "numpy",
    "matplotlib.pyplot",
    "openpyxl",
    "PyPDF2",

    # media
    "moviepy.editor",
    "pygame",

    # system / utilities
    "psutil",
    "watchdog",
    "pyperclip",
    "yagmail",
    "apscheduler.schedulers.background",

    # GUI libs (optional)
    "customtkinter",
    "tkinter",
]

# dedupe while preserving order
_seen = set()
DEFAULT_IMPORTS = [x for x in REQUESTED_LIBS if not (x in _seen or _seen.add(x))]

import tempfile
import subprocess
import os
import textwrap
import ast
from typing import Tuple, Dict, Any
import ast
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple


def format_and_autofix_code(
    code: str,
    *,
    spaces_per_level: int = 4,
    timeout: int = 8,
    tool_sequence: Optional[List[List[str]]] = None,
    max_size_kb: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize, format, and attempt to auto-fix a Python code string using available tools.
    Returns (final_code, report).

    Strategy:
    1. Normalize newlines, dedent, replace tabs with spaces, ensure trailing newline.
    2. Fast AST check; if it fails, try minimal whitespace fixes.
    3. Persist code to a temp file and run a sequence of formatters/fixers if present on PATH:
       default sequence: isort -> ruff --fix -> black -> isort -> autopep8
       (each step is skipped if the executable isn't found).
    4. Run optional static checkers (pyflakes, flake8) to gather diagnostics.
    5. Re-parse AST and return formatted code + detailed report.

    Notes:
    - This function never executes the code itself; only parses and formats.
    - External tools are only invoked if present on PATH. Missing tools are noted in report.
    - The return 'report' contains per-tool results, AST checks, warnings and errors.
    """
    start_ts = time.time()
    report: Dict[str, Any] = {
        "tools_run": [],
        "tool_results": {},  # name -> { returncode, stdout, stderr, duration }
        "errors": [],
        "warnings": [],
        "steps": [],
        "ast_ok_before": None,
        "ast_ok_after_minifix": None,
        "final_ast_ok": None,
        "size_kb": None,
        "elapsed_seconds": None,
    }

    # 0) Quick size guard
    try:
        size_kb = (len(code.encode("utf-8")) // 1024) + 1
        report["size_kb"] = size_kb
        if size_kb > max_size_kb:
            report["errors"].append(f"Code size {size_kb}KB exceeds max_size_kb {max_size_kb}KB")
            # Do not attempt big, risky automatic fixes; return normalized input
            normalized = textwrap.dedent(code).replace("\r\n", "\n").replace("\r", "\n")
            report["elapsed_seconds"] = time.time() - start_ts
            return normalized, report
    except Exception as e:
        # non-fatal; continue
        report["warnings"].append(f"Size check failed: {e}")

    # 1) Basic normalization
    code_norm = code.replace("\r\n", "\n").replace("\r", "\n")
    code_norm = textwrap.dedent(code_norm)
    # replace tabs with spaces (conservative)
    code_norm = code_norm.replace("\t", " " * spaces_per_level)
    # strip trailing/leading blank lines but keep one trailing newline
    code_norm = "\n".join([ln.rstrip() for ln in code_norm.splitlines()]).strip() + "\n"

    # 2) Initial AST check
    try:
        ast.parse(code_norm)
        report["ast_ok_before"] = True
        report["steps"].append("ast_ok_before")
    except SyntaxError as se:
        report["ast_ok_before"] = False
        report["errors"].append(f"Initial SyntaxError: {se}")
        report["steps"].append("initial_ast_failed")
        # Minimal whitespace/indentation heuristics to try to fix common issues
        try:
            lines = code_norm.splitlines()
            fixed_lines: List[str] = []
            for ln in lines:
                # collapse trailing spaces, ensure consistent leading spaces multiple of spaces_per_level
                leading = len(ln) - len(ln.lstrip(" "))
                # round down to nearest indent level
                new_leading = (leading // spaces_per_level) * spaces_per_level
                fixed_ln = " " * new_leading + ln.lstrip(" ")
                fixed_lines.append(fixed_ln)
            code_candidate = "\n".join(fixed_lines).rstrip() + "\n"
            ast.parse(code_candidate)
            code_norm = code_candidate
            report["ast_ok_after_minifix"] = True
            report["warnings"].append("Initial SyntaxError resolved by minimal whitespace/indent fixes.")
            report["steps"].append("ast_ok_after_minifix")
        except SyntaxError as se2:
            report["ast_ok_after_minifix"] = False
            report["errors"].append(f"SyntaxError after minimal fix: {se2}")
            report["steps"].append("ast_still_bad_after_minifix")
            # continue: maybe external tools can help

    # 3) Prepare temp file
    fd, path = tempfile.mkstemp(prefix="format_tmp_", suffix=".py")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code_norm)

        # Helper to run commands defensively
        def _run_cmd(cmd: List[str], tool_name: str) -> Optional[subprocess.CompletedProcess]:
            # check tool availability
            if not cmd:
                return None
            exe = cmd[0]
            if shutil.which(exe) is None:
                report["warnings"].append(f"{tool_name} not found on PATH; skipped.")
                return None
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                duration = time.time() - start
                report["tools_run"].append(tool_name)
                report["tool_results"][tool_name] = {
                    "returncode": proc.returncode,
                    "stdout": (proc.stdout or "").strip(),
                    "stderr": (proc.stderr or "").strip(),
                    "duration_seconds": duration,
                    "cmd": " ".join(cmd),
                }
                # non-zero return is captured as a warning but does not abort
                if proc.returncode != 0:
                    report["warnings"].append(f"{tool_name} exited with code {proc.returncode}")
                return proc
            except subprocess.TimeoutExpired:
                report["warnings"].append(f"{tool_name} timed out after {timeout}s")
                return None
            except Exception as e:
                report["warnings"].append(f"{tool_name} error: {e}")
                return None

        # Default sequence (each entry is a argv-style list)
        if tool_sequence is None:
            tool_sequence = [
                ["isort", path],
                ["ruff", "check", path, "--fix"],  # some ruff versions support `ruff check --fix`
                ["ruff", "check", "--fix", path],  # alternative ordering if previous fails
                ["black", "--fast", path],
                ["isort", path],  # isort again after black if needed
                ["autopep8", "--in-place", "--aggressive", path],
            ]

        # Run each tool but dedupe attempts to similar tools (avoid running same tool twice unnecessarily)
        seen_tools = set()
        for cmd in tool_sequence:
            if not cmd:
                continue
            tool_name = cmd[0]
            # collapse some aliases so we don't double-run the same tool
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)
            # try several variants (some CLIs vary) — we attempt the provided form
            _run_cmd(cmd, tool_name)

        # 4) Read back the file
        try:
            with open(path, "r", encoding="utf-8") as f:
                final_code = f.read()
        except Exception as e:
            report["errors"].append(f"Failed to read temp file after formatting: {e}")
            final_code = code_norm

        # 5) Final AST check
        try:
            ast.parse(final_code)
            report["final_ast_ok"] = True
            report["steps"].append("final_ast_ok")
        except SyntaxError as se_final:
            report["final_ast_ok"] = False
            report["errors"].append(f"Final SyntaxError: {se_final}")
            report["steps"].append("final_ast_bad")
            # fallback: try to use the best known earlier variant
            # prefer the minimally fixed code if earlier AST was OK
            if report.get("ast_ok_before"):
                final_code = code_norm  # original normalized code
            else:
                # keep the file content anyway (could be partially fixed)
                final_code = final_code or code_norm

        # 6) Optional static diagnostics (run once each if available)
        for checker_cmd, checker_name in (
            (["pyflakes", path], "pyflakes"),
            (["flake8", path], "flake8"),
            (["ruff", "check", path], "ruff_check"),
        ):
            if shutil.which(checker_cmd[0]) is None:
                report["warnings"].append(f"{checker_name} not found on PATH; skipped.")
                continue
            try:
                proc = subprocess.run(checker_cmd, capture_output=True, text=True, timeout=timeout)
                report.setdefault("static_output", {})[checker_name] = {
                    "returncode": proc.returncode,
                    "stdout": (proc.stdout or "").strip()[:5000],
                    "stderr": (proc.stderr or "").strip()[:5000],
                }
            except subprocess.TimeoutExpired:
                report["warnings"].append(f"{checker_name} timed out.")
            except Exception as e:
                report["warnings"].append(f"{checker_name} error: {e}")

        # Final trimming: ensure trailing newline and consistent CRLF normalization
        final_code = final_code.replace("\r\n", "\n").replace("\r", "\n")
        if not final_code.endswith("\n"):
            final_code = final_code + "\n"

        report["elapsed_seconds"] = time.time() - start_ts
        return final_code, report

    finally:
        # clean up temp file
        try:
            os.remove(path)
        except Exception:
            pass

def _summarize_formatter_report(fmt_report: dict, max_chars: int = 2000) -> str:
    """
    Create a concise, LLM-friendly summary of formatter / AST failures.
    """
    parts = []

    for err in fmt_report.get("errors", []):
        parts.append(f"ERROR: {err}")

    for tool, info in fmt_report.get("tool_results", {}).items():
        if info.get("returncode", 0) != 0:
            stderr = (info.get("stderr") or "").strip()
            if stderr:
                parts.append(f"{tool} stderr:\n{stderr}")

    static = fmt_report.get("static_output", {})
    for name, out in static.items():
        stdout = out.get("stdout")
        if stdout:
            parts.append(f"{name} output:\n{stdout}")

    summary = "\n\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n\n[truncated]"

    return summary or "Unknown formatting or syntax error."


def _build_import_block(imports: List[str], enforce_allowlist: Optional[set] = None) -> str:
    """
    Build a safe import block.
    For dotted names, import the full dotted module via importlib.import_module and
    assign the last segment to a variable (non-fatal).
    """
    lines = []
    lines.append("# ---- auto-imports (generated by ActionEngine) ----")
    lines.append("import importlib\n")
    for name in imports:
        root = name.split(".", 1)[0]
        if enforce_allowlist is not None and root not in enforce_allowlist:
            lines.append(f"# skipping import {name} (not in allowlist)")
            continue
        if "." in name:
            var_name = name.split(".")[-1]
            safe_name = name.replace("'", "\\'")
            lines.append("try:")
            lines.append(f"    _tmp = importlib.import_module('{safe_name}')")
            lines.append(f"    {var_name} = _tmp")
            lines.append("except Exception:")
            lines.append(f"    {var_name} = None")
        else:
            var_name = name
            lines.append("try:")
            lines.append(f"    import {name}")
            lines.append("except Exception:")
            lines.append(f"    {var_name} = None")
    lines.append("# -----------------------------------\n")
    return "\n".join(lines)


@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class ActionEngine:
    def __init__(self,
                 extra_imports: Optional[List[str]] = None,
                 python_executable: Optional[str] = None,
                 allowlist_modules: Optional[set] = None):
        self.python_executable = python_executable or sys.executable
        base = list(DEFAULT_IMPORTS)
        if extra_imports:
            base.extend(extra_imports)
        seen = set()
        self.imports = [x for x in base if not (x in seen or seen.add(x))]
        self.allowlist = allowlist_modules or None

    def check_available_modules(self) -> Dict[str, bool]:
        availability = {}
        for full in self.imports:
            root = full.split(".", 1)[0]
            if root in availability:
                continue
            try:
                importlib.import_module(root)
                availability[root] = True
            except Exception:
                availability[root] = False
        return availability

    def prepare_script(self, user_code: str, header: Optional[str] = None, resource_limits: Optional[dict] = None) -> str:
        import_block = _build_import_block(self.imports, enforce_allowlist=self.allowlist)
        header = header or "# Script generated by ActionEngine"
        limit_block = ""
        if resource_limits:
            limit_lines = [
                "try:",
                "    import resource",
            ]
            if "mem_bytes" in resource_limits:
                limit_lines.append(f"    resource.setrlimit(resource.RLIMIT_AS, ( {resource_limits['mem_bytes']}, {resource_limits['mem_bytes']} ))")
            if "cpu_seconds" in resource_limits:
                limit_lines.append(f"    resource.setrlimit(resource.RLIMIT_CPU, ( {resource_limits['cpu_seconds']}, {resource_limits['cpu_seconds']} ))")
            limit_lines.append("except Exception:")
            limit_lines.append("    pass")
            limit_block = "\n".join(limit_lines) + "\n\n"
        final = f"{import_block}\n{limit_block}{header}\n\n{user_code}\n"
        return final

    def execute(self,
                user_code: str,
                timeout: int = 30,
                dry_run: bool = False,
                working_dir: Optional[str] = None,
                env: Optional[Dict[str, str]] = None,
                resource_limits: Optional[dict] = None) -> Tuple[Optional[ExecutionResult], Optional[str]]:
        """
        Backwards-compatible: writes a temp file and runs it. Kept for compatibility.
        """
        script_text = self.prepare_script(user_code, resource_limits=resource_limits)
        if dry_run:
            return None, script_text

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="action_engine_", suffix=".py")
        os.close(tmp_fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(script_text)

        child_env = os.environ.copy()
        if env:
            child_env.update(env)

        try:
            proc = subprocess.run(
                [self.python_executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env=child_env,
            )
            return ExecutionResult(proc.returncode, proc.stdout, proc.stderr, False), tmp_path
        except subprocess.TimeoutExpired as te:
            return ExecutionResult(-1, te.stdout or "", str(te), True), tmp_path
        except Exception as e:
            return ExecutionResult(-1, "", str(e), False), tmp_path

    def run_code_block(self,
                       code_block: str,
                       timeout: int = 60,
                       dry_run: bool = False,
                       working_dir: Optional[str] = None,
                       env: Optional[Dict[str, str]] = None,
                       resource_limits: Optional[dict] = None) -> Tuple[Optional[ExecutionResult], Optional[str]]:
        """
        Compose an import block and the given code block, then execute it via:
            python -c "<full_script>"
        Returns (ExecutionResult, script_text_or_None). If dry_run=True, returns (None, script_text).
        """
        import_block = _build_import_block(self.imports, enforce_allowlist=self.allowlist)
        limit_block = ""
        if resource_limits:
            limit_lines = [
                "try:",
                "    import resource",
            ]
            if "mem_bytes" in resource_limits:
                limit_lines.append(f"    resource.setrlimit(resource.RLIMIT_AS, ( {resource_limits['mem_bytes']}, {resource_limits['mem_bytes']} ))")
            if "cpu_seconds" in resource_limits:
                limit_lines.append(f"    resource.setrlimit(resource.RLIMIT_CPU, ( {resource_limits['cpu_seconds']}, {resource_limits['cpu_seconds']} ))")
            limit_lines.append("except Exception:")
            limit_lines.append("    pass")
            limit_block = "\n".join(limit_lines) + "\n\n"

        header = "# Script generated by ActionEngine (in-memory execution)\n"
        full_script = f"{import_block}\n{limit_block}{header}\n{code_block}\n"

        if dry_run:
            return None, full_script

        child_env = os.environ.copy()
        if env:
            child_env.update(env)

        try:
            proc = subprocess.run(
                [self.python_executable, "-c", full_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env=child_env,
            )
            return ExecutionResult(proc.returncode, proc.stdout, proc.stderr, False), None
        except subprocess.TimeoutExpired as te:
            return ExecutionResult(-1, te.stdout or "", str(te), True), None
        except Exception as e:
            return ExecutionResult(-1, "", str(e), False), None



    # -------------------------------------------------------------------------
    # NEW unified test: generate a single big code block that imports & tests all libs
    # -------------------------------------------------------------------------
    def generate_full_test_code(self) -> str:
        """
        Returns one big code block that imports and exercises most of the requested libraries.
        Each test prints standardized lines:
          [TEST][<lib>] OK: <message>
          [TEST][<lib>] SKIP: <reason>
          [TEST][<lib>] ERROR: <details>
        The code is intentionally defensive and non-destructive.
        """
        return textwrap.dedent(r"""
        import sys, os, traceback, json

        def _ok(lib, msg=""):
            print(f"[TEST][{lib}] OK: {msg}")

        def _skip(lib, reason=""):
            print(f"[TEST][{lib}] SKIP: {reason}")

        def _err(lib, ex):
            tb = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            print(f"[TEST][{lib}] ERROR: {ex}\n{tb}")

        # -------- requests + BeautifulSoup ----------
        try:
            import requests
            r = requests.get("https://example.com", timeout=10)
            if r.status_code == 200:
                _ok("requests", f"status={r.status_code}")
            else:
                _err("requests", Exception(f"status={r.status_code}"))
        except Exception as e:
            _err("requests", e)

        try:
            from bs4 import BeautifulSoup
            # parse a tiny html
            soup = BeautifulSoup("<html><body><h1>Hi</h1></body></html>", "html.parser")
            _ok("bs4", soup.h1.get_text())
        except Exception as e:
            _err("bs4", e)

        # -------- selenium (webdriver-manager) ----------
        try:
            import selenium
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            opts = Options()
            opts.add_argument("--headless=new")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=opts)
            driver.get("https://example.com")
            _ok("selenium", f"title={driver.title}")
            driver.quit()
        except Exception as e:
            _err("selenium", e)

        # -------- playwright (sync) ----------
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto("https://example.com")
                _ok("playwright", page.title())
                browser.close()
        except Exception as e:
            _err("playwright", e)

        # -------- desktop automation checks (non-destructive) ----------
        try:
            import pyautogui
            # only query mouse position (do NOT move/click)
            pos = pyautogui.position()
            _ok("pyautogui", f"position={pos}")
        except Exception as e:
            _err("pyautogui", e)

        try:
            import pywinauto
            _ok("pywinauto", "imported")
        except Exception as e:
            _err("pywinauto", e)

        try:
            import keyboard
            _ok("keyboard", "imported")
        except Exception as e:
            _err("keyboard", e)

        try:
            import pynput
            _ok("pynput", "imported")
        except Exception as e:
            _err("pynput", e)

        # -------- imaging / cv / OCR / screenshots ----------
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.new("RGB", (200, 80), color=(73, 109, 137))
            im_path = "test_image_pil.png"
            im.save(im_path)
            _ok("PIL", f"wrote {im_path}")
        except Exception as e:
            _err("PIL", e)

        try:
            import cv2
            import numpy as _np
            arr = _np.zeros((10,10,3), dtype=_np.uint8)
            cv2.imwrite("debug_cv.png", arr)
            _ok("cv2", "wrote debug_cv.png")
        except Exception as e:
            _err("cv2", e)

        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            # simple check; will raise if tesseract binary not installed
            try:
                from PIL import Image
                Image.new("RGB",(10,10)).save("ocr_probe.png")
                _ = pytesseract.image_to_string("ocr_probe.png")
                _ok("pytesseract", "ran image_to_string")
            except Exception as inner:
                _err("pytesseract", inner)
        except Exception as e:
            _err("pytesseract", e)
        # -------- audio / speech / synthesis ----------

        # edge-tts (speech)
        try:
            import asyncio
            import edge_tts

            async def _edge_tts_test():
                communicate = edge_tts.Communicate(
                    text="Hello from ActionEngine",
                    voice="en-US-AriaNeural"
                )
                await communicate.save("test_edge_tts.wav")

            asyncio.run(_edge_tts_test())
            _ok("edge_tts", "wrote test_edge_tts.wav")
        except Exception as e:
            _err("edge_tts", e)

        # pedalboard (DSP effects)
        try:
            from pedalboard import Pedalboard, Gain
            from pedalboard.io import AudioFile
            import numpy as np

            sr = 44100
            tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.25, int(sr * 0.25)))
            board = Pedalboard([Gain(gain_db=6)])

            effected = board(tone, sr)

            with AudioFile("test_pedalboard.wav", "w", sr, 1) as f:
                f.write(effected)

            _ok("pedalboard", "wrote test_pedalboard.wav")
        except Exception as e:
            _err("pedalboard", e)

        # -------- Gemini (local TTT.py integration) ----------

        try:
            import asyncio
            from TTT import Gemini

            async def _gemini_test():
                async with Gemini() as session:
                    result = await session.send(
                        user_prompt="Say hello in one short sentence.",
                        system_prompt="Respond briefly.",
                        return_blocks=["response"],
                        stream=False
                    )
                    return result

            res = asyncio.run(_gemini_test())
            if res and "blocks" in res:
                _ok("Gemini", f"blocks={list(res['blocks'].keys())}")
            else:
                _skip("Gemini", "no result blocks returned")

        except Exception as e:
            _err("Gemini", e)

        try:
            import mss
            with mss.mss() as s:
                mon = s.monitors[1] if len(s.monitors)>1 else s.monitors[0]
                img = s.grab(mon)
                _ok("mss", f"grabbed monitor size={img.size}")
        except Exception as e:
            _err("mss", e)

        # -------- data / pandas / numpy / matplotlib / openpyxl ----------
        try:
            import pandas as pd
            import numpy as np
            df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
            csv_path = "test_df.csv"
            df.to_csv(csv_path, index=False)
            _ok("pandas", f"wrote {csv_path}")
        except Exception as e:
            _err("pandas", e)

        try:
            import numpy as np
            _ok("numpy", f"version={np.__version__}")
        except Exception as e:
            _err("numpy", e)

        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot([0,1,2],[0,1,0])
            plt.savefig("test_plot.png")
            _ok("matplotlib", "wrote test_plot.png")
        except Exception as e:
            _err("matplotlib", e)

        try:
            import openpyxl
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws["A1"] = "demo"
            wb.save("test_openpyxl.xlsx")
            _ok("openpyxl", "wrote test_openpyxl.xlsx")
        except Exception as e:
            _err("openpyxl", e)

        # -------- PDF / PyPDF2 ----------
        try:
            import PyPDF2
            _ok("PyPDF2", f"version? has__version__={hasattr(PyPDF2,'__version__')}")
        except Exception as e:
            _err("PyPDF2", e)

        # -------- media (moviepy, pygame) ----------
        try:
            from moviepy import ColorClip
            clip = ColorClip(
                size=(64, 48),
                color=(255, 0, 0),
                duration=0.2
            )
        
            clip.write_videofile(
                "test_moviepy.mp4",
                fps=24,
                codec="libx264",
                audio=False,
            )
            _ok("moviepy", "wrote test_moviepy.mp4")
        except Exception as e:
            _err("moviepy", e)

        try:
            import pygame
            pygame.init()
            pygame.display.init()
            _ok("pygame", "initialized")
            pygame.display.quit()
            pygame.quit()
        except Exception as e:
            _err("pygame", e)

        # -------- system / psutil / watchdog / clipboard / email / scheduler ----------
        try:
            import psutil
            _ok("psutil", f"cpu_count={psutil.cpu_count(logical=True)}")
        except Exception as e:
            _err("psutil", e)

        try:
            import watchdog
            _ok("watchdog", "imported")
        except Exception as e:
            _err("watchdog", e)

        try:
            import pyperclip
            pyperclip.copy("test")
            v = pyperclip.paste()
            _ok("pyperclip", f"roundtrip={v}")
        except Exception as e:
            _err("pyperclip", e)

        try:
            import yagmail
            _ok("yagmail", "imported")
        except Exception as e:
            _err("yagmail", e)

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            _ok("apscheduler", "imported")
        except Exception as e:
            _err("apscheduler", e)

        print("[TEST][ALL] COMPLETE")
        """)

    def run_full_test(self, dry_run: bool = False, timeout: int = 120) -> Tuple[ExecutionResult, Optional[str]]:
        """
        Convenience runner: generate the full test code, optionally dry-run, then execute.
        Returns (ExecutionResult, script_text_or_None). If dry_run=True, ExecutionResult is None and returned script is second item.
        """
        code = self.generate_full_test_code()
        return self.run_code_block(code, timeout=timeout, dry_run=dry_run)
DEFAULT_ACTIONAGENT_SYSTEM_PROMPT = (
    "CRITICAL INDENTATION RULES - READ CAREFULLY:\n"
    "1. Python code MUST use EXACT 4-space indentation\n"
    "2. After EVERY colon (:), the next line MUST be indented by 4 spaces\n"
    "3. Examples of CORRECT indentation:\n"
    "   try:\\n"
    "       x = 1  # 4 spaces after try:\\n"
    "   for i in range(5):\\n"
    "       print(i)  # 4 spaces after for:\\n"
    "   if True:\\n"
    "       pass  # 4 spaces after if:\\n"
    "\n"
    "4. Common mistakes to AVOID:\n"
    "   - WRONG: try:\\n products = []  # only 1 space\n"
    "   - CORRECT: try:\\n    products = []  # 4 spaces\n"
    "\n"
    "5. String literals MUST use \\\\n for newlines, not actual newlines\n"
    "   - WRONG: 'line1\\nline2' with actual newline in string\n"
    "   - CORRECT: 'line1\\\\nline2' with \\\\n escape\n"
    "\n"
    "You are ActionAgent. Turn user requests into runnable Python for ActionEngine.\n\n"

    "FORMAT (MUST follow exactly):\n"
    "1) Plan: short 1–3 sentences describing approach, key packages, assumptions, and approvals. "
    "Wrap the plan in <think>...</think>.\n"
    "2) Code: exactly one <code>...</code> block containing a SINGLE valid JSON object.\n"
    "   The JSON object MUST have exactly one key: \"python\".\n"
    "   The value of \"python\" MUST be a single multiline string containing ONLY runnable Python code.\n"
    "   Use real newlines and real indentation. Do NOT escape newlines.\n"
    "   Do NOT include prose, markdown, or comments outside the Python code.\n\n"

    "Hard rules:\n"
    "- Plan must be inside <think> tags.\n"
    "- Code must be inside exactly one <code>...</code> block.\n"
    "- <code> must contain ONLY valid JSON (no trailing commas, no comments).\n"
    "- The JSON MUST contain only: {\"python\": \"...\"}.\n"
    "- Do NOT include backticks, markdown, or explanations inside <code>.\n"
    "- If secrets or approvals are required, ask in the plan and DO NOT emit code.\n"
    "- No placeholders like <API_KEY> or REPLACE_ME.\n"
    "- Defensive code: handle exceptions, validate inputs, avoid destructive actions by default.\n"
    "- Machine-readable output:\n"
    "  On success print: \"ACTIONENGINE: OK - <short message>\"\n"
    "  On failure print: \"ACTIONENGINE: ERROR - <short message>\"\n"
    "- If ambiguous or unsafe, ask a clarifying question in the plan and DO NOT emit code.\n\n"

    "Allowed packages:\n"
    "requests, bs4, selenium, webdriver_manager, playwright, pyautogui, pywinauto, "
    "keyboard, pynput, PIL, cv2, pytesseract, mss, pandas, numpy, matplotlib.pyplot, "
    "openpyxl, PyPDF2, moviepy.editor, pygame, edge_tts, pedalboard, psutil, watchdog, "
    "pyperclip, yagmail, apscheduler.schedulers.background, and Python stdlib only.\n\n"

    "Gemini integration (optional):\n"
    "- Use: from TTT import Gemini\n"
    "- Return blocks MUST include 'code'.\n\n"

    "STRICT OUTPUT EXAMPLE:\n"
    "<think>\n"
    "I will create a CSV file with sales data using pandas. I'll generate random sales numbers "
    "for 3 products over 30 days, save to CSV, and print a success message.\n"
    "</think>\n\n"
    "<code>\n"
    "{\n"
    "  \"python\": \"import pandas as pd\\\\nimport numpy as np\\\\nfrom datetime import datetime, timedelta\\\\n\\\\ntry:\\\\n    products = ['Product A', 'Product B', 'Product C']\\\\n    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]\\\\n    \\\\n    data = []\\\\n    for date in dates:\\\\n        for product in products:\\\\n            data.append({\\\\n                'Date': date,\\\\n                'Product': product,\\\\n                'Sales': np.random.randint(10, 100),\\\\n                'Revenue': round(np.random.uniform(100.0, 500.0), 2)\\\\n            })\\\\n    \\\\n    df = pd.DataFrame(data)\\\\n    df.to_csv('sales_data.csv', index=False)\\\\n    print('ACTIONENGINE: OK - sales_data.csv created')\\\\nexcept Exception as e:\\\\n    print(f'ACTIONENGINE: ERROR - {e}')\"\n"
    "}\n"
    "</code>\n\n"
    "End of instructions."
)


_CODE_RE = re.compile(r"(?s)(?<=<code>)(.*?)(?=</code>)", re.IGNORECASE)
_RESPONSE_RE = re.compile(r"<response>\s*(.*?)\s*</response>", re.DOTALL | re.IGNORECASE)

# --- single clean sanitizer ---
def clean_gemini_code(raw: str) -> str:
    """
    Extract Python code from a <code> block containing JSON:
    { "python": "<multiline python code>" }

    Guarantees:
    - Exact indentation preserved
    - Newlines preserved
    """
    if not raw:
        return ""
    print(raw)
    # Unescape HTML entities if needed
    raw = html.unescape(raw)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"<code> block is not valid JSON: {e}")

    if not isinstance(payload, dict) or "python" not in payload:
        raise ValueError("<code> JSON must contain exactly one key: 'python'")

    code = payload["python"]
    print(code)
    if not isinstance(code, str) or not code.strip():
        raise ValueError("'python' value must be a non-empty string")

    # Preserve all indentation and newlines
    return code


def _extract_code_from_gemini(res: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract runnable Python code from Gemini output.
    Never raises. Returns (code, error_message).
    """
    if not res:
        return None, "Empty Gemini response"

    blocks = res.get("blocks", {})

    # 1) Preferred: structured code block
    if isinstance(blocks, dict) and blocks.get("code"):
        try:
            return clean_gemini_code(blocks["code"][0]), None
        except Exception as e:
            return None, f"Invalid <code> JSON: {e}"

    # 2) Fallback: scan full_text
    full_text = res.get("full_text", "")
    if full_text:
        m = _CODE_RE.search(full_text)
        if m:
            try:
                return clean_gemini_code(m.group(1)), None
            except Exception as e:
                return None, f"Invalid <code> JSON in full_text: {e}"

    return None, "No <code> block found"



# Example Gemini caller (same as before)
async def _ask_gemini_for_code(sys_prompt: str, user_text: str):
    from TTT import Gemini
    async with Gemini() as session:
        result = await session.send(
            user_prompt=user_text,
            system_prompt=sys_prompt,
            return_blocks=["code", "response"],
            required_blocks=["code"],
            stream=False,
            max_retries=3,
        )
        return result


def action_tool(
    user_prompt: str,
    engine,  # ActionEngine instance
    *,
    system_prompt: Optional[str] = None,
    max_attempts: int = 10,
    timeout: int = 300,
) -> str:
    system_prompt = system_prompt or DEFAULT_ACTIONAGENT_SYSTEM_PROMPT

    attempt_logs = []
    last_err_summary = None

    for attempt in range(1, max_attempts + 1):
        attempt_log = {
            "attempt": attempt,
            "phase": None,
            "error": None,
            "details": None,
        }

        # ---------------------------
        # 1) Ask Gemini for code
        # ---------------------------
        try:
            res = asyncio.run(_ask_gemini_for_code(system_prompt, user_prompt))
        except Exception as e:
            attempt_log.update(
                phase="gemini_call",
                error=str(e),
            )
            attempt_logs.append(attempt_log)

            last_err_summary = f"Gemini call failed: {e}"
            if attempt < max_attempts:
                system_prompt += (
                    "\n\nPrevious attempt failed during Gemini API call.\n"
                    f"Error:\n{e}"
                )
                continue
            break

        # ---------------------------
        # 2) Extract code
        # ---------------------------
        code_text, extract_err = _extract_code_from_gemini(res)
        if not code_text:
            attempt_log.update(
                phase="code_extraction",
                error=extract_err,
            )
            attempt_logs.append(attempt_log)

            last_err_summary = extract_err or "Code extraction failed"
            if attempt < max_attempts:
                system_prompt += (
                    "\n\nYour previous reply could not be parsed.\n"
                    f"Reason:\n{last_err_summary}\n\n"
                    "Return exactly ONE <code>...</code> block containing ONLY:\n"
                    '{ "python": "<valid runnable python code>" }'
                )
                continue
            break

        # ---------------------------
        # 3) Format + AST validation
        # ---------------------------
        formatted_code, fmt_report = format_and_autofix_code(code_text)
        if not fmt_report.get("final_ast_ok", False):
            attempt_log.update(
                phase="format_ast",
                error="Formatter / AST validation failed",
                details=fmt_report,
            )
            attempt_logs.append(attempt_log)

            last_err_summary = "Formatter / AST validation failed"
            if attempt < max_attempts:
                system_prompt += (
                    "\n\nFormatter / syntax validation failed.\n"
                    "Formatter report:\n"
                    f"{fmt_report}\n\n"
                    "Fix the reported issues and return corrected code."
                )
                continue
            break

        # ---------------------------
        # 4) Execute with ActionEngine
        # ---------------------------
        try:
            result, _ = engine.run_code_block(
                formatted_code, timeout=timeout, dry_run=False
            )
        except Exception as e:
            attempt_log.update(
                phase="execution_exception",
                error=str(e),
            )
            attempt_logs.append(attempt_log)

            last_err_summary = f"Execution raised exception: {e}"
            if attempt < max_attempts:
                system_prompt += (
                    "\n\nCode raised an exception during execution:\n"
                    f"{e}\n\nFix and retry."
                )
                continue
            break

        if result is None:
            attempt_log.update(
                phase="execution_result",
                error="ActionEngine returned None",
            )
            attempt_logs.append(attempt_log)

            last_err_summary = "ActionEngine returned no result"
            if attempt < max_attempts:
                system_prompt += (
                    "\n\nCode executed but returned no result.\n"
                    "Fix logic and retry."
                )
                continue
            break

        # ---------------------------
        # 5) Check execution result
        # ---------------------------
        if result.returncode == 0:
            return "Action Complete"

        attempt_log.update(
            phase="execution_failure",
            error=f"returncode={result.returncode}",
            details={
                "stdout": (result.stdout or "")[:3000],
                "stderr": (result.stderr or "")[:3000],
            },
        )
        attempt_logs.append(attempt_log)

        last_err_summary = (
            f"returncode={result.returncode}\n"
            f"STDOUT:\n{attempt_log['details']['stdout']}\n\n"
            f"STDERR:\n{attempt_log['details']['stderr']}"
        )

        if attempt < max_attempts:
            system_prompt += (
                "\n\nExecution failed.\n"
                f"{last_err_summary}\n\nFix and retry."
            )
            continue

        break

    # ---------------------------
    # FINAL FAILURE REPORT
    # ---------------------------
    debug_report = [
        "ACTION ABORTED",
        "=" * 80,
        f"User prompt:\n{user_prompt}",
        "-" * 80,
        f"Total attempts: {len(attempt_logs)}",
        "-" * 80,
    ]

    for log in attempt_logs:
        debug_report.extend(
            [
                f"ATTEMPT {log['attempt']}",
                f"Phase: {log['phase']}",
                f"Error: {log['error']}",
                f"Details: {log['details']}",
                "-" * 80,
            ]
        )

    return "\n".join(debug_report)



if __name__ == "__main__":
    # 1) create an instance of ActionEngine
    engine = ActionEngine()

    # 2) define a more complex user action
    user_action = (
        "I want a small file I can open that shows some made-up sales numbers "
        "for a few products over the last month. "
        "Please save it for me and let me know when it's ready."
    )

    # 3) run the action_tool
    result = action_tool(user_action, engine)

    print("ActionTool Result:", result)

'''
if __name__ == "__main__":
    engine = ActionEngine()

    print("\n==============================")
    print(" ActionEngine: FULL LIB TEST ")
    print("==============================\n")


    # Live execution
    print("\n>>> LIVE EXECUTION\n")
    result, _ = engine.run_full_test(dry_run=False, timeout=300)
    print("\n--- STDOUT ---\n")
    print(result.stdout)
    print("\n--- STDERR ---\n")
    print(result.stderr)
    print("\nExit code:", result.returncode)
'''
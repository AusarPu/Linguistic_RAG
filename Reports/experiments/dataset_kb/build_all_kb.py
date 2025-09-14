#!/usr/bin/env python3
"""
批量为四个数据集构建各自独立的知识库块文件，避免跨数据集污染。
"""
from subprocess import run
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

def run_script(name: str):
    p = THIS_DIR / name
    print(f"\n== 运行: {p}")
    result = run([sys.executable, str(p)], capture_output=True, text=True, encoding="utf-8")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)
    print(f"退出码: {result.returncode}")


def main():
    scripts = [
        "build_hotpotqa_kb.py",
        "build_msmarco_kb.py",
        "build_nq_kb.py",
        "build_triviaqa_kb.py",
    ]
    for s in scripts:
        run_script(s)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""运行所有纯 Python 预实验。

启动命令：
    python pre实验/run_all.py

参数说明：
    本启动器不接收命令行参数；实验列表由 EXPERIMENTS 常量控制。
"""

from __future__ import annotations

import importlib
import json
import time


EXPERIMENTS = ("h1", "h2", "h4", "h5")


def main() -> None:
    results = []
    for name in EXPERIMENTS:
        t0 = time.perf_counter()
        module = importlib.import_module(name)
        module.main()
        results.append(
            {
                "experiment": name.upper(),
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
        )
    print(json.dumps({"pre_experiments": results}, indent=2))


if __name__ == "__main__":
    main()

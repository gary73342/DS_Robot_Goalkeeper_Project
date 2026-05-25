#!/usr/bin/env python3
"""
log.py

兩種用法：

1. 當腳本（供 C++ 節點的 pipe 使用，行為與舊版相同）：
   command 2>&1 | python3 log.py <取樣間隔> <日誌路徑>

2. 當模組 import（供 Python 節點在 process 內部 tee）：
   from log import setup_log
   setup_log("/path/to/node.log", sample_rate=1)
   之後 print() / rospy.loginfo() 都會同時輸出到終端機與日誌檔。
"""

import sys
from datetime import datetime


# ==============================================================================
# 模組用途：TeeWriter + setup_log
# ==============================================================================

class _TeeWriter:
    """將寫入同時送到終端機（即時顯示）和日誌檔（依 sample_rate 取樣）。"""

    def __init__(self, terminal, logfile, sample_rate=1):
        self.terminal    = terminal
        self.logfile     = logfile
        self.sample_rate = sample_rate
        self._count      = 0
        self._pending    = ""

    def write(self, data):
        self.terminal.write(data)
        self.terminal.flush()
        self._pending += data
        while '\n' in self._pending:
            line, self._pending = self._pending.split('\n', 1)
            self._count += 1
            if self._count % self.sample_rate == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                self.logfile.write(f'[{ts}] {line}\n')
                self.logfile.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log(log_path, sample_rate=1):
    """
    將 stdout 與 stderr 都 tee 到 log_path。
    log_path 為 None 時不做任何事（方便沒傳參數時直接呼叫）。
    """
    if not log_path:
        return
    f = open(log_path, 'a')
    sys.stdout = _TeeWriter(sys.__stdout__, f, sample_rate)
    sys.stderr = _TeeWriter(sys.__stderr__, f, sample_rate)


# ==============================================================================
# 腳本用途：pipe 模式（供 C++ 節點）
# ==============================================================================

if __name__ == "__main__":
    sample_rate = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    log_file    = sys.argv[2]      if len(sys.argv) > 2 else None

    count = 0
    f = open(log_file, 'a') if log_file else None

    try:
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()
            count += 1
            if count % sample_rate == 0 and f:
                ts = datetime.now().strftime('%H:%M:%S')
                f.write(f'[{ts}] {line}')
                f.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if f:
            f.close()

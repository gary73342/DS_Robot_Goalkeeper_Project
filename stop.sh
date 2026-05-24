#!/bin/bash
SESSION="goalkeeper"

if ! tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' 不存在"
    exit 0
fi

tmux kill-session -t $SESSION
echo "已關閉所有節點"

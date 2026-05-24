#!/bin/bash
SESSION="goalkeeper"
PROJECT="$HOME/DS_Robot_Goalkeeper_Project"
VENV="$PROJECT/.venv/bin/activate"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' 已存在，直接附加..."
    tmux select-window -t "${SESSION}:dashboard"
    tmux attach -t $SESSION
    exit 0
fi

go_next() {
    echo -e "  ${CYAN}[提示]${NC} 可另開終端機執行 tmux attach -t $SESSION 查看"
    echo -e "  確認正常後按 ${YELLOW}Enter${NC} 繼續下一個..."
    read -r
    echo ""
}

# 建立 session 與 dashboard 4 格
tmux new-session -d -s $SESSION -n "dashboard"
tmux split-window -h -t "${SESSION}:dashboard.0"
tmux split-window -v -t "${SESSION}:dashboard.0"
tmux split-window -v -t "${SESSION}:dashboard.1"

# 其他獨立視窗
tmux new-window -t $SESSION -n "realsense"
tmux new-window -t $SESSION -n "rviz"
tmux new-window -t $SESSION -n "yolo_server"
tmux new-window -t $SESSION -n "debug"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}   Goalkeeper 節點啟動腳本     ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "請確認 SSH + roscore 已就緒，按 Enter 開始..."
read -r
echo ""

# 1. realsense
echo -e "${GREEN}[1/7]${NC} 啟動 realsense camera..."
tmux send-keys -t "${SESSION}:realsense" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:realsense" "roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_color:=true enable_depth:=true initial_reset:=true" Enter
go_next

# 2. rviz
echo -e "${GREEN}[2/7]${NC} 啟動 rviz..."
tmux send-keys -t "${SESSION}:rviz" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:rviz" "roslaunch sample_code minibot_rviz.launch" Enter
go_next

# 3. yolo_server（不需要 ROS1 環境）
echo -e "${GREEN}[3/7]${NC} 啟動 yolo_server..."
tmux send-keys -t "${SESSION}:yolo_server" "source $VENV && cd $PROJECT/python && python3 yolo_server.py" Enter
go_next

# 4. yolo_ros_node → dashboard 左下
echo -e "${GREEN}[4/7]${NC} 啟動 yolo_ros_node..."
tmux send-keys -t "${SESSION}:dashboard.2" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:dashboard.2" "cd $PROJECT/python && python3 yolo_ros_node.py" Enter
go_next

# 5. localization_node → dashboard 左上
echo -e "${GREEN}[5/7]${NC} 啟動 localization_node..."
tmux send-keys -t "${SESSION}:dashboard.0" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:dashboard.0" "rosrun sample_code localization_node" Enter
go_next

# 6. perception_node → dashboard 右上
echo -e "${GREEN}[6/7]${NC} 啟動 perception_node..."
tmux send-keys -t "${SESSION}:dashboard.1" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:dashboard.1" "rosrun sample_code perception_node" Enter
go_next

# 7. fusion_node → dashboard 右下
echo -e "${GREEN}[7/7]${NC} 啟動 fusion_node..."
tmux send-keys -t "${SESSION}:dashboard.3" "ros1" Enter
sleep 3
tmux send-keys -t "${SESSION}:dashboard.3" "cd $PROJECT/python && python3 fusion_node.py" Enter

# debug 備用終端機
tmux send-keys -t "${SESSION}:debug" "ros1" Enter

echo ""
echo -e "${GREEN}=== 所有節點已啟動！進入 Dashboard... ===${NC}"
sleep 1

tmux select-window -t "${SESSION}:dashboard"
tmux attach -t $SESSION

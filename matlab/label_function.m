clear; clc;

%% === 1. 穩健讀取與切幀邏輯 (修正為弧度判斷) ===
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', 'data', 'laser_data');
filename = fullfile(data_dir, 'laser0517_1.dat');

if ~exist(filename, 'file')
    error('文件 %s 不存在，請確認路徑', filename);
end

fprintf('正在讀取文件: %s...\n', filename);

% 使用 importdata 最穩，直接讀出數值矩陣
data = importdata(filename);
if isstruct(data)
    data = data.data;
end

% Python 寫入的已經是「弧度 (Radians)」！
angles_rad = data(:,1); 
distances  = data(:,2);

% 利用角度跳變來切分 Frame
% 掃描一圈會從約 +3.14 跳回 -3.14，差距大約 6.28
angle_diff = diff(angles_rad);
frame_starts = find(abs(angle_diff) > 5) + 1;
frame_starts = [1; frame_starts];

num_scans = length(frame_starts);
fprintf('成功切分出 %d 幀 (Frames)\n', num_scans);

points_per_scan = diff([frame_starts; length(angles_rad)+1]);
max_pts = max(points_per_scan);

x_frames = zeros(num_scans, max_pts);
y_frames = zeros(num_scans, max_pts);

for i = 1:num_scans
    idx_range = frame_starts(i):(frame_starts(i)+points_per_scan(i)-1);
    ang = angles_rad(idx_range);
    dist = distances(idx_range);
    
    % XY轉換
    x_frames(i, 1:length(idx_range)) = dist .* cos(ang);
    y_frames(i, 1:length(idx_range)) = dist .* sin(ang);
end

fprintf('數據重組完成！準備進入標記畫面...\n');

%% === 2. 升級版：多類別鍵盤手動標記 ===
segment_labels = [];
figure('Position',[100 100 800 800]);
color_goal = [0 0.8 0]; % 綠色
color_ball = [1 0 0];   % 紅色

for frame_idx = 1:num_scans
    clf;
    xy = [x_frames(frame_idx,:)', y_frames(frame_idx,:)'];
    
    % ★ 關鍵防呆：徹底過濾掉無效點與 Python 寫入的 NaN
    valid_mask = (xy(:,1)~=0 | xy(:,2)~=0) & ~isnan(xy(:,1));
    xy = xy(valid_mask, :);
    
    if isempty(xy)
        continue;
    end
    
    % Segment 分段 (使用改良版的動態記憶體配置)
    [Seg, Si_n, S_n] = Segment(xy);
    frame_segment_labels = zeros(S_n,1); 
    
    scatter(xy(:,1), xy(:,2), 30, 'b', 'filled'); hold on;
    plot(0,0,'ro','MarkerSize',10,'MarkerFaceColor','k'); 
    axis equal; grid on; xlim([-2 2]); ylim([-2 2]); 
    
    title(sprintf('第 %d/%d 幀\n游標指著點，按 1=球門(綠), 2=足球(紅), 0=取消\n按 Enter 換下一幀', frame_idx, num_scans));
    xlabel('X (m)'); ylabel('Y (m)');
    
    goal_count = 0; ball_count = 0;
    
    while true
        [x_click, y_click, button] = ginput(1);
        if isempty(x_click)
            break; % 按 Enter 下一幀
        end
        
        % 判斷標籤邏輯
        current_label = -1;
        if button == 1 || button == 49 % 左鍵 或 鍵盤 '1'
            current_label = 1;
        elseif button == 3 || button == 50 % 右鍵 或 鍵盤 '2'
            current_label = 2;
        elseif button == 48 % 鍵盤 '0' 取消標記
            current_label = 0;
        else
            fprintf('無效按鍵，請按 1(球門), 2(足球), 0(取消)\n');
            continue;
        end
        
        seg_found = false;
        for s = 1:S_n
            pts_idx = Seg(1:Si_n(s),s);
            seg_points = xy(pts_idx,:);
            d = sqrt((seg_points(:,1)-x_click).^2 + (seg_points(:,2)-y_click).^2);
            
            if min(d) < 0.05
                frame_segment_labels(s) = current_label;
                if current_label == 1
                    scatter(seg_points(:,1), seg_points(:,2), 50, color_goal, 'filled');
                    goal_count = goal_count + 1;
                    fprintf('  -> 標記為 [球門]\n');
                elseif current_label == 2
                    scatter(seg_points(:,1), seg_points(:,2), 50, color_ball, 'filled');
                    ball_count = ball_count + 1;
                    fprintf('  -> 標記為 [足球]\n');
                elseif current_label == 0
                    scatter(seg_points(:,1), seg_points(:,2), 30, 'b', 'filled');
                    fprintf('  -> 取消標記，退回為 [雜訊]\n');
                end
                seg_found = true;
                break;
            end
        end
        if ~seg_found
            fprintf('點擊太遠了，未對應任何 segment，請靠近一點！\n');
        end
    end
    
    for s = 1:S_n
        segment_labels = [segment_labels; frame_idx, s, frame_segment_labels(s)];
    end
    fprintf('第 %d 幀完成: 標記 %d 個球門, %d 個球\n', frame_idx, goal_count, ball_count);
    pause(0.1);
end

%% === 3. 生成與儲存 labels_per_point ===
segment_labels_table = array2table(segment_labels, 'VariableNames', {'FrameIndex','SegmentIndex','Label'});
save('label_data_multi.mat','segment_labels_table','x_frames','y_frames');

num_frames = size(x_frames,1);
num_points = size(x_frames,2);
labels_per_point = zeros(num_frames, num_points, 'uint8'); 

if ~isempty(segment_labels)
    for frame_idx = 1:num_frames
        x = x_frames(frame_idx,:);
        y = y_frames(frame_idx,:);
        xy = [x', y'];
        
        valid_mask = (x'~=0 | y'~=0) & ~isnan(x');
        valid_idx = find(valid_mask);
        xy_valid = xy(valid_mask,:);
        
        if isempty(xy_valid)
            continue;
        end
        
        [Seg, Si_n, S_n] = Segment(xy_valid);
        
        frame_mask = segment_labels_table.FrameIndex==frame_idx;
        frame_segment_labels = segment_labels_table.Label(frame_mask);
        
        for s = 1:S_n
            if frame_segment_labels(s) > 0 
                seg_pts = Seg(1:Si_n(s), s);      
                orig_pts_idx = valid_idx(seg_pts); 
                labels_per_point(frame_idx, orig_pts_idx) = frame_segment_labels(s);
            end
        end
    end
end

num_goal_pts = sum(labels_per_point(:) == 1);
num_ball_pts = sum(labels_per_point(:) == 2);
num_noise_pts = sum(labels_per_point(:) == 0);
total_pts = numel(labels_per_point);

fprintf('\n=== 標記統計結果 ===\n');
fprintf('總點數: %d\n', total_pts);
fprintf('雜訊點數 (0): %d (%.2f%%)\n', num_noise_pts, 100*num_noise_pts/total_pts);
fprintf('球門點數 (1): %d (%.2f%%)\n', num_goal_pts, 100*num_goal_pts/total_pts);
fprintf('足球點數 (2): %d (%.2f%%)\n', num_ball_pts, 100*num_ball_pts/total_pts);

%% === 4. Segment 函數 (動態記憶體防爆版) ===
function [Seg, Si_n, S_n] = Segment(xy)
    x = xy(:,1); y = xy(:,2);
    threshold = 0.05;
    
    n_s = size(xy,1);
    if n_s == 0
        Seg = []; Si_n = []; S_n = 0; return;
    end
    
    S_i = 1; S_n = 1;
    seg_list = cell(n_s, 1);
    current_seg = zeros(n_s, 1);
    current_seg(S_i) = 1;
    
    for i = 2:n_s
        if sqrt((x(i)-x(i-1))^2 + (y(i)-y(i-1))^2) < threshold
            S_i = S_i + 1;
            current_seg(S_i) = i;
        else
            seg_list{S_n} = current_seg(1:S_i); 
            S_n = S_n + 1;
            S_i = 1;
            current_seg(S_i) = i; 
        end
    end
    seg_list{S_n} = current_seg(1:S_i); 
    
    max_pts_in_seg = 0;
    for j = 1:S_n
        max_pts_in_seg = max(max_pts_in_seg, length(seg_list{j}));
    end
    
    Seg = zeros(max_pts_in_seg, S_n);
    Si_n = zeros(S_n, 1);
    
    for j = 1:S_n
        pts = seg_list{j};
        Si_n(j) = length(pts);
        Seg(1:length(pts), j) = pts;
    end
end

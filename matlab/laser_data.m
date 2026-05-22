% 適用minibot
clear; clc;

% 檢查文件是否存在
filename = 'laser0517_2.dat';
if ~exist(filename, 'file')
    error('文件 %s 不存在。請確保文件在當前工作目錄中。', filename);
else
    fprintf('找到文件: %s\n', filename);
end

% 嘗試讀取文件
try
    data = importdata(filename);
    if isstruct(data)
        data = data.data;
    end
    fprintf('成功讀取數據\n');
catch ME
    fprintf('讀取文件失敗: %s\n', ME.message);
    fprintf('嘗試使用load函數...\n');
    try
        data = load(filename);
    catch
        error('無法讀取數據文件。請檢查文件格式。');
    end
end

% 顯示數據基本信息
fprintf('數據尺寸: %s\n', mat2str(size(data)));

% ---------------------------------------------------------
% 關鍵修正 1：Minibot 的第一列數據已經是「弧度 (Rad)」
% ---------------------------------------------------------
if size(data, 2) >= 2
    angles_rad = data(:, 1);  % 直接讀取弧度
    distances = data(:, 2);
else
    error('數據格式錯誤，預期至少需要兩列 (角度與距離)');
end

fprintf('角度範圍: %.2f rad 到 %.2f rad\n', min(angles_rad), max(angles_rad));

% ---------------------------------------------------------
% 關鍵修正 2：幀邊界檢測 (Frame Boundary Detection)
% 從 +pi 跳回 -pi 的差值大約是 6.28，所以門檻設為 5 即可
% ---------------------------------------------------------
angle_diff = diff(angles_rad);
frame_starts = find(abs(angle_diff) > 5) + 1;  % 修正跳變點門檻

% 添加第一幀的開始
if ~isempty(frame_starts)
    frame_starts = [1; frame_starts];
    num_scans = length(frame_starts);
    
    if num_scans > 1
        points_per_scan = frame_starts(2) - frame_starts(1);
    else
        points_per_scan = length(angles_rad);
    end
else
    % 若找不到跳變點，依據 Minibot 步距 (約 0.0087 rad = 0.5度)，一圈通常是 720 點
    points_per_scan = 720;  
    num_scans = floor(length(angles_rad) / points_per_scan);
    frame_starts = 1:points_per_scan:num_scans*points_per_scan;
    num_scans = length(frame_starts);
end

fprintf('檢測到 %d 點/幀，共 %d 幀\n', points_per_scan, num_scans);

% 重組數據為幀格式（XY座標）
x_frames = zeros(num_scans, points_per_scan);
y_frames = zeros(num_scans, points_per_scan);

for i = 1:num_scans
    start_idx = frame_starts(i);
    end_idx = min(start_idx + points_per_scan - 1, length(angles_rad));
    
    actual_points = end_idx - start_idx + 1;
    
    % 提取當前幀的角度和距離
    current_angles = angles_rad(start_idx:end_idx);
    current_distances = distances(start_idx:end_idx);
    
    % 轉換為 XY 座標 (極座標轉直角座標)
    x_frames(i, 1:actual_points) = current_distances .* cos(current_angles);
    y_frames(i, 1:actual_points) = current_distances .* sin(current_angles);
end

% 保存為.mat文件
x = x_frames;
y = y_frames;
save('laser.mat', 'x', 'y');
fprintf('XY座標數據已保存為 laser.mat\n');

% 動畫顯示（XY座標系）
fprintf('開始XY座標系動畫顯示...\n');
figure('Position', [100, 100, 800, 800]);

for i = 1:num_scans
    clf;
    
    current_x = x_frames(i, :);
    current_y = y_frames(i, :);
    current_distances = distances(frame_starts(i):min(frame_starts(i)+points_per_scan-1, end));
    
    % 過濾有效數據 (Minibot 實測上通常把 0 視為無效點)
    valid_idx = current_distances > 0.1 & current_distances < 10;
    x_valid = current_x(valid_idx);
    y_valid = current_y(valid_idx);
    
    % 繪製有效激光點
    if ~isempty(x_valid)
        scatter(x_valid, y_valid, 20, 'b', 'filled');
    end
    
    hold on;
    
    % 繪製機器人位置（原點）
    plot(0, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % 設置座標軸
    axis equal;
    grid on;
    xlim([-6, 6]);
    ylim([-6, 6]);
    
    title(sprintf('Minibot 激光掃描 - 第 %d/%d 幀', i, num_scans));
    xlabel('X (米)');
    ylabel('Y (米)');
    
    pause(0.05);
end

fprintf('動畫完成！\n');
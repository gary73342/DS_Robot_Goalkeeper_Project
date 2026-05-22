%% 
% =========================================================================
% ===      專題專用 Random Forest 訓練腳本 (多分類: 雜訊/球門/球)       ===
% =========================================================================
%
% 修改說明: 
% 1. 支援三元分類：0=雜訊, 1=球門(紙杯), 2=足球。
% 2. 升級 Gini Impurity 演算法，支援多類別機率平方和計算。
% 3. 升級 balance_data，確保三種類別在訓練時不會被單一類別（如雜訊）淹沒。
% 4. 引入 confusionchart 取代純文字的混淆矩陣，視覺化評估結果。
%
clear; clc; close all;

%% 1. 參數設定
% --- 檔案設定 ---
data_files = {
    'label_data_multi.mat'  % 請確認這是你的最新多類別數據檔名
};

% --- 模型參數 ---
N_TREES = 40;             % 隨機森林的樹數量 (考慮到多分類較複雜，稍微提升至 40 棵)
MAX_DEPTH = 12;           % 最大樹深，提升一點以容納多分類邏輯
MIN_LEAF_SIZE = 5;        % 葉節點最少樣本數 (避免過擬合)
DESIRED_NOISE_RATIO = 3;  % 雜訊比例上限 (相對於目標物)
global_frame_index_offset = 0; 

fprintf('=== 開始訓練守門員機器人 (三元分類 Random Forest) ===\n');
fprintf('使用的特徵: [1.距離, 2.標準差, 3.線性度, 4.曲率, 5.跨度, 6.點數, 7.梯度]\n');

%% 2. 提取檔案特徵 (Segment-Based)
X_all_segments = [];     
Y_all_segments = [];     
Frame_all_segments = []; 

for f_idx = 1:length(data_files)
    file_name = data_files{f_idx};
    fprintf('\n--- 正在處理檔案: %s ---\n', file_name);
    
    if ~exist(file_name, 'file')
        warning('檔案 %s 不存在，已跳過。', file_name);
        continue;
    end
    
    S = load(file_name);
    
    % 調用輔助函數提取特徵[cite: 13]
    [features_file, labels_file, frame_indices_file] = ...
        extract_segment_features_v2(S, global_frame_index_offset);
    
    % 附加到全局數據池
    X_all_segments = [X_all_segments; features_file];
    Y_all_segments = [Y_all_segments; labels_file];
    Frame_all_segments = [Frame_all_segments; frame_indices_file];
    
    num_frames_in_file = size(S.x_frames, 1);
    global_frame_index_offset = global_frame_index_offset + num_frames_in_file;
    
    fprintf('從 %s 提取了 %d 個 Segments。\n', file_name, size(features_file, 1));
end

fprintf('\n--- 特徵提取完成 ---\n');
fprintf('總 Segments 數: %d\n', length(Y_all_segments));
fprintf('標籤分佈:\n');
fprintf('  [0] 雜訊/牆壁: %d\n', sum(Y_all_segments == 0));
fprintf('  [1] 球門(紙杯): %d\n', sum(Y_all_segments == 1));
fprintf('  [2] 足球:       %d\n', sum(Y_all_segments == 2));

%% 3. 以 "幀" 為單位分割 訓練/測試集
all_unique_frames = unique(Frame_all_segments);
num_total_frames = length(all_unique_frames);

rng(42); 
train_frame_indices = all_unique_frames(randperm(num_total_frames, round(0.7 * num_total_frames)));

train_mask = ismember(Frame_all_segments, train_frame_indices);
test_mask = ~train_mask;

X_train_raw = X_all_segments(train_mask, :);
Y_train_raw = Y_all_segments(train_mask); 
X_test_raw = X_all_segments(test_mask, :);
Y_test_raw = Y_all_segments(test_mask);   

fprintf('\n--- 數據分割完成 ---\n');
fprintf('訓練集: %d, 測試集: %d\n', size(X_train_raw, 1), size(X_test_raw, 1));

%% 4. 標準化
% 移除常數特徵[cite: 13]
feature_variance = std(X_train_raw);
constant_features = find(feature_variance < 1e-8);
if ~isempty(constant_features)
    X_train_raw(:, constant_features) = [];
    X_test_raw(:, constant_features) = [];
end

% 計算 mu 和 sig[cite: 13]
mu = mean(X_train_raw);
sig = std(X_train_raw); sig(sig==0)=1;

% 應用標準化
X_train_std = (X_train_raw - mu) ./ sig;
X_test_std  = (X_test_raw - mu) ./ sig; 

fprintf('數據標準化完成。\n');

%% 5. 訓練多分類模型 (Custom Random Forest)
fprintf('\n=================================================\n');
fprintf('=== 訓練 Custom Random Forest (多分類進化版) ===\n');
fprintf('=================================================\n');

% 直接使用原始標籤 (0, 1, 2)，不再強制二元轉換
Y_train_final = Y_train_raw;
Y_test_final  = Y_test_raw;

% 對訓練集進行三類別平衡
[X_bal, Y_bal] = balance_data_multiclass(X_train_std, Y_train_final, DESIRED_NOISE_RATIO);

% 訓練自定義隨機森林
rf_model = custom_rf_train(X_bal, Y_bal, N_TREES, MAX_DEPTH, MIN_LEAF_SIZE);

% 預測
Y_train_pred = custom_rf_predict(rf_model, X_bal);
Y_test_pred  = custom_rf_predict(rf_model, X_test_std);

% 繪製混淆矩陣 (自定義無 Toolbox 視覺化)
figure('Name', '多分類混淆矩陣評估', 'Position', [100, 100, 1000, 450]);
subplot(1,2,1);
custom_confusion_matrix(Y_bal, Y_train_pred, {'Noise(0)', 'Goal(1)', 'Ball(2)'}, '訓練集');
subplot(1,2,2);
custom_confusion_matrix(Y_test_final, Y_test_pred, {'Noise(0)', 'Goal(1)', 'Ball(2)'}, '測試集');

% 終端機文字輸出
fprintf('\n');
print_multiclass_metrics(Y_bal, Y_train_pred, {'Noise(0)', 'Goal(1)', 'Ball(2)'}, '訓練集');
print_multiclass_metrics(Y_test_final, Y_test_pred, {'Noise(0)', 'Goal(1)', 'Ball(2)'}, '測試集');


% 儲存模型 (純文字格式，供 C++ 讀取)
output_file_txt = 'rf_model_multi_7feat.txt';
output_rf_to_txt(output_file_txt, rf_model, mu, sig);
fprintf('\n✅ 最終多分類模型已匯出到 %s\n', output_file_txt);


%% =========================================================================
% ===                       輔助函數 (Helper Functions)                 ===
% =========================================================================

% --- 輔助函數 1: 特徵提取 (維持原邏輯) ---
function [features_all_segments, labels_all_segments, frames_all_segments] = ...
    extract_segment_features_v2(S, global_frame_index_offset)
    
    x_frames = S.x_frames;
    y_frames = S.y_frames;
    T = S.segment_labels_table;
    
    num_frames = size(x_frames, 1);
    
    features_cell = cell(num_frames, 1);
    labels_cell = cell(num_frames, 1);
    frames_cell = cell(num_frames, 1);

    for f = 1:num_frames
        frame_features_list = {};
        frame_labels_list = {};
        
        x = x_frames(f, :);
        y = y_frames(f, :);
        valid_idx = find((x~=0 | y~=0) & ~isnan(x) & ~isnan(y));
        
        if isempty(valid_idx), continue; end
        
        xy_valid = [x(valid_idx)', y(valid_idx)'];
        [Seg, Si_n, S_n] = Segment(xy_valid);
        
        frame_labels_table = T(T.FrameIndex==f, :);
        current_frame_index = f + global_frame_index_offset;
        
        for s = 1:S_n
            pts_local_idx = Seg(1:Si_n(s), s);
            seg_x = xy_valid(pts_local_idx, 1);
            seg_y = xy_valid(pts_local_idx, 2);
            seg_r = sqrt(seg_x.^2 + seg_y.^2);
            
            num_points = length(seg_x);
            if num_points < 2, continue; end
            
            feat_r = mean(seg_r);
            feat_std = std(seg_r); if isnan(feat_std), feat_std = 0; end
            if num_points >= 3
                Xc = [seg_x - mean(seg_x), seg_y - mean(seg_y)];
                C = cov(Xc); eigvals = eig(C);
                if length(eigvals) >= 2, feat_lin = min(eigvals) / (sum(eigvals) + eps);
                else, feat_lin = 0; end
            else
                feat_lin = 0; 
            end
            
            local_curvatures = zeros(num_points, 1);
            if num_points >= 3
                for i = 2:num_points-1
                    x1=seg_x(i-1); y1=seg_y(i-1); x2=seg_x(i); y2=seg_y(i); x3=seg_x(i+1); y3=seg_y(i+1);
                    area2 = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
                    a = hypot(x2-x1, y2-y1); b = hypot(x3-x2, y3-y2); c = hypot(x3-x1, y3-y1);
                    R = (a*b*c)/(4*abs(area2)+eps);
                    local_curvatures(i) = 1/(R+eps);
                end
                feat_curv = mean(local_curvatures);
            else
                feat_curv = 0;
            end
            feat_span = hypot(seg_x(end)-seg_x(1), seg_y(end)-seg_y(1));
            feat_count = num_points;
            if num_points > 1, feat_grad = mean(abs(diff(seg_r))); else, feat_grad = 0; end
            
            segment_features = [feat_r, feat_std, feat_lin, feat_curv, feat_span, feat_count, feat_grad];
            
            label_row = frame_labels_table.SegmentIndex == s;
            Y_label = 0; 
            if any(label_row), Y_label = frame_labels_table.Label(label_row); end
            
            frame_features_list{end+1} = segment_features;
            frame_labels_list{end+1} = Y_label;
        end
        
        if ~isempty(frame_features_list)
            features_cell{f} = cell2mat(frame_features_list');
            labels_cell{f} = cell2mat(frame_labels_list');
            frames_cell{f} = repmat(current_frame_index, length(frame_labels_list), 1);
        end
    end
    features_all_segments = vertcat(features_cell{:});
    labels_all_segments = vertcat(labels_cell{:});
    frames_all_segments = vertcat(frames_cell{:});
end

% --- 輔助函數 2: 多類別資料平衡 (重寫版) ---
function [X_balanced, Y_balanced] = balance_data_multiclass(X_train, Y_train, desired_noise_ratio)
    rng(42);
    idx_0 = find(Y_train == 0); % 雜訊
    idx_1 = find(Y_train == 1); % 球門
    idx_2 = find(Y_train == 2); % 足球
    
    n_1 = length(idx_1);
    n_2 = length(idx_2);
    
    % 以目標物 (球與球門) 中數量較少者為基準，避免某一類完全主導
    base_n = max(min(n_1, n_2), 10); 
    
    % 決定各類別抽取數量 (雜訊多樣性高，允許依比例放大)
    n_0_desired = min(length(idx_0), round(base_n * desired_noise_ratio));
    n_1_desired = min(length(idx_1), round(base_n * 1.5)); 
    n_2_desired = min(length(idx_2), round(base_n * 1.5));
    
    sel_0 = idx_0(randperm(length(idx_0), n_0_desired));
    sel_1 = idx_1(randperm(length(idx_1), n_1_desired));
    sel_2 = idx_2(randperm(length(idx_2), n_2_desired));
    
    balanced_idx = [sel_0; sel_1; sel_2];
    balanced_idx = balanced_idx(randperm(length(balanced_idx))); % 打亂順序
    
    X_balanced = X_train(balanced_idx, :);
    Y_balanced = Y_train(balanced_idx);
end

% --- 輔助函數 3: 分群 Segment (維持原邏輯) ---
function [Seg,Si_n,S_n] = Segment(xy)
    x = xy(:,1); y = xy(:,2);
    threshold = 0.05;
    S_i = 1; S_n = 1;
    n0ind = 1:size(x,1);
    n_s = length(n0ind);
    
    Seg = zeros(n_s,n_s);
    Seg(1,1) = n0ind(1);
    
    for i = 2:n_s
        if sqrt((x(n0ind(i))-x(n0ind(i-1)))^2 + (y(n0ind(i))-y(n0ind(i-1)))^2) < threshold
            S_i = S_i + 1;
            Seg(S_i,S_n) = n0ind(i);
        else
            S_n = S_n + 1;
            S_i = 1;
            Seg(S_i,S_n) = n0ind(i);
        end
    end
    
    Si_n = zeros(S_n,1);
    for j = 1:S_n
        Si_n(j) = sum(Seg(:,j)~=0);
    end
end

% =========================================================================
% ===                 手刻版 Random Forest 核心函數                     ===
% =========================================================================

function model = custom_rf_train(X, Y, n_trees, max_depth, min_samples_leaf)
    [N, D] = size(X);
    num_features_split = max(1, round(sqrt(D))); 
    model.trees = cell(n_trees, 1);
    
    for t = 1:n_trees
        % Bagging: 抽樣後放回 (Bootstrap sampling)[cite: 13]
        idx = randi(N, N, 1);
        X_boot = X(idx, :);
        Y_boot = Y(idx);
        
        model.trees{t} = build_tree(X_boot, Y_boot, max_depth, min_samples_leaf, 1, num_features_split);
        fprintf('  已完成第 %d / %d 棵樹訓練\n', t, n_trees);
    end
end

function node = build_tree(X, Y, max_depth, min_leaf, depth, m_try)
    node = struct('is_leaf', false, 'feature', 0, 'threshold', 0, 'left', [], 'right', [], 'prediction', -1);
    
    if depth >= max_depth || length(Y) <= min_leaf || length(unique(Y)) == 1
        node.is_leaf = true;
        node.prediction = mode(Y);
        return;
    end
    
    [N, D] = size(X);
    best_gini = inf; best_feat = 0; best_thr = 0;
    
    features = randperm(D, m_try);
    
    for f = features
        vals = unique(X(:, f));
        if length(vals) < 2, continue; end
        
        all_thr_cand = (vals(1:end-1) + vals(2:end)) / 2;
        n_candidates = numel(all_thr_cand);
        
        n_thresholds = min(20, n_candidates); 
        if n_candidates > n_thresholds
            indices = randperm(n_candidates, n_thresholds);
            thr_cand = all_thr_cand(indices); 
        else
            thr_cand = all_thr_cand; 
        end
        
        % --- 多分類 Gini 不純度計算核心 ---
        edges = [-0.5, 0.5, 1.5, 2.5]; % 分別抓出 0, 1, 2 的數量
        for t = 1:length(thr_cand)
            thr = thr_cand(t);
            idx_L = X(:, f) < thr;
            idx_R = ~idx_L;
            
            n_L = sum(idx_L); n_R = sum(idx_R);
            if n_L == 0 || n_R == 0, continue; end
            
            counts_L = histcounts(Y(idx_L), edges);
            p_L = counts_L / n_L;
            gini_L = 1 - sum(p_L.^2);
            
            counts_R = histcounts(Y(idx_R), edges);
            p_R = counts_R / n_R;
            gini_R = 1 - sum(p_R.^2);
            
            gini_split = (n_L * gini_L + n_R * gini_R) / N;
            
            if gini_split < best_gini
                best_gini = gini_split;
                best_feat = f;
                best_thr = thr;
            end
        end
    end
    
    if best_feat == 0
        node.is_leaf = true;
        node.prediction = mode(Y);
        return;
    end
    
    node.feature = best_feat;
    node.threshold = best_thr;
    idx_L = X(:, best_feat) < best_thr;
    idx_R = ~idx_L;
    
    node.left = build_tree(X(idx_L, :), Y(idx_L), max_depth, min_leaf, depth+1, m_try);
    node.right = build_tree(X(idx_R, :), Y(idx_R), max_depth, min_leaf, depth+1, m_try);
end

function y_pred = custom_rf_predict(model, X)
    N = size(X, 1);
    n_trees = length(model.trees);
    tree_preds = zeros(N, n_trees);
    
    for t = 1:n_trees
        for i = 1:N
            tree_preds(i, t) = predict_single_tree(model.trees{t}, X(i, :));
        end
    end
    y_pred = mode(tree_preds, 2);
end

function y = predict_single_tree(node, x)
    if node.is_leaf
        y = node.prediction;
        return;
    end
    if x(node.feature) < node.threshold
        y = predict_single_tree(node.left, x);
    else
        y = predict_single_tree(node.right, x);
    end
end

% --- 輔助函數 5 & 6: 匯出模型 (維持原格式，自動支援多分類) ---[cite: 13]
function output_rf_to_txt(filename, model, mu, sig)
    fid = fopen(filename, 'w');
    fprintf(fid, '### Custom Random Forest Model (7 Features) ###\n');
    fprintf(fid, '# Features: [r, std, lin, curv, span, count, grad]\n');
    fprintf(fid, '# N_TREES: %d\n', length(model.trees));
    fprintf(fid, '# Feature Dimension: %d\n', length(mu));
    
    fprintf(fid, '# SECTION 1: Mu/Sigma\n');
    fprintf(fid, '%.8f ', mu); fprintf(fid, '\n');
    fprintf(fid, '%.8f ', sig); fprintf(fid, '\n');
    
    fprintf(fid, '# SECTION 2: Trees (Pre-order Traversal)\n');
    fprintf(fid, '# Format: is_leaf feature threshold prediction\n');
    
    for t = 1:length(model.trees)
        fprintf(fid, 'TREE %d\n', t);
        write_tree_node(fid, model.trees{t});
    end
    fclose(fid);
end

function write_tree_node(fid, node)
    if node.is_leaf
        fprintf(fid, '1 0 0.000000 %d\n', node.prediction);
    else
        fprintf(fid, '0 %d %.8f 0\n', node.feature, node.threshold);
        write_tree_node(fid, node.left);
        write_tree_node(fid, node.right);
    end
end

% --- 輔助函數: 手刻版視覺化混淆矩陣 (無 Toolbox 依賴) ---
function custom_confusion_matrix(y_true, y_pred, class_names, fig_title)
    % 取得唯一類別 (0, 1, 2)
    classes = unique([y_true(:); y_pred(:)]);
    num_classes = length(classes);
    
    % 初始化混淆矩陣
    conf_mat = zeros(num_classes, num_classes);
    
    % 計算混淆矩陣數值
    for i = 1:length(y_true)
        row_idx = find(classes == y_true(i));
        col_idx = find(classes == y_pred(i));
        if ~isempty(row_idx) && ~isempty(col_idx)
            conf_mat(row_idx, col_idx) = conf_mat(row_idx, col_idx) + 1;
        end
    end
    
    % 計算 Row-normalized (Recall) 以便設定顏色深淺
    % 加 eps 防止除以零
    row_sum = sum(conf_mat, 2) + eps; 
    conf_mat_norm = conf_mat ./ row_sum;
    
    % 繪製熱力圖 (Heatmap)
    imagesc(conf_mat_norm);
    colormap(flipud(hot)); % 使用熱力圖配色，翻轉讓深色代表高數值
    caxis([0 1]); % 限制顏色範圍 0~1
    colorbar;
    
    % 設定座標軸標籤
    set(gca, 'XTick', 1:num_classes, 'XTickLabel', class_names);
    set(gca, 'YTick', 1:num_classes, 'YTickLabel', class_names);
    xlabel('Predicted Class (預測類別)', 'FontWeight', 'bold');
    ylabel('True Class (實際類別)', 'FontWeight', 'bold');
    title(fig_title, 'FontSize', 12);
    
    % 在方格內印出實際數量與百分比
    for r = 1:num_classes
        for c = 1:num_classes
            val = conf_mat(r, c);
            pct = conf_mat_norm(r, c) * 100;
            
            % 如果背景顏色太深，字體用白色；太淺用黑色
            if conf_mat_norm(r, c) > 0.5
                text_color = 'w';
            else
                text_color = 'k';
            end
            
            % 顯示文字
            str = sprintf('%d\n(%.1f%%)', val, pct);
            text(c, r, str, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'Color', text_color, 'FontWeight', 'bold');
        end
    end
    
    % 加上框線讓格子更明顯
    hold on;
    for i = 0.5:1:num_classes+0.5
        plot([0.5, num_classes+0.5], [i, i], 'k-', 'LineWidth', 1);
        plot([i, i], [0.5, num_classes+0.5], 'k-', 'LineWidth', 1);
    end
    hold off;
end

% --- 輔助函數: 終端機多分類效能評估輸出 ---
function print_multiclass_metrics(y_true, y_pred, class_names, dataset_name)
    classes = unique([y_true(:); y_pred(:)]);
    num_classes = length(classes);
    conf_mat = zeros(num_classes, num_classes);
    
    % 計算混淆矩陣
    for i = 1:length(y_true)
        row_idx = find(classes == y_true(i));
        col_idx = find(classes == y_pred(i));
        if ~isempty(row_idx) && ~isempty(col_idx)
            conf_mat(row_idx, col_idx) = conf_mat(row_idx, col_idx) + 1;
        end
    end
    
    % 輸出混淆矩陣
    fprintf('=== %s 效能評估 ===\n', dataset_name);
    fprintf('Confusion Matrix (列=實際, 欄=預測):\n');
    fprintf('         ');
    for c = 1:num_classes
        fprintf('%10s ', class_names{c});
    end
    fprintf('\n');
    
    for r = 1:num_classes
        fprintf('%8s ', class_names{r});
        for c = 1:num_classes
            fprintf('%10d ', conf_mat(r, c));
        end
        fprintf('\n');
    end
    fprintf('---------------------------------------------------\n');
    
    % 計算並輸出 Acc, Precision, Recall
    total_correct = sum(diag(conf_mat));
    total_samples = sum(conf_mat(:));
    overall_acc = total_correct / total_samples;
    
    fprintf('整體準確率 (Overall Accuracy): %.2f%%\n\n', overall_acc * 100);
    
    fprintf('%-12s | %-12s | %-12s | %-12s\n', '類別 (Class)', 'Precision(精確率)', 'Recall(召回率)', 'F1-Score');
    fprintf('---------------------------------------------------\n');
    
    for k = 1:num_classes
        % Precision = TP / (TP + FP) (這一欄的總和)
        tp = conf_mat(k, k);
        col_sum = sum(conf_mat(:, k));
        prec = tp / (col_sum + eps);
        
        % Recall = TP / (TP + FN) (這一列的總和)
        row_sum = sum(conf_mat(k, :));
        rec = tp / (row_sum + eps);
        
        % F1-Score = 2 * (Prec * Rec) / (Prec + Rec)
        f1 = 2 * (prec * rec) / (prec + rec + eps);
        
        fprintf('%-12s | %10.2f%% | %10.2f%% | %10.2f\n', ...
            class_names{k}, prec * 100, rec * 100, f1);
    end
    fprintf('===================================================\n\n');
end
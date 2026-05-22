#include "RandomForest.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

// 實作 Node 建構子
Node::Node() : is_leaf(false), feature_idx(-1), threshold(0.0f), prediction(-1), left(nullptr), right(nullptr) {}

// 實作 RandomForest 建構子 (裝甲防彈版)
RandomForest::RandomForest(const string& model_path) {
    ifstream infile(model_path);
    if (!infile.is_open()) {
        cerr << "[ERROR] 無法開啟模型檔案: " << model_path << endl;
        return;
    }

    cout << "[DEBUG] 成功開啟檔案，開始解析..." << endl;
    string line;
    int tree_count = 0;

    while (getline(infile, line)) {
        // 🚨 關鍵防護 1：強制清除 Windows 的 \r 符號，防止字串比對失敗
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.find("SECTION 1") != string::npos) {
            cout << "[DEBUG] 找到 SECTION 1，準備讀取 Mu/Sigma..." << endl;
            
            // 讀取 Mu
            if (getline(infile, line)) {
                stringstream ss_mu(line);
                float val;
                while (ss_mu >> val) mu.push_back(val);
                cout << "[DEBUG] Mu 讀取完畢，大小: " << mu.size() << endl;
            }
            
            // 讀取 Sigma
            if (getline(infile, line)) {
                stringstream ss_sig(line);
                float val;
                while (ss_sig >> val) sig.push_back(val);
                cout << "[DEBUG] Sigma 讀取完畢，大小: " << sig.size() << endl;
            }
        } 
        else if (line.find("TREE") != string::npos) {
            Node* root = parseNode(infile);
            if (root != nullptr) {
                trees.push_back(root);
                tree_count++;
            }
        }
    }
    cout << "已成功載入隨機森林，總計 " << trees.size() << " 棵樹。" << endl;
}

// 實作 Node 解析 (裝甲防彈版)
Node* RandomForest::parseNode(ifstream& infile) {
    string line;
    if (!getline(infile, line)) return nullptr;
    
    // 🚨 關鍵防護 2：清除 \r
    if (!line.empty() && line.back() == '\r') line.pop_back();
    
    // 🚨 關鍵防護 3：如果遇到空行，自動跳過並讀取下一行
    if (line.empty()) return parseNode(infile);

    stringstream ss(line);
    int is_leaf, feat, pred;
    float thr;
    
    // 🚨 關鍵防護 4：如果遇到亂碼或格式不對，報錯而不是死迴圈
    if (!(ss >> is_leaf >> feat >> thr >> pred)) {
        cerr << "[ERROR] 節點解析失敗，讀到異常內容: " << line << endl;
        return nullptr;
    }

    Node* node = new Node();
    node->is_leaf = (is_leaf == 1);
    node->feature_idx = feat - 1; // MATLAB 1-based 轉 C++ 0-based
    node->threshold = thr;
    node->prediction = pred;

    if (!node->is_leaf) {
        node->left = parseNode(infile);
        node->right = parseNode(infile);
    }
    return node;
}
// 實作單棵樹推論
int RandomForest::predictSingleTree(Node* node, const vector<float>& x) {
    if (node->is_leaf) return node->prediction;
    if (x[node->feature_idx] < node->threshold) {
        return predictSingleTree(node->left, x);
    } else {
        return predictSingleTree(node->right, x);
    }
}
// 實作多數決推論
int RandomForest::predict(const vector<float>& raw_features) {
    // 防護 1：特徵數量不對
    if (raw_features.size() != num_features) {
        std::cerr << "[ERROR] 收到特徵數: " << raw_features.size() << "，預期: " << num_features << std::endl;
        return -1;
    }

    // 🚨 防護 2：揪出未初始化的陣列 (防止 Segfault)
    if (mu.size() != num_features || sig.size() != num_features) {
        std::cerr << "[FATAL] 模型參數 mu 或 sig 未正確載入！" << std::endl;
        std::cerr << "mu size: " << mu.size() << ", sig size: " << sig.size() << std::endl;
        return -1; 
    }

    // 1. 標準化特徵 (現在這裡絕對安全了)
    vector<float> x_std(num_features);
    for (int i = 0; i < num_features; ++i) {
        float s = (sig[i] == 0.0f) ? 1e-6f : sig[i];  // ← 用 local 變數，不動 sig
        x_std[i] = (raw_features[i] - mu[i]) / s;
    }

    // 2. 投票
    map<int, int> votes;
    for (auto root : trees) {
        int p = predictSingleTree(root, x_std);
        votes[p]++;
    }

    // 3. 多數決 (Majority Vote)
    int best_class = -1;
    int max_votes = -1;
    for (auto const& pair : votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            best_class = pair.first;
        }
    }
    return best_class;
}


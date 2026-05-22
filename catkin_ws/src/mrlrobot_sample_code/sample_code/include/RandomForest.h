#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <string>
#include <map>

// 定義樹節點結構
struct Node {
    bool is_leaf;
    int feature_idx;
    float threshold;
    int prediction;
    Node *left, *right;

    Node(); // 建構子宣告
};

class RandomForest {
private:
    std::vector<float> mu, sig;
    std::vector<Node*> trees;
    int num_features = 7; // [r, std, lin, curv, span, count, grad]

    // 內部輔助函數宣告
    Node* parseNode(std::ifstream& infile);
    int predictSingleTree(Node* node, const std::vector<float>& x);

public:
    // 公開介面宣告
    RandomForest(const std::string& model_path);
    int predict(const std::vector<float>& raw_features);
};

#endif // RANDOMFOREST_H
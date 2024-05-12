#ifndef NEXTSOFTWARE_PARTITIONPARAMS_H
#define NEXTSOFTWARE_PARTITIONPARAMS_H

#include <iostream>
#include <vector>
#include <map>

typedef std::pair<int, int> cushape;

static std::vector<std::string> ShapeList = {
  "32x8",
  "8x32",
  "16x16",
  "16x32",
  "32x16",
  "32x32",
};

enum MyFastMode {
  FASTER,
  FAST,
  MEDIUM,
};
static const MyFastMode MyFastPartitionMode = FASTER; // 用于控制阈值，

static const int FastMode_K = 2;

struct BasicParams {

};

struct PredictionParams {
  int qp;
  int k;

  double threshold_NS_level1;
  double threshold_NS_level2;
  double threshold_QT_level1;
  double threshold_QT_level2;
  double threshold_MTT;

  std::string cnn_model_root_path;
  std::string lgbm_model_root_path;
  std::vector<cushape> cushape_list;
  std::vector<std::string> shape_list;

  PredictionParams();
  PredictionParams(double th1, double th2, double th3, double th4, double th5);
  void set_qp(int QP);
};

static std::map<MyFastMode, PredictionParams> FastPartitionModeParams = {
  {FASTER, PredictionParams(0.9, 0.9, 0.9, 0.9, 0.9)},
  {FAST, PredictionParams(0.7, 0.7, 0.7, 0.7, 0.7)},
  {MEDIUM, PredictionParams(0.8, 0.8, 0.8, 0.8, 0.8)}
};



#endif // !NEXTSOFTWARE_PARTITIONPARAMS_H

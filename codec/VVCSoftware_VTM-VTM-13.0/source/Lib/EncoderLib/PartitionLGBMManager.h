#ifndef NEXTSOFTWARE_PARTITIONLGBMMANAGER
#define NEXTSOFTWARE_PARTITIONLGBMMANAGER

#include "PartitionParams.h"
#include "Unit.h"

#include <map>
#include <algorithm>
#include <lightgbm/c_api.h>

//C_API_DTYPE_FLOAT64 // 双精度， C++ 中的 api 接受的数据类型只有两种：单精度 float 和 双精度 double

class LGBMPredictionManager {
private:
  PredictionParams* lgbm_params_;
  std::map<std::string, BoosterHandle> lgbm_models_; // 可能会针对不同尺寸训练模型
  std::map<std::string, FastConfigHandle> lgbm_fast_handles_;

  std::map<std::string, int> num_features_; 
  std::map<std::string, int64_t> num_classes_; // 可能不同尺寸会用到不同数量的特征、类别，因此使用map，进行存储
  std::map<std::string, int> out_num_iterations_; //  用于返回模型的迭代次数（树的数量）。这是一个输出参数，用于获取模型中的迭代次数。

public:
  LGBMPredictionManager(int qp);
  void load_model();
  void predict(const UnitArea& cuArea, std::vector<double>& input_features, std::vector<double>& output_prob);
  void get_lgbm_params(const UnitArea& cuArea, int& num_features, int& num_classes);
};



#endif // !NEXTSOFTWARE_PARTITIONLGBMMANAGER

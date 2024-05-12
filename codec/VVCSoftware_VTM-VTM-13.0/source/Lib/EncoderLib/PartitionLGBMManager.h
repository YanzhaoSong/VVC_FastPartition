#ifndef NEXTSOFTWARE_PARTITIONLGBMMANAGER
#define NEXTSOFTWARE_PARTITIONLGBMMANAGER

#include "PartitionParams.h"
#include "Unit.h"

#include <map>
#include <algorithm>
#include <lightgbm/c_api.h>

//C_API_DTYPE_FLOAT64 // ˫���ȣ� C++ �е� api ���ܵ���������ֻ�����֣������� float �� ˫���� double

class LGBMPredictionManager {
private:
  PredictionParams* lgbm_params_;
  std::map<std::string, BoosterHandle> lgbm_models_; // ���ܻ���Բ�ͬ�ߴ�ѵ��ģ��
  std::map<std::string, FastConfigHandle> lgbm_fast_handles_;

  std::map<std::string, int> num_features_; 
  std::map<std::string, int64_t> num_classes_; // ���ܲ�ͬ�ߴ���õ���ͬ������������������ʹ��map�����д洢
  std::map<std::string, int> out_num_iterations_; //  ���ڷ���ģ�͵ĵ�������������������������һ��������������ڻ�ȡģ���еĵ���������

public:
  LGBMPredictionManager(int qp);
  void load_model();
  void predict(const UnitArea& cuArea, std::vector<double>& input_features, std::vector<double>& output_prob);
  void get_lgbm_params(const UnitArea& cuArea, int& num_features, int& num_classes);
};



#endif // !NEXTSOFTWARE_PARTITIONLGBMMANAGER

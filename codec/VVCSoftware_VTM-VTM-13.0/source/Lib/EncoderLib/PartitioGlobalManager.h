#ifndef NEXTSOFTWARE_PARTITIONGLOBALMANAGER_H
#define NEXTSOFTWARE_PARTITIONGLOBALMANAGER_H

#include "PartitionUtils.h"
#include "PartitionParams.h"
#include "PartitionCNNManager.h"
#include "PartitionLGBMManager.h"
#include "PartitionFeatureExtraction.h"
#include "PartitionImageDataManager.h"

#include "EncModeCtrl.h"
#include "Unit.h"
#include "UnitPartitioner.h"

class PartitionGlobalManager {
private:
  PredictionParams* params_;
  std::vector<double> cnn_prob_;
  std::vector<double> lgbm_prob_;
  std::vector<EncTestModeType> mode_list_;
  std::vector<double> features_;
  bool can_QT_; // 用于标识当前是否能够进行QT划分，在 prepare_for_lgbm 中进行设置
  std::string shape_;
  int qp;

  double Threshold_L1_NS;
  double Threshold_L1_QT;
  double Threshold_L1_MTT;
  double Threshold_L2_NS;
  double Threshold_L3_NS;
  double Threshold_L4_NS;
  double Threshold_L2_MTT;

public:
  bool only_use_cnn_; // 标志，标识后面是否需要在进行LGBM的预测
  int cnn_pred_mode_;
  int lgbm_pred_mode_;

  CNNPredictionManager* cnn_predictor_;
  LGBMPredictionManager* lgbm_predictor_; // 不用智能指针，而是使用 new 分配内存，new 分配的内存并不会消失。

  CUImageDataManager cu_image_data_;
  CTUImageDataManager* ctu_image_data_;
  LGBMFeaturesManager lgbm_features_manager_;


  PartitionGlobalManager(CNNPredictionManager* cnn_predictor, LGBMPredictionManager* lgbm_predictor);
  void prepare_for_cnn(const UnitArea& cuArea, CTUImageDataManager& ctu_data); // 为CNN 预测进行准备，比如图像数据获取、预处理 等操作
  void cnn_predict(const UnitArea& cuArea);
  void reset_testmodes_with_cnn_result(std::vector<EncTestMode>& testModes);

  // 该函数的调用时间应该为 帧内模式测试完成之后。
  void prepare_for_lgbm(const CodingStructure& cs, Partitioner& partitioner, const ComprCUCtx& cuECtx, const CodingUnit& cuArea); // 为 LGBM 预测进行准备，比如 特征获取、特征的各种操作使其和训练时输入特征一致。
  void lgbm_predict(const UnitArea& cuArea);
  void reset_testmodes_with_lgbm_result(std::vector<EncTestMode>& testModes);

  void reset_testmodes(std::vector<EncTestMode>& testModes, std::vector<bool>& mask);

  void set_qp(const CodingStructure& cs) { params_->set_qp(cs.baseQP); qp = cs.baseQP; }
  void set_thresholds(const std::vector<double>& thresholds);

  const std::vector<double>& get_cnn_prob() const { return cnn_prob_; };
};



#endif // !NEXTSOFTWARE_PARTITIONGLOBALMANAGER_H

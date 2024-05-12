#ifndef NEXTSOFTWARE_PARTITIONCNNMANAGER_H
#define NEXTSOFTWARE_PARTITIONCNNMANAGER_H

#include "Unit.h"

#include "PartitionUtils.h"
#include "PartitionParams.h"
#include "PartitionImageDataManager.h"
#include <torch/script.h>
#include <map>


class CNNPredictionManager {
private:
  PredictionParams* cnn_params_;
  std::map<std::string, torch::jit::script::Module> cnn_models_;

public:

  CNNPredictionManager(int qp);
  void load_model();
  void get_input_tensor(const std::unique_ptr<cv::Mat>& src, std::vector<torch::jit::IValue>& dst);
  void predict(const UnitArea& cuArea, const std::unique_ptr<cv::Mat>& src, std::vector<double>& output_prob);
};


#endif // !NEXTSOFTWARE_PARTITIONCNNMANAGER_H

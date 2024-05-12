#include "PartitionLGBMManager.h"
#include "PartitionUtils.h"

#include <string>

LGBMPredictionManager::LGBMPredictionManager(int qp) {
  lgbm_params_ = &FastPartitionModeParams[MyFastPartitionMode];
  lgbm_params_->qp = qp;
  lgbm_params_->shape_list = {
    "32x32",
    "32x16",
    "16x32",
    "32x8",
    "8x32",
    "16x16",
  };

  num_features_ = {
    {"32x32", 96},  // 96， 58,  42, 118
    {"32x16", 96},
    {"16x32", 96},
    {"16x16", 96},
    {"32x8", 96},
    {"8x32", 96},
  };

  num_classes_ = {
    {"32x32", 6},
    {"32x16", 5},
    {"16x32", 5},
    {"16x16", 5},
    {"32x8", 4},
    {"8x32", 4},
  };

  out_num_iterations_ = {
    {"32x32", 0},
    {"32x16", 0},
    {"16x32", 0},
    {"16x16", 0},
    {"32x8", 0},
    {"8x32", 0},
  };

  load_model();
}

void LGBMPredictionManager::load_model() {
  for (auto& shape : lgbm_params_->shape_list) {
    std::string model_path = lgbm_params_->lgbm_model_root_path + "QP" + std::to_string(lgbm_params_->qp) + "_" + shape + ".txt"; // 路径样式：models\\lgbm_models\\QP37_32x32.txt

    // 首先判断模型是否存在
    if (!is_file_exist(model_path)) {
      std::cout << "lgbm model does not exist: " << model_path << std::endl;
      exit(-1);
    }
    
    try
    {
      lgbm_models_[shape] = BoosterHandle{};
      LGBM_BoosterCreateFromModelfile(
        model_path.c_str(),  // 模型的存储路径
        &out_num_iterations_[shape],  // 模型的 iterations
        &lgbm_models_[shape] // 输出模型的地址
      );

      // 转换为 FastConfigHandle
      LGBM_BoosterPredictForMatSingleRowFastInit(
        lgbm_models_[shape], // 模型 BoosterHandle
        C_API_PREDICT_NORMAL,  // predict_type 
        0, // start_iteration 
        out_num_iterations_[shape], // num_iteration 
        C_API_DTYPE_FLOAT64, // data_type
        num_features_[shape], // ncols
        "", // Additional parameters 
        &lgbm_fast_handles_[shape]
      );
    }
    catch (const std::exception&)
    {
      std::cout << "error loading the lgbm model." << std::endl;
    }
  }
}

void LGBMPredictionManager::get_lgbm_params(const UnitArea& cuArea, int& num_features, int& num_classes) {
  num_features = num_features_[get_shape(cuArea)];
  num_classes = num_classes_[get_shape(cuArea)];
}

void LGBMPredictionManager::predict(const UnitArea& cuArea, std::vector<double>& input_features, std::vector<double>& output_prob) {
  std::string shape = get_shape(cuArea);

  //std::cout << input_features.size() << std::endl;
  //exit(0);

  void* in_p = static_cast<void*>(input_features.data());
  std::vector<double> out{ 0, 0, 0, 0, 0, 0 };
  double* out_result = static_cast<double*>(out.data());
  int64_t out_len;

  //LGBM_BoosterPredictForMat(
  //  lgbm_models_[shape],
  //  in_p,
  //  C_API_DTYPE_FLOAT64, // 这里必须和输入的数据类型匹配，否则会出问题。
  //  1, // nrow
  //  num_features_[shape], // ncol, 即特征的数量
  //  1, // is_row_major
  //  C_API_PREDICT_NORMAL, // predict_type
  //  0, // start_iteration
  //  out_num_iterations_[shape], // num_iterations  out_num_iterations_[shape]
  //  "",
  //  &out_len,
  //  out_result
  //);

  // 使用 LGBM_BoosterPredictForMatSingleRowFast
  LGBM_BoosterPredictForMatSingleRowFast(
    lgbm_fast_handles_[shape],
    in_p,
    &out_len,
    out_result
  );

  if (shape == "32x32") {
    output_prob = std::vector<double>(out_result, out_result + 6);
  }
  else if (shape == "32x16" || shape == "16x32") { // 0, 2, 3, 4, 5
    auto tmp = std::vector<double>(out_result, out_result + 5);
    output_prob = {tmp[0], 0, tmp[1], tmp[2], tmp[3], tmp[4]};
  }
  else if (shape == "32x8" || shape == "16x8") { // 0, 2, 3, 5
    auto tmp = std::vector<double>(out_result, out_result + 4);
    output_prob = { tmp[0], 0, tmp[1], tmp[2], 0, tmp[3] };
  }
  else if (shape == "8x32" || shape == "8x16") { // 0, 2, 3, 4
    auto tmp = std::vector<double>(out_result, out_result + 4);
    output_prob = { tmp[0], 0, tmp[1], tmp[2], tmp[3], 0 };
  }
  else if (shape == "16x16") { // 0, 2, 3, 4, 5 , 原本应该存在模式1， 但是新的LGBM 模型去除了该模式的预测
    auto tmp = std::vector<double>(out_result, out_result + 5);
    output_prob = { tmp[0], 0, tmp[1], tmp[2], tmp[3], tmp[4] };
  }
  else if (shape == "8x8") {
    auto tmp = std::vector<double>(out_result, out_result + 3);
    output_prob = { tmp[0], 0, tmp[1], tmp[2], 0, 0 };
  }
  
  //std::cout << "LGBM: ";
  //print_vector(output_prob);
}

/*
各个参数的含义：
  handle: 模型句柄，由 LGBM_BoosterCreate 或 LGBM_BoosterCreateFromModelfile 等函数创建得到。

  data: 包含要进行预测的单行数据的指针。数据的格式和类型由 data_type 参数指定。

  data_type: 数据的类型。可以是以下值之一：

  ncol: 数据的列数，即特征的数量。

  is_row_major: 数据的存储顺序。可以是以下值之一：
    0：列主序（column-major order）。
    1：行主序（row-major order）。

  predict_type: 预测的类型。可以是以下值之一：

  start_iteration: 预测的起始迭代次数。

  num_iteration: 预测的迭代次数。它指定了从 start_iteration 开始要使用的迭代次数的数量。在某些情况下，你可能希望只使用模型的前几个迭代，以获得更快的预测速度。

  parameter: 预测的参数，通常设置为 nullptr。

  out_len: 输出参数，返回预测结果的长度。

  out_result: 输出参数，用于存储预测结果。
*/
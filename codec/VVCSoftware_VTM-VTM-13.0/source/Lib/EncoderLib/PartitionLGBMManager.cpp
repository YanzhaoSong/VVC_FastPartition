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
    {"32x32", 96},  // 96�� 58,  42, 118
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
    std::string model_path = lgbm_params_->lgbm_model_root_path + "QP" + std::to_string(lgbm_params_->qp) + "_" + shape + ".txt"; // ·����ʽ��models\\lgbm_models\\QP37_32x32.txt

    // �����ж�ģ���Ƿ����
    if (!is_file_exist(model_path)) {
      std::cout << "lgbm model does not exist: " << model_path << std::endl;
      exit(-1);
    }
    
    try
    {
      lgbm_models_[shape] = BoosterHandle{};
      LGBM_BoosterCreateFromModelfile(
        model_path.c_str(),  // ģ�͵Ĵ洢·��
        &out_num_iterations_[shape],  // ģ�͵� iterations
        &lgbm_models_[shape] // ���ģ�͵ĵ�ַ
      );

      // ת��Ϊ FastConfigHandle
      LGBM_BoosterPredictForMatSingleRowFastInit(
        lgbm_models_[shape], // ģ�� BoosterHandle
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
  //  C_API_DTYPE_FLOAT64, // ���������������������ƥ�䣬���������⡣
  //  1, // nrow
  //  num_features_[shape], // ncol, ������������
  //  1, // is_row_major
  //  C_API_PREDICT_NORMAL, // predict_type
  //  0, // start_iteration
  //  out_num_iterations_[shape], // num_iterations  out_num_iterations_[shape]
  //  "",
  //  &out_len,
  //  out_result
  //);

  // ʹ�� LGBM_BoosterPredictForMatSingleRowFast
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
  else if (shape == "16x16") { // 0, 2, 3, 4, 5 , ԭ��Ӧ�ô���ģʽ1�� �����µ�LGBM ģ��ȥ���˸�ģʽ��Ԥ��
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
���������ĺ��壺
  handle: ģ�;������ LGBM_BoosterCreate �� LGBM_BoosterCreateFromModelfile �Ⱥ��������õ���

  data: ����Ҫ����Ԥ��ĵ������ݵ�ָ�롣���ݵĸ�ʽ�������� data_type ����ָ����

  data_type: ���ݵ����͡�����������ֵ֮һ��

  ncol: ���ݵ���������������������

  is_row_major: ���ݵĴ洢˳�򡣿���������ֵ֮һ��
    0��������column-major order����
    1��������row-major order����

  predict_type: Ԥ������͡�����������ֵ֮һ��

  start_iteration: Ԥ�����ʼ����������

  num_iteration: Ԥ��ĵ�����������ָ���˴� start_iteration ��ʼҪʹ�õĵ�����������������ĳЩ����£������ϣ��ֻʹ��ģ�͵�ǰ�����������Ի�ø����Ԥ���ٶȡ�

  parameter: Ԥ��Ĳ�����ͨ������Ϊ nullptr��

  out_len: �������������Ԥ�����ĳ��ȡ�

  out_result: ������������ڴ洢Ԥ������
*/
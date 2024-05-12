#include "PartitioGlobalManager.h"

//// ��ʼ�� static ��Ա
//CNNPredictionManager PartitionGlobalManager::cnn_predictor_ = CNNPredictionManager();
//LGBMPredictionManager PartitionGlobalManager::lgbm_predictor_ = LGBMPredictionManager();


PartitionGlobalManager::PartitionGlobalManager(CNNPredictionManager* cnn_predictor, LGBMPredictionManager* lgbm_predictor)
  : params_ { &FastPartitionModeParams[MyFastPartitionMode] }
  , cnn_predictor_{cnn_predictor}
  , lgbm_predictor_{ lgbm_predictor }
  , cnn_prob_ {0.0}
  , lgbm_prob_ {0.0}
  , cnn_pred_mode_{-1}
  , lgbm_pred_mode_{-1}
  , cu_image_data_ {}
  , lgbm_features_manager_ {}
  , features_ {0.0}
  , only_use_cnn_ {false}
  , can_QT_ {true}
  , shape_{""}
{
  mode_list_ = { 
    ETM_INTRA,
    ETM_SPLIT_QT,
    ETM_SPLIT_BT_H,
    ETM_SPLIT_BT_V,
    ETM_SPLIT_TT_H,
    ETM_SPLIT_TT_V, 
  };
}

void PartitionGlobalManager::set_thresholds(const std::vector<double>& thresholds) {
  Threshold_L1_NS = thresholds[0];
  Threshold_L1_QT = thresholds[1];
  Threshold_L1_MTT = thresholds[2];
  Threshold_L2_NS = thresholds[3];
  Threshold_L2_MTT = thresholds[4];
}

void PartitionGlobalManager::prepare_for_cnn(const UnitArea& cuArea, CTUImageDataManager& ctu_data) {
  cu_image_data_.init_cu_data(cuArea, ctu_data);
  ctu_image_data_ = &ctu_data;
  shape_ = get_shape(cuArea);

  // ��Բ�ͬ�ߴ�����ж�
  //auto shape_list_only_use_cnn = { "32x16", "16x32" };
  //auto flag = (std::find(shape_list_only_use_cnn.begin(), shape_list_only_use_cnn.end(), get_shape(cuArea)) != shape_list_only_use_cnn.end());
  //if (flag) {
  //  only_use_cnn_ = true;
  //}
}

void PartitionGlobalManager::cnn_predict(const UnitArea& cuArea) {
  cnn_predictor_->predict(cuArea, std::make_unique<cv::Mat> (cu_image_data_.image_preprocess()), cnn_prob_);

  // ��ȡģʽ
  cnn_pred_mode_ = argmax(cnn_prob_);
}

void PartitionGlobalManager::reset_testmodes_with_cnn_result(std::vector<EncTestMode>& testModes) {
  std::vector<bool> mask;

  if (cnn_prob_[1] >= Threshold_L1_QT) {
    mask = {1, 0, 1, 1, 1, 1};
  }
  else {
    mask = generate_mask_k(cnn_prob_, 2, false);
  }
  reset_testmodes(testModes, mask);
  
  //
  //for (auto i : mask) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;

  //std::cout << "CNN: ";
  //std::cout << testModes.size() << std::endl;
  //exit(0);
}

void PartitionGlobalManager::prepare_for_lgbm(const CodingStructure& cs, Partitioner& partitioner, const ComprCUCtx& cuECtx, const CodingUnit& cuArea) {
  // 
  can_QT_ = partitioner.canSplit(CU_QUAD_SPLIT, cs); // ����ʹ��const 

  lgbm_features_manager_.init(); // ��ʼ���������ڴ�ռ�
  
  // ��ȡ����
  lgbm_features_manager_.coding_data->set_coding_data(cs, partitioner, cuECtx);
  lgbm_features_manager_.texture_data->set_texture_data(cu_image_data_.get_pixel_data(), ctu_image_data_, cuArea);
  lgbm_features_manager_.context_data->set_context_data(cs, partitioner);
  
  // ��������
  lgbm_features_manager_.get_features(features_);

}

void PartitionGlobalManager::lgbm_predict(const UnitArea& cuArea) {
  // ����ģ�ͽ���Ԥ��
  lgbm_predictor_->predict(cuArea, features_, lgbm_prob_);

  lgbm_pred_mode_ = argmax(lgbm_prob_);
}

void PartitionGlobalManager::reset_testmodes_with_lgbm_result(std::vector<EncTestMode>& testModes) {
  // ע�⣺��LGBM �н���ģʽ����ʱ������LGBM ��ģʽ�������� ֡�ڲ�����֮����˱��뱣��֡��ģʽ
  std::vector<bool> mask = {0, 0, 0, 0, 0, 0};

  //auto mask1 = generate_mask_k(cnn_prob_, 2);
  //auto mask2 = generate_mask_k(lgbm_prob_, 1);

  //auto mask = multiply_vectors(mask1, mask2);

  //mask = generate_mask_with_threshold(lgbm_prob_, 0.4);

  if (shape_ == "32x32") {
    if (cnn_pred_mode_ == 0 && cnn_prob_[0] >= Threshold_L1_NS) {
      mask = {0, 1, 1, 1, 1, 1};
    }
    //else if (cnn_pred_mode_ == 2 || cnn_pred_mode_ == 3 || cnn_pred_mode_ == 4 || cnn_pred_mode_ == 5) {
    //  std::vector<bool> mask1;
    //  if (cnn_prob_[cnn_pred_mode_] >= Threshold_L1_MTT) {
    //    mask1 = generate_mask_k(cnn_prob_, 1);
    //  }
    //  else {
    //    mask1 = generate_mask_k(cnn_prob_, 2);
    //  }
    //  auto mask2 = generate_mask_k(lgbm_prob_, 1);
    //  mask = multiply_vectors(mask1, mask2);
    //}
    else {
      //auto mask1 = generate_mask_k(cnn_prob_, 2);
      auto mask2 = generate_mask_k(lgbm_prob_, 1);

      // 
      auto mask1 = generate_mask_with_threshold(cnn_prob_, Threshold_L1_MTT);

      mask = multiply_vectors(mask1, mask2);
    }
  }
  else if (shape_ == "32x16" || shape_ == "16x32" ) {  // || shape_ == "8x32" || shape_ == "32x8" || shape_ == "16x16"
    if (lgbm_pred_mode_ == 0 && lgbm_prob_[0] >= Threshold_L2_NS) {
      mask = { 0, 1, 1, 1, 1, 1 };
    }
    else {
      mask = generate_mask_with_threshold(lgbm_prob_, Threshold_L2_MTT);
    }
  }
  else if (shape_ == "8x32" || shape_ == "32x8") {
    if (lgbm_pred_mode_ == 0 && lgbm_prob_[0] >= Threshold_L2_NS) {
      mask = { 0, 1, 1, 1, 1, 1 };
    }
    else {
      mask = generate_mask_with_threshold(lgbm_prob_, Threshold_L2_MTT);
    }
  }
  else if (shape_ == "16x16") {
    if (lgbm_pred_mode_ == 0 && lgbm_prob_[0] >= Threshold_L2_NS) {
      mask = { 0, 1, 1, 1, 1, 1 };
    }
    else {
      mask = generate_mask_with_threshold(lgbm_prob_, Threshold_L2_MTT);
      mask[1] = 0; // ������ڷ�֧��������QT�Ĳ��ԣ�����ȥ��QT�Ĳ���
    }
  }

  reset_testmodes(testModes, mask);

  // 2023.12.15 added 
  // ���ԣ����ϼ��
  //std::cout << "CNN: ";
  //for (auto i : mask1) {
  //  std::cout << i << ", ";
  //}
  //std::cout << "\nCNN+LGBM: ";
  //for (auto i : mask) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;
  // 
  //std::cout << testModes.size() << std::endl;
  //exit(0);
  //std::cout << testModes.size() << std::endl;
  // 2023.12.16 end
}


void PartitionGlobalManager::reset_testmodes(std::vector<EncTestMode>& testModes, std::vector<bool>& mask) {
  std::vector<EncTestMode> new_testModes{}; // ��ȿ�����ֱ�Ӹ�ֵҲ����ȿ���

  std::vector<EncTestModeType> modes_need_to_be_pruned = apply_mask(mode_list_, mask); // ��ȡ��Ҫ��������ģʽ�б�

  //for (auto i : modes_need_to_be_pruned) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;

  for (const auto& mode : testModes) {
    if (mode.type == ETM_POST_DONT_SPLIT && mask[0] == 1)  // �ڲ�����֡�ڵĲ��Ե�����£�����ȥ����ģʽ�����򱨴�
      continue;

    if (!is_in(mode.type, modes_need_to_be_pruned))
      new_testModes.push_back(mode);
  }

  testModes = new_testModes;
}
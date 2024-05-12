#include "PartitionParams.h"

PredictionParams::PredictionParams()
  : qp{32}
  , k{FastMode_K}
  , threshold_NS_level1{0}
  , threshold_NS_level2{0}
  , threshold_QT_level1{0}
  , threshold_QT_level2{0}
  , threshold_MTT{0}
  , cnn_model_root_path {}
  , lgbm_model_root_path {}
  , cushape_list {}
  , shape_list {}
{
}

PredictionParams::PredictionParams(double th1, double th2, double th3, double th4, double th5) {
  k = FastMode_K;
  qp = 32;

  cnn_model_root_path = "models\\cnn_models\\"; // 当前路径应该是工作目录，即 EncoderApp.exe 所处的目录
  lgbm_model_root_path = "models\\lgbm_models\\";
  cushape_list = {
    std::make_pair(16, 16),
    std::make_pair(16, 32),
    std::make_pair(32, 16),
    std::make_pair(32, 32),
  };

  shape_list = ShapeList;

  threshold_NS_level1 = th1;
  threshold_NS_level2 = th2;
  threshold_QT_level1 = th3;
  threshold_QT_level2 = th4;
  threshold_MTT = th5;
}

void PredictionParams::set_qp(int QP) {
  qp = QP;
}


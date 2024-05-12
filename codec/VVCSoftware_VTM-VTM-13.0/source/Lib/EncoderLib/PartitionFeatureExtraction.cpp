#include "PartitionFeatureExtraction.h"
#include "PartitionUtils.h"

#include <numeric>
#include <map>

/*
* 管理全部的特征信息，处理特征数据，并且获得与LGBM的数据接口适配的数据
*/
LGBMFeaturesManager::LGBMFeaturesManager() 
  : num_features{0}
  , coding_data{nullptr}
  , texture_data{nullptr}
  , context_data{nullptr}
{
}

void LGBMFeaturesManager::init() {
  coding_data = std::make_unique<CodingData>();
  texture_data = std::make_unique<TextureData>();
  context_data = std::make_unique<ContextData>();
}

void LGBMFeaturesManager::get_features(std::vector<double>& dst) {
  std::vector<double> original_features = {};

  // 采用 map 的形式存储，便于处理，但是过于繁杂
  std::map<std::string, double> features_map;
  features_map["qp"] = coding_data->qp;
  features_map["width"] = coding_data->width;
  features_map["height"] = coding_data->height;
  features_map["size"] = coding_data->size;
  features_map["QTD"] = coding_data->QTD;
  features_map["BTD"] = coding_data->BTD;
  features_map["TTD"] = coding_data->TTD;
  features_map["MTTD"] = coding_data->MTTD;
  features_map["QTMTD"] = coding_data->QTMTD;
  features_map["currIntraMode"] = coding_data->currIntraMode;
  features_map["mrlIdx"] = coding_data->mrlIdx;
  features_map["ispMode"] = coding_data->ispMode;
  features_map["mtsFlag"] = coding_data->mtsFlag;
  features_map["lfnstIdx"] = coding_data->lfnstIdx;
  features_map["mipFlag"] = coding_data->mipFlag;
  features_map["mipTransposedFlag"] = coding_data->mipTransposedFlag;
  features_map["isModeVer"] = coding_data->isModeVer;
  features_map["intraPredAngleMode"] = coding_data->intraPredAngleMode;

  features_map["bestPredModeDCT2"] = coding_data->bestPredModeDCT2;

  features_map["lastSplit"] = coding_data->lastSplit;
  features_map["qtBeforeBt"] = coding_data->qtBeforeBt;
  features_map["currIntraCost"] = coding_data->currIntraCost;
  features_map["currIntraFracBits"] = coding_data->currIntraFracBits;
  features_map["currIntraDistoration"] = coding_data->currIntraDistoration;

  features_map["mean"] = texture_data->mean;
  features_map["stddev"] = texture_data->stddev;
  features_map["diffStdDevVer"] = texture_data->diffStdDevVer;
  features_map["diffStdDevHor"] = texture_data->diffStdDevHor;
  features_map["Gx"] = texture_data->Gx;
  features_map["Gy"] = texture_data->Gy;
  features_map["ratioGxGy"] = texture_data->ratioGxGy;
  features_map["normGradient"] = texture_data->normGradient;
  features_map["pixelSum"] = texture_data->pixelSum;
  features_map["entropy"] = texture_data->entropy;
  features_map["skewness"] = texture_data->skewness;
  features_map["kurtosis"] = texture_data->kurtosis;

  features_map["neighAvgQT"] = context_data->neighAvgQT;
  features_map["neighHigherQT"] = context_data->neighHigherQT;
  features_map["neighAvgMTT"] = context_data->neighAvgMTT;
  features_map["neighHigherMTT"] = context_data->neighHigherMTT;
  features_map["neighAvgHorNum"] = context_data->neighAvgHorNum;
  features_map["neighAvgVerNum"] = context_data->neighAvgVerNum;

  //features_map["NS_Prob"] = cnn_output[0];
  //features_map["QT_Prob"] = cnn_output[1];
  //features_map["BH_Prob"] = cnn_output[2];
  //features_map["BV_Prob"] = cnn_output[3];
  //features_map["TH_Prob"] = cnn_output[4];
  //features_map["TV_Prob"] = cnn_output[5];

  features_map["contrast_0"] = texture_data->GLCMFeatures[0][0];
  features_map["contrast_1"] = texture_data->GLCMFeatures[0][1];
  features_map["contrast_2"] = texture_data->GLCMFeatures[0][2];
  features_map["contrast_3"] = texture_data->GLCMFeatures[0][3];
  features_map["energy_0"] = texture_data->GLCMFeatures[1][0];
  features_map["energy_1"] = texture_data->GLCMFeatures[1][1];
  features_map["energy_2"] = texture_data->GLCMFeatures[1][2];
  features_map["energy_3"] = texture_data->GLCMFeatures[1][3];
  features_map["homogeneity_0"] = texture_data->GLCMFeatures[2][0];
  features_map["homogeneity_1"] = texture_data->GLCMFeatures[2][1];
  features_map["homogeneity_2"] = texture_data->GLCMFeatures[2][2];
  features_map["homogeneity_3"] = texture_data->GLCMFeatures[2][3];
  features_map["dissimilarity_0"] = texture_data->GLCMFeatures[3][0];
  features_map["dissimilarity_1"] = texture_data->GLCMFeatures[3][1];
  features_map["dissimilarity_2"] = texture_data->GLCMFeatures[3][2];
  features_map["dissimilarity_3"] = texture_data->GLCMFeatures[3][3];
  features_map["contrast"] = texture_data->GLCMFeatures[4][0];
  features_map["energy"] = texture_data->GLCMFeatures[4][1];
  features_map["homogeneity"] = texture_data->GLCMFeatures[4][2];
  features_map["dissimilarity"] = texture_data->GLCMFeatures[4][3];


  // 子块特征
  features_map["BH_Above_mean"] = texture_data->BH_Above_mean;
  features_map["BH_Above_stddev"] = texture_data->BH_Above_stddev;
  features_map["BH_Above_Gx"] = texture_data->BH_Above_Gx;
  features_map["BH_Above_Gy"] = texture_data->BH_Above_Gy;
  features_map["BH_Above_ratioGxGy"] = texture_data->BH_Above_ratioGxGy;
  features_map["BH_Above_normGradient"] = texture_data->BH_Above_normGradient;

  features_map["BH_Below_mean"] = texture_data->BH_Below_mean;
  features_map["BH_Below_stddev"] = texture_data->BH_Below_stddev;
  features_map["BH_Below_Gx"] = texture_data->BH_Below_Gx;
  features_map["BH_Below_Gy"] = texture_data->BH_Below_Gy;
  features_map["BH_Below_ratioGxGy"] = texture_data->BH_Below_ratioGxGy;
  features_map["BH_Below_normGradient"] = texture_data->BH_Below_normGradient;

  features_map["BV_Right_mean"] = texture_data->BV_Right_mean;
  features_map["BV_Right_stddev"] = texture_data->BV_Right_stddev;
  features_map["BV_Right_Gx"] = texture_data->BV_Right_Gx;
  features_map["BV_Right_Gy"] = texture_data->BV_Right_Gy;
  features_map["BV_Right_ratioGxGy"] = texture_data->BV_Right_ratioGxGy;
  features_map["BV_Right_normGradient"] = texture_data->BV_Right_normGradient;

  features_map["BV_Left_mean"] = texture_data->BV_Left_mean;
  features_map["BV_Left_stddev"] = texture_data->BV_Left_stddev;
  features_map["BV_Left_Gx"] = texture_data->BV_Left_Gx;
  features_map["BV_Left_Gy"] = texture_data->BV_Left_Gy;
  features_map["BV_Left_ratioGxGy"] = texture_data->BV_Left_ratioGxGy;
  features_map["BV_Left_normGradient"] = texture_data->BV_Left_normGradient;

  features_map["TH_Above_mean"] = texture_data->TH_Above_mean;
  features_map["TH_Above_stddev"] = texture_data->TH_Above_stddev;
  features_map["TH_Above_Gx"] = texture_data->TH_Above_Gx;
  features_map["TH_Above_Gy"] = texture_data->TH_Above_Gy;
  features_map["TH_Above_ratioGxGy"] = texture_data->TH_Above_ratioGxGy;
  features_map["TH_Above_normGradient"] = texture_data->TH_Above_normGradient;

  features_map["TH_Middle_mean"] = texture_data->TH_Middle_mean;
  features_map["TH_Middle_stddev"] = texture_data->TH_Middle_stddev;
  features_map["TH_Middle_Gx"] = texture_data->TH_Middle_Gx;
  features_map["TH_Middle_Gy"] = texture_data->TH_Middle_Gy;
  features_map["TH_Middle_ratioGxGy"] = texture_data->TH_Middle_ratioGxGy;
  features_map["TH_Middle_normGradient"] = texture_data->TH_Middle_normGradient;

  features_map["TH_Below_mean"] = texture_data->TH_Below_mean;
  features_map["TH_Below_stddev"] = texture_data->TH_Below_stddev;
  features_map["TH_Below_Gx"] = texture_data->TH_Below_Gx;
  features_map["TH_Below_Gy"] = texture_data->TH_Below_Gy;
  features_map["TH_Below_ratioGxGy"] = texture_data->TH_Below_ratioGxGy;
  features_map["TH_Below_normGradient"] = texture_data->TH_Below_normGradient;

  features_map["TV_Right_mean"] = texture_data->TV_Right_mean;
  features_map["TV_Right_stddev"] = texture_data->TV_Right_stddev;
  features_map["TV_Right_Gx"] = texture_data->TV_Right_Gx;
  features_map["TV_Right_Gy"] = texture_data->TV_Right_Gy;
  features_map["TV_Right_ratioGxGy"] = texture_data->TV_Right_ratioGxGy;
  features_map["TV_Right_normGradient"] = texture_data->TV_Right_normGradient;

  features_map["TV_Middle_mean"] = texture_data->TV_Middle_mean;
  features_map["TV_Middle_stddev"] = texture_data->TV_Middle_stddev;
  features_map["TV_Middle_Gx"] = texture_data->TV_Middle_Gx;
  features_map["TV_Middle_Gy"] = texture_data->TV_Middle_Gy;
  features_map["TV_Middle_ratioGxGy"] = texture_data->TV_Middle_ratioGxGy;
  features_map["TV_Middle_normGradient"] = texture_data->TV_Middle_normGradient;

  features_map["TV_Left_mean"] = texture_data->TV_Left_mean;
  features_map["TV_Left_stddev"] = texture_data->TV_Left_stddev;
  features_map["TV_Left_Gx"] = texture_data->TV_Left_Gx;
  features_map["TV_Left_Gy"] = texture_data->TV_Left_Gy;
  features_map["TV_Left_ratioGxGy"] = texture_data->TV_Left_ratioGxGy;
  features_map["TV_Left_normGradient"] = texture_data->TV_Left_normGradient;

  // 特征处理
  dst.clear();
  features_preprocess(features_map, dst);
  
}

// 对特征进行一些处理，计算获得新的特征、处理缺失值、错误取值等，将特征调整为训练输入的顺序
void LGBMFeaturesManager::features_preprocess(std::map<std::string, double>& src, std::vector<double>& dst) {
  
  // 按顺序填充数据，必须与Python训练时的输入顺序一致
  dst = {
    src["QTD"], src["BTD"], src["TTD"], src["MTTD"], src["QTMTD"],
    src["currIntraMode"], src["mrlIdx"], src["ispMode"], src["mtsFlag"], src["lfnstIdx"], src["mipFlag"], src["mipTransposedFlag"], 
    src["isModeVer"], src["intraPredAngleMode"], 
    src["currIntraFracBits"], src["currIntraDistortion"], src["currIntraCost"],

    src["bestPredModeDCT2"], // 新增

    src["mean"], src["stddev"], src["diffStdDevVer"], src["diffStdDevHor"], src["Gx"], src["Gy"], src["ratioGxGy"], src["normGradient"], 
    src["entropy"], src["skewness"], src["kurtosis"], src["pixelSum"], 

    //src["contrast_0"], src["contrast_1"], src["contrast_2"], src["contrast_3"], 
    //src["energy_0"], src["energy_1"], src["energy_2"], src["energy_3"], 
    //src["homogeneity_0"], src["homogeneity_1"], src["homogeneity_2"], src["homogeneity_3"], 
    //src["dissimilarity_0"], src["dissimilarity_1"], src["dissimilarity_2"], src["dissimilarity_3"], 
    //src["contrast"], src["energy"],  src["homogeneity"], src["dissimilarity"],

    src["BH_Above_mean"], src["BH_Above_stddev"], src["BH_Above_Gx"], src["BH_Above_Gy"], src["BH_Above_ratioGxGy"], src["BH_Above_normGradient"], 
    src["BH_Below_mean"], src["BH_Below_stddev"], src["BH_Below_Gx"], src["BH_Below_Gy"], src["BH_Below_ratioGxGy"], src["BH_Below_normGradient"],
    src["BV_Left_mean"], src["BV_Left_stddev"], src["BV_Left_Gx"], src["BV_Left_Gy"], src["BV_Left_ratioGxGy"], src["BV_Left_normGradient"],
    src["BV_Right_mean"], src["BV_Right_stddev"], src["BV_Right_Gx"], src["BV_Right_Gy"], src["BV_Right_ratioGxGy"], src["BV_Right_normGradient"],
    src["TH_Above_mean"], src["TH_Above_stddev"], src["TH_Above_Gx"], src["TH_Above_Gy"], src["TH_Above_ratioGxGy"], src["TH_Above_normGradient"],
    src["TH_Middle_mean"], src["TH_Middle_stddev"], src["TH_Middle_Gx"], src["TH_Middle_Gy"], src["TH_Middle_ratioGxGy"], src["TH_Middle_normGradient"],
    src["TH_Below_mean"], src["TH_Below_stddev"], src["TH_Below_Gx"], src["TH_Below_Gy"], src["TH_Below_ratioGxGy"], src["TH_Below_normGradient"],
    src["TV_Left_mean"], src["TV_Left_stddev"], src["TV_Left_Gx"], src["TV_Left_Gy"], src["TV_Left_ratioGxGy"], src["TV_Left_normGradient"],
    src["TV_Middle_mean"], src["TV_Middle_stddev"], src["TV_Middle_Gx"], src["TV_Middle_Gy"], src["TV_Middle_ratioGxGy"], src["TV_Middle_normGradient"],
    src["TV_Right_mean"], src["TV_Right_stddev"], src["TV_Right_Gx"], src["TV_Right_Gy"], src["TV_Right_ratioGxGy"], src["TV_Right_normGradient"],

    src["neighAvgQT"], src["neighHigherQT"],  src["neighAvgMTT"], src["neighHigherMTT"], src["neighAvgHorNum"], src["neighAvgVerNum"],
    //src["NS_Prob"], src["QT_Prob"],  src["BH_Prob"], src["BV_Prob"], src["TH_Prob"], src["TV_Prob"], // 未使用特征
    //src["qtBeforeBt"], src["lastSplit"]  // 旧版本特征
  };

  //for (auto i : dst) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;
  //exit(0);
}

/*
* 基础的编码信息
*/
CodingData::CodingData() 
  : qp{-1}
  , width{-1}
  , height{-1}
  , size{-1}
  , QTD{-1}
  , BTD{-1}
  , TTD{-1}
  , MTTD{-1}
  , QTMTD{-1}
  , currIntraMode{-1}
  , mrlIdx{-1}
  , ispMode{-1}
  , mtsFlag{-1}
  , lfnstIdx{-1}
  , mipFlag{-1}
  , mipTransposedFlag{-1}
  , isModeVer{-1}
  , intraPredAngleMode{-1}
  , lastSplit{-1}
  , qtBeforeBt{-1}
  , currIntraCost{0.0}
  , currIntraFracBits{0.0}
  , currIntraDistoration{0.0}
{
}

void CodingData::set_coding_data(const CodingStructure& cs, const Partitioner& partitioner, const ComprCUCtx& cuECtx) {
  qp = cs.baseQP;
  width = cs.area.lwidth();
  height = cs.area.lheight();
  size = width * height;

  QTD = partitioner.currQtDepth;
  BTD = partitioner.currBtDepth;
  TTD = partitioner.currTrDepth;
  MTTD = partitioner.currMtDepth;
  QTMTD = partitioner.currDepth;

  currIntraMode = cs.pus[0]->intraDir[0];
  mrlIdx = cs.pus[0]->multiRefIdx;
  ispMode = cs.cus[0]->ispMode;
  mtsFlag = cs.cus[0]->mtsFlag;
  lfnstIdx = cs.cus[0]->lfnstIdx;
  mipFlag = cs.cus[0]->mipFlag;
  mipTransposedFlag = cs.pus[0]->mipTransposedFlag; // 标记 -1 为缺失值

  isModeVer = currIntraMode >= 34;
  intraPredAngleMode = isModeVer ? currIntraMode - 50 : -(currIntraMode - 18);

  currIntraCost = cs.cost; // 标记 0 为缺失值 
  currIntraFracBits = cs.fracBits;
  currIntraDistoration = cs.dist;

  lastSplit = partitioner.getPartStack().back().split;
  qtBeforeBt = partitioner.t_qtBeforeBt;

  bestPredModeDCT2 = cuECtx.bestPredModeDCT2;
}

std::vector<int> CodingData::get_coding_data() {
  return std::vector<int> {
    qp, width, height, size,
      QTD, BTD, TTD, MTTD, QTMTD,
      currIntraMode, mrlIdx, ispMode, mtsFlag, lfnstIdx, mipFlag, mipTransposedFlag
  };
}

std::vector<double> CodingData::get_intra_cost() {
  return std::vector<double> {
    currIntraCost, currIntraFracBits, currIntraDistoration
  };
}


TextureData::TextureData() 
  : mean{0.0}
  , stddev{0.0}
  , diffStdDevVer{0.0}
  , diffStdDevHor{0.0}
  , Gx{0.0}
  , Gy{0.0}
  , ratioGxGy{0.0}
  , normGradient{0.0}
  , pixelSum{0.0}
  , GLCMFeatures{0.0}
  , entropy{0.0}
  , skewness{0.0}
  , kurtosis{0.0}
{
}

void TextureData::getBHFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y) {
  int height = src.rows;
  int width = src.cols;

  // 分块，上下两块
  cv::Mat aboveBlock = src(cv::Rect(0, 0, width, height / 2));
  cv::Mat belowBlock = src(cv::Rect(0, height / 2, width, height / 2));

  // 计算两块的标准差
  cv::Scalar meanAbove, stddevAbove;
  cv::Scalar meanBelow, stddevBelow;

  cv::meanStdDev(aboveBlock, meanAbove, stddevAbove);
  cv::meanStdDev(belowBlock, meanBelow, stddevBelow);

  BH_Above_mean = meanAbove[0];
  BH_Above_stddev = stddevAbove[0];

  BH_Below_mean = meanBelow[0];
  BH_Below_stddev = stddevBelow[0];

  // 计算标准差的绝对差值
  double stdDevDifference = cv::abs(stddevAbove[0] - stddevBelow[0]);
  diffStdDevHor = stdDevDifference;

  // 计算梯度
  auto aboveGradient = ctu_data->getGradientFeatures(x, y, width, height / 2);
  BH_Above_Gx = aboveGradient[0];
  BH_Above_Gy = aboveGradient[1];
  BH_Above_ratioGxGy = aboveGradient[2];
  BH_Above_normGradient = aboveGradient[3];

  auto belowGradient = ctu_data->getGradientFeatures(x, y + height / 2, width, height / 2);
  BH_Below_Gx = belowGradient[0];
  BH_Below_Gy = belowGradient[1];
  BH_Below_ratioGxGy = belowGradient[2];
  BH_Below_normGradient = belowGradient[3];
}

void TextureData::getBVFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y) {
  int height = src.rows;
  int width = src.cols;

  // 分块，左右两块
  cv::Mat leftBlock = src(cv::Rect(0, 0, width / 2, height));
  cv::Mat rightBlock = src(cv::Rect(width / 2, 0, width / 2, height));

  // 计算两块的标准差
  cv::Scalar meanLeft, stddevLeft;
  cv::Scalar meanRight, stddevRight;

  cv::meanStdDev(leftBlock, meanLeft, stddevLeft);
  cv::meanStdDev(rightBlock, meanRight, stddevRight);

  BV_Left_mean = meanLeft[0];
  BV_Left_stddev = stddevLeft[0];

  BV_Right_mean = meanRight[0];
  BV_Right_stddev = stddevRight[0];

  // 计算标准差的绝对差值
  double stdDevDifference = cv::abs(stddevLeft[0] - stddevRight[0]);
  diffStdDevVer = stdDevDifference;

  // 计算梯度
  auto leftGradient = ctu_data->getGradientFeatures(x, y, width / 2, height);
  BV_Left_Gx = leftGradient[0];
  BV_Left_Gy = leftGradient[1];
  BV_Left_ratioGxGy = leftGradient[2];
  BV_Left_normGradient = leftGradient[3];

  auto rightGradient = ctu_data->getGradientFeatures(x + width / 2, y, width / 2, height);
  BV_Right_Gx = rightGradient[0];
  BV_Right_Gy = rightGradient[1];
  BV_Right_ratioGxGy = rightGradient[2];
  BV_Right_normGradient = rightGradient[3];
}

void TextureData::getTHFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y) {
  int height = src.rows;
  int width = src.cols;

  // 分块，上中下两块
  cv::Mat aboveBlock = src(cv::Rect(0, 0, width, height / 4));
  cv::Mat middleBlock = src(cv::Rect(0, height / 4, width, height / 2));
  cv::Mat belowBlock = src(cv::Rect(0, height / 4 + height / 2, width, height / 4));

  // 计算三块的标准差
  cv::Scalar meanAbove, stddevAbove;
  cv::Scalar meanMiddle, stddevMiddle;
  cv::Scalar meanBelow, stddevBelow;

  cv::meanStdDev(aboveBlock, meanAbove, stddevAbove);
  cv::meanStdDev(middleBlock, meanMiddle, stddevMiddle);
  cv::meanStdDev(belowBlock, meanBelow, stddevBelow);

  TH_Above_mean = meanAbove[0];
  TH_Above_stddev = stddevAbove[0];

  TH_Middle_mean = meanMiddle[0];
  TH_Middle_stddev = stddevMiddle[0];

  TH_Below_mean = meanBelow[0];
  TH_Below_stddev = stddevBelow[0];

  // 计算梯度
  auto aboveGradient = ctu_data->getGradientFeatures(x, y, width, height / 4);
  TH_Above_Gx = aboveGradient[0];
  TH_Above_Gy = aboveGradient[1];
  TH_Above_ratioGxGy = aboveGradient[2];
  TH_Above_normGradient = aboveGradient[3];

  auto middleGradient = ctu_data->getGradientFeatures(x, y + height / 4, width, height / 2);
  TH_Middle_Gx = middleGradient[0];
  TH_Middle_Gy = middleGradient[1];
  TH_Middle_ratioGxGy = middleGradient[2];
  TH_Middle_normGradient = middleGradient[3];


  auto belowGradient = ctu_data->getGradientFeatures(x, y + height / 4 + height / 2, width, height / 4);
  TH_Below_Gx = belowGradient[0];
  TH_Below_Gy = belowGradient[1];
  TH_Below_ratioGxGy = belowGradient[2];
  TH_Below_normGradient = belowGradient[3];
}

void TextureData::getTVFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y) {
  int height = src.rows;
  int width = src.cols;

  // 分块，上中下两块
  cv::Mat leftBlock = src(cv::Rect(0, 0, width / 4, height));
  cv::Mat middleBlock = src(cv::Rect(width / 4, 0, width / 2, height));
  cv::Mat rightBlock = src(cv::Rect(width / 4 + width / 2, 0, width / 4, height));

  // 计算三块的标准差
  cv::Scalar meanLeft, stddevLeft;
  cv::Scalar meanMiddle, stddevMiddle;
  cv::Scalar meanRight, stddevRight;

  cv::meanStdDev(leftBlock, meanLeft, stddevLeft);
  cv::meanStdDev(rightBlock, meanRight, stddevRight);
  cv::meanStdDev(middleBlock, meanMiddle, stddevMiddle);

  TV_Left_mean = meanLeft[0];
  TV_Left_stddev = stddevLeft[0];

  TV_Right_mean = meanRight[0];
  TV_Right_stddev = stddevRight[0];

  TV_Middle_mean = meanMiddle[0];
  TV_Middle_stddev = stddevMiddle[0];

  // 计算梯度
  auto leftGradient = ctu_data->getGradientFeatures(x, y, width / 4, height);
  TV_Left_Gx = leftGradient[0];
  TV_Left_Gy = leftGradient[1];
  TV_Left_ratioGxGy = leftGradient[2];
  TV_Left_normGradient = leftGradient[3];

  auto middleGradient = ctu_data->getGradientFeatures(x + width / 4, y, width / 2, height);
  TV_Middle_Gx = middleGradient[0];
  TV_Middle_Gy = middleGradient[1];
  TV_Middle_ratioGxGy = middleGradient[2];
  TV_Middle_normGradient = middleGradient[3];


  auto rightGradient = ctu_data->getGradientFeatures(x + width / 4 + width / 2, y, width / 4, height);
  TV_Right_Gx = rightGradient[0];
  TV_Right_Gy = rightGradient[1];
  TV_Right_ratioGxGy = rightGradient[2];
  TV_Right_normGradient = rightGradient[3];
}

void TextureData::set_texture_data(const std::unique_ptr<cv::Mat>& src, CTUImageDataManager* ctu_data, const CodingUnit& cuArea) {
  auto ctuArea = ctu_data->get_ctuArea();
  int x_in_ctu = cuArea.lx() - ctuArea.lx();
  int y_in_ctu = cuArea.ly() - ctuArea.ly();

  calcMeanStdDev(*src, mean, stddev);
  //calcGradientFeatures(*src, Gx, Gy, ratioGxGy, normGradient); // 旧版本的梯度计算方式

  //calcDiffStdDevVer(*src, diffStdDevVer); // 这里的数值在 getBHFeatures 以及 getBVFeatures 中计算
  //calcDiffStdDevHor(*src, diffStdDevHor);

  calcEntropy(*src, entropy);
  calcSkewnessKurtosis(*src, skewness, kurtosis);
  //GLCMFeatures = calcGLCMFeatures(*src, 1, GLCMGRAY_8);

  // 以下代码用于计算新的特征
  auto gradients = ctu_data->getGradientFeatures(x_in_ctu, y_in_ctu, cuArea.lwidth(), cuArea.lheight()); // 新增
  Gx = gradients[0];
  Gy = gradients[1];
  ratioGxGy = gradients[2];
  normGradient = gradients[3];

  getBHFeatures(*src, ctu_data, x_in_ctu, y_in_ctu); // 新增
  getBVFeatures(*src, ctu_data, x_in_ctu, y_in_ctu);
  getTHFeatures(*src, ctu_data, x_in_ctu, y_in_ctu);
  getTVFeatures(*src, ctu_data, x_in_ctu, y_in_ctu);

  // 2024.1.9 added
  // 测试：两种计算梯度的方式所获得的梯度的差异？
  // 结果：计算结果完全一致，并不存在任何差异。
  // 
  // std::cout << "方法2：" << "Gx = " << Gx << ", Gy = " << Gy << ", ratioGxGy = " << ratioGxGy << ", normGradient = " << normGradient << std::endl;
  // 2024.1.9 end

}

std::vector<double> TextureData::get_texture_data() {
  std::vector<double> features = {
    mean, stddev, diffStdDevVer, diffStdDevHor, Gx, Gy,
      ratioGxGy, normGradient, pixelSum, entropy, skewness, kurtosis
  };
  
  for (auto i : GLCMFeatures) {
    for (auto j : i) {
      features.push_back(j);
    }
  }

  return features;
}


ContextData::ContextData() 
  : neighAvgQT{-1}
  , neighHigherQT{-1}
  , neighAvgMTT{-1}
  , neighHigherMTT{-1}
  , neighAvgHorNum{-1}
  , neighAvgVerNum{-1}
{
}

void ContextData::set_context_data(const CodingStructure& cs, const Partitioner& partitioner) {
  // 初始化为 0 
  neighAvgQT = 0;
  neighHigherQT = 0;
  neighAvgMTT = 0;
  neighHigherMTT = 0;
  neighAvgHorNum = 0;
  neighAvgVerNum = 0;

  const CodingUnit* cuLeft = cs.getCU(cs.area.blocks[partitioner.chType].pos().offset(-1, 0), partitioner.chType);
  const CodingUnit* cuAbove = cs.getCU(cs.area.blocks[partitioner.chType].pos().offset(0, -1), partitioner.chType); // 注意：这里获得的CU的尺寸并不是和当前CU尺寸一致的
  const CodingUnit* cuLeftAbove = cs.getCU(cs.area.blocks[partitioner.chType].pos().offset(-1, -1), partitioner.chType);
  const CodingUnit* cuRightAbove = cs.getCU(cs.area.blocks[partitioner.chType].pos().offset(1, -1), partitioner.chType);

  int validNeighbor = 0;
  int neighTotalQT = 0;
  int neighTotalMTT = 0;
  std::vector<int> splitVector;

  int currQtDepth = partitioner.currQtDepth;
  int currMttDepth = partitioner.currMtDepth;

  if (cuLeft) {
    validNeighbor++;
    neighTotalQT += cuLeft->qtDepth;
    neighTotalMTT += cuLeft->mtDepth;

    if (cuLeft->qtDepth > currQtDepth)
      neighHigherQT++;
    if (cuLeft->mtDepth > currMttDepth)
      neighHigherMTT++;

    parse_split_series(cuLeft->splitSeries, splitVector, cuLeft->depth);
    for (auto i : splitVector) {
      if (i == 2 || i == 4)
        neighAvgHorNum++;
      else if (i == 3 || i == 5)
        neighAvgVerNum++;
    }
  }

  if (cuAbove) {
    validNeighbor++;
    neighTotalQT += cuAbove->qtDepth;
    neighTotalMTT += cuAbove->mtDepth;

    if (cuAbove->qtDepth > currQtDepth)
      neighHigherQT++;
    if (cuAbove->mtDepth > currMttDepth)
      neighHigherMTT++;

    parse_split_series(cuAbove->splitSeries, splitVector, cuAbove->depth);
    for (auto i : splitVector) {
      if (i == 2 || i == 4)
        neighAvgHorNum++;
      else if (i == 3 || i == 5)
        neighAvgVerNum++;
    }
  }

  if (cuLeftAbove) {
    validNeighbor++;
    neighTotalQT += cuLeftAbove->qtDepth;
    neighTotalMTT += cuLeftAbove->mtDepth;

    if (cuLeftAbove->qtDepth > currQtDepth)
      neighHigherQT++;
    if (cuLeftAbove->mtDepth > currMttDepth)
      neighHigherMTT++;

    parse_split_series(cuLeftAbove->splitSeries, splitVector, cuLeftAbove->depth);
    for (auto i : splitVector) {
      if (i == 2 || i == 4)
        neighAvgHorNum++;
      else if (i == 3 || i == 5)
        neighAvgVerNum++;
    }
  }

  if (cuRightAbove) {
    validNeighbor++;
    neighTotalQT += cuRightAbove->qtDepth;
    neighTotalMTT += cuRightAbove->mtDepth;

    if (cuRightAbove->qtDepth > currQtDepth)
      neighHigherQT++;
    if (cuRightAbove->mtDepth > currMttDepth)
      neighHigherMTT++;

    parse_split_series(cuRightAbove->splitSeries, splitVector, cuRightAbove->depth);
    for (auto i : splitVector) {
      if (i == 2 || i == 4)
        neighAvgHorNum++;
      else if (i == 3 || i == 5)
        neighAvgVerNum++;
    }
  }

  // 计算均值
  if (validNeighbor) {
    neighAvgQT = int(neighTotalQT / validNeighbor);
    neighAvgMTT = int(neighTotalMTT / validNeighbor);
    neighAvgHorNum = int(neighAvgHorNum / validNeighbor);
    neighAvgVerNum = int(neighAvgVerNum / validNeighbor);
  }
}

void ContextData::parse_split_series(uint64_t splitSeries, std::vector<int>& splitVector, int depth) {
  const int SPLIT_DMULT = 5;
  const int CTU_LEVEL = -1; // 这里的CTU_LEVEL是一个占位符，你需要根据实际情况定义它
  splitVector.clear();

  for (int d = 0; d < depth; d++) {
    int splitValue = (splitSeries >> (d * SPLIT_DMULT)) & ((1 << SPLIT_DMULT) - 1);
    splitVector.push_back(splitValue);
  }
}

std::vector<int> ContextData::get_context_data() {
  return std::vector<int> {
    neighAvgQT, neighHigherQT, neighAvgMTT, neighHigherMTT, neighAvgHorNum, neighAvgVerNum
  };
}



/*
纹理特征计算
*/
void calcMeanStdDev(const cv::Mat& src, double& mean, double& stddev) {
  cv::Scalar mean_; // 均值
  cv::Scalar stddev_; // 标准差
  cv::meanStdDev(src, mean_, stddev_); // 计算均值和标准差
  mean = mean_[0];
  stddev = stddev_[0]; 
}

void calcGradientFeatures(const cv::Mat& src, double& gradient_x, double& gradient_y, double& ratio_gx_gy, double& norm_gradient) {
  int width = src.rows;
  int height = src.cols;

  cv::Mat image_gradX; // 计算梯度
  cv::Mat image_gradY;
  cv::Sobel(src, image_gradX, CV_16S, 1, 0, 3);
  cv::Sobel(src, image_gradY, CV_16S, 0, 1, 3);

  gradient_x = cv::sum(cv::abs(image_gradX))[0];
  gradient_y = cv::sum(cv::abs(image_gradY))[0];
  ratio_gx_gy = gradient_x / (gradient_y + 0.0000001);  // 防止出现除0的情况
  norm_gradient = (gradient_x + gradient_y) / (width * height);
}


void calcDiffStdDevVer(const cv::Mat& src, double& diff_stddev_ver) {
  // 获取图像的尺寸
  int height = src.rows;
  int width = src.cols;

  // 将图像分为两块
  cv::Mat leftBlock = src(cv::Rect(0, 0, width / 2, height));
  cv::Mat rightBlock = src(cv::Rect(width / 2, 0, width / 2, height));

  // 计算两块的标准差
  cv::Scalar meanLeft, stddevLeft;
  cv::Scalar meanRight, stddevRight;

  cv::meanStdDev(leftBlock, meanLeft, stddevLeft);
  cv::meanStdDev(rightBlock, meanRight, stddevRight);

  // 计算标准差的绝对差值
  diff_stddev_ver = cv::abs(stddevLeft[0] - stddevRight[0]);

}

void calcDiffStdDevHor(const cv::Mat& src, double& diff_stddev_hor) {
  // 获取图像的尺寸
  int height = src.rows;
  int width = src.cols;

  // 将图像分为两块
  cv::Mat aboveBlock = src(cv::Rect(0, 0, width, height / 2));
  cv::Mat belowBlock = src(cv::Rect(0, height / 2, width, height / 2));

  // 计算两块的标准差
  cv::Scalar meanAbove, stddevAbove;
  cv::Scalar meanBelow, stddevBelow;

  cv::meanStdDev(aboveBlock, meanAbove, stddevAbove);
  cv::meanStdDev(belowBlock, meanBelow, stddevBelow);

  // 计算标准差的绝对差值
  diff_stddev_hor = cv::abs(stddevAbove[0] - stddevBelow[0]);

}

// 计算图像的熵
void calcEntropy(const cv::Mat& src, double& entropy) {
  cv::Mat hist;
  int histSize = 256;
  float range[] = { 0, 256 };
  const float* histRange = { range };

  // 计算直方图
  cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

  // 归一化直方图
  hist /= (src.rows * src.cols);

  // 计算图像的熵
  entropy = 0.0; // 首先初始化
  for (int i = 0; i < histSize; i++) {
    if (hist.at<float>(i) > 0.0) {
      entropy -= hist.at<float>(i) * log2(hist.at<float>(i));
    }
  }
}

// 计算图像的偏度
void calcSkewnessKurtosis(const cv::Mat& src, double& skewness, double& kurtosis) {
  cv::Scalar mean, stddev;
  cv::meanStdDev(src, mean, stddev);

  skewness = 0.0;
  kurtosis = 0.0;
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      double pixel = src.at<uchar>(y, x);
      skewness += std::pow(pixel - mean[0], 3);
      kurtosis += std::pow(pixel - mean[0], 4);
    }
  }
  skewness /= (src.rows * src.cols * std::pow(stddev[0] + 0.0000001, 3));
  kurtosis /= (src.rows * src.cols * std::pow(stddev[0] + 0.0000001, 4));
}


void GrayMagnitude(const cv::Mat& src, cv::Mat& dst, GLCMGrayLevel level) {
  cv::Mat tmp;
  src.copyTo(tmp);
  if (tmp.channels() == 3)
    cvtColor(tmp, tmp, CV_BGR2GRAY);

  // 直方图均衡化
  // Equalize Histogram
  equalizeHist(tmp, tmp);

  for (int j = 0; j < tmp.rows; j++)
  {
    const uchar* current = tmp.ptr<uchar>(j);
    uchar* output = dst.ptr<uchar>(j);

    for (int i = 0; i < tmp.cols; i++)
    {
      switch (level)
      {
      case GLCMGRAY_4:
        output[i] = cv::saturate_cast<uchar>(current[i] / 64);
        break;
      case GLCMGRAY_8:
        output[i] = cv::saturate_cast<uchar>(current[i] / 32);
        break;
      case GLCMGRAY_16:
        output[i] = cv::saturate_cast<uchar>(current[i] / 16);
        break;
      default:
        std::cout << "ERROR in GrayMagnitude(): No Such GrayLevel." << std::endl;
        return;
      }
    }
  }
}


// 计算GLCM特征
std::array<std::array<double, 4>, 5> calcGLCMFeatures(const cv::Mat& src, int d, GLCMGrayLevel level) {
  int num_directions = 4; // 四个方向：0°，45°，90°，135°
  int num_levels = 256;
  std::array<std::array<double, 4>, 5> features = { 0 };

  switch (level) {
  case GLCMGRAY_4:
    num_levels = 4; break;
  case GLCMGRAY_8:
    num_levels = 8; break;
  case GLCMGRAY_16:
    num_levels = 16;  break;
  default:
    std::cout << "ERROR in CalcuOneGLCM(): No Such Gray Level." << std::endl;
    break;
  }

  cv::Mat MagnitudedImage(src.rows, src.cols, CV_8UC1);
  GrayMagnitude(src, MagnitudedImage, level);

  for (int theta = 0; theta < num_directions; ++theta) {
    // 创建GLCM
    cv::Mat glcm = cv::Mat::zeros(num_levels, num_levels, CV_32F);

    // 计算GLCM
    for (int y = 0; y < MagnitudedImage.rows; y++) {
      for (int x = 0; x < MagnitudedImage.cols; x++) {
        int i = static_cast<int>(MagnitudedImage.at<uchar>(y, x));
        int j = 0;

        switch (theta) {
        case 0: // 0°
          if (x + d < MagnitudedImage.cols) {
            j = static_cast<int>(MagnitudedImage.at<uchar>(y, x + d));
          }
          break;
        case 1: // 45°
          if (y >= d && x + d < MagnitudedImage.cols) {
            j = static_cast<int>(MagnitudedImage.at<uchar>(y - d, x + d)); // 为
          }
          break;
        case 2: // 90°
          if (y >= d) {
            j = static_cast<int>(MagnitudedImage.at<uchar>(y - d, x));
          }
          break;
        case 3: // 135°
          if (y >= d && x >= d) {
            j = static_cast<int>(MagnitudedImage.at<uchar>(y - d, x - d));
          }
          break;
        }

        if (i >= num_levels || j >= num_levels) {
          continue; // 跳过超出灰度级别的像素
        }

        glcm.at<float>(i, j) += 1;
      }
    }

    // 归一化GLCM
    glcm /= cv::sum(glcm)[0];

    // 计算GLCM特征
    double contrast = 0.0;
    double energy = 0.0;
    double correlation = 0.0;
    double homogeneity = 0.0;
    double dissimilarity = 0.0;

    for (int i = 0; i < num_levels; i++) {
      for (int j = 0; j < num_levels; j++) {
        double p = glcm.at<float>(i, j);
        contrast += p * std::pow(i - j, 2);
        energy += std::pow(p, 2);
        homogeneity += p / (1 + std::abs(i - j));
        dissimilarity += p * std::abs(i - j);

        //double mean_i = glcm.at<float>(i, 0);
        //double mean_j = glcm.at<float>(0, j);
        //double std_dev_i = std::sqrt(glcm.at<float>(i, 0) - mean_i * mean_i);
        //double std_dev_j = std::sqrt(glcm.at<float>(0, j) - mean_j * mean_j);
        //if (std_dev_i > 0.0 && std_dev_j > 0.0) {
        //  correlation += p * (i - mean_i) * (j - mean_j) / (std_dev_i * std_dev_j);
        //}
      }
    }

    features[0][theta] = contrast;
    features[1][theta] = energy; // 这计算的是角二阶矩？即 sqrt{ASM}
    features[2][theta] = homogeneity;
    features[3][theta] = dissimilarity;
  }
  // 4 个方向计算完

  // 计算均值
  for (int i = 0; i < 4; i++) {
    features[4][i] = std::accumulate(features[i].begin(), features[i].end(), 0.0) / 4;
  }

  return features;
}

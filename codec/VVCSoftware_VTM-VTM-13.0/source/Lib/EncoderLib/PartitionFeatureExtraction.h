#ifndef NEXTSOFTWARE_PARTITIONFEATUREETRATION_H
#define NEXTSOFTWARE_PARTITIONFEATUREETRATION_H

#include "CodingStructure.h"
#include "EncModeCtrl.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

enum GLCMGrayLevel {
  GLCMGRAY_4,
  GLCMGRAY_8,
  GLCMGRAY_16
};

/*
基础的CU编码信息的记录，只需要记录LGBM需要的特征，图像数据由 ImageDataManager 提供
*/
class CodingData {
public:
  int qp;
  int width;
  int height;
  int size;
  // 如果使用同一个模型对不同尺寸的CU进行预测时，这些数据才有用

  int QTD;
  int BTD;
  int TTD;
  int MTTD;
  int QTMTD;

  int currIntraMode; // 记录 67 中帧内预测模式之一
  int mrlIdx;
  int ispMode;
  int mtsFlag;
  int lfnstIdx;
  int mipFlag;
  int mipTransposedFlag; // 标记 -1 为缺失值

  int isModeVer;
  int intraPredAngleMode;

  int lastSplit;
  int qtBeforeBt;

  double currIntraCost;
  double currIntraFracBits;
  double currIntraDistoration;

  // 2024.16 新增
  int bestPredModeDCT2;
  
public:
  CodingData();
  void set_coding_data(const CodingStructure& cs, const Partitioner& partitioner, const ComprCUCtx& cuECtx);
  std::vector<int> get_coding_data();
  std::vector<double> get_intra_cost();
};

/*
纹理特征数据
*/
class TextureData {

public:
  double mean;
  double stddev;
  double diffStdDevVer;
  double diffStdDevHor;
  double Gx;
  double Gy;
  double ratioGxGy;
  double normGradient;

  double pixelSum;

  // 一些其他特征
  double entropy;
  double skewness; // 偏度
  double kurtosis; // 峰度

  // 水平二叉树划分之后的子块特征
  double BH_Above_mean;
  double BH_Above_stddev;
  double BH_Above_Gx;
  double BH_Above_Gy;
  double BH_Above_ratioGxGy;
  double BH_Above_normGradient;

  double BH_Below_mean;
  double BH_Below_stddev;
  double BH_Below_Gx;
  double BH_Below_Gy;
  double BH_Below_ratioGxGy;
  double BH_Below_normGradient;

  // 垂直二叉树
  double BV_Right_mean;
  double BV_Right_stddev;
  double BV_Right_Gx;
  double BV_Right_Gy;
  double BV_Right_ratioGxGy;
  double BV_Right_normGradient;

  double BV_Left_mean;
  double BV_Left_stddev;
  double BV_Left_Gx;
  double BV_Left_Gy;
  double BV_Left_ratioGxGy;
  double BV_Left_normGradient;

  // 水平三叉树
  double TH_Above_mean;
  double TH_Above_stddev;
  double TH_Above_Gx;
  double TH_Above_Gy;
  double TH_Above_ratioGxGy;
  double TH_Above_normGradient;

  double TH_Middle_mean;
  double TH_Middle_stddev;
  double TH_Middle_Gx;
  double TH_Middle_Gy;
  double TH_Middle_ratioGxGy;
  double TH_Middle_normGradient;

  double TH_Below_mean;
  double TH_Below_stddev;
  double TH_Below_Gx;
  double TH_Below_Gy;
  double TH_Below_ratioGxGy;
  double TH_Below_normGradient;

  // 垂直三叉树
  double TV_Right_mean;
  double TV_Right_stddev;
  double TV_Right_Gx;
  double TV_Right_Gy;
  double TV_Right_ratioGxGy;
  double TV_Right_normGradient;

  double TV_Middle_mean;
  double TV_Middle_stddev;
  double TV_Middle_Gx;
  double TV_Middle_Gy;
  double TV_Middle_ratioGxGy;
  double TV_Middle_normGradient;

  double TV_Left_mean;
  double TV_Left_stddev;
  double TV_Left_Gx;
  double TV_Left_Gy;
  double TV_Left_ratioGxGy;
  double TV_Left_normGradient;

  // 基于GLCM 的特征
// 5 行 4 列， 按行顺序为 contrast, energy, homogeneity, dissimilarity, 以及每个特征的均值。
  std::array<std::array<double, 4>, 5> GLCMFeatures;
  
public:
  TextureData();
  void set_texture_data(const std::unique_ptr<cv::Mat>& src, CTUImageDataManager* ctu_data, const CodingUnit& cuArea);
  std::vector<double> get_texture_data();
  void getBHFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y);
  void getBVFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y);
  void getTHFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y);
  void getTVFeatures(const cv::Mat& src, CTUImageDataManager* ctu_data, int x, int y);
};

/*
上下文信息 / 语境信息
*/
class ContextData {
public:
  int neighAvgQT; // 标记 -1 为缺失值
  int neighHigherQT;
  int neighAvgMTT;
  int neighHigherMTT;

  int neighAvgHorNum;
  int neighAvgVerNum;

public:
  ContextData();
  void set_context_data(const CodingStructure& cs, const Partitioner& partitioner);
  void parse_split_series(uint64_t splitSeries, std::vector<int>& splitVector, int depth);
  std::vector<int> get_context_data();
};


/*
* 管理全部的特征信息，处理特征数据，并且获得与LGBM的数据接口适配的数据
*/
class LGBMFeaturesManager {
public:
  std::unique_ptr<CodingData> coding_data;
  std::unique_ptr<TextureData> texture_data;
  std::unique_ptr<ContextData> context_data;

  int num_features;

  LGBMFeaturesManager();
  void init();
  void get_features(std::vector<double>& dst); // 获取最终的输入LGBM 模型的特征
  void features_preprocess(std::map<std::string, double>& src, std::vector<double>& dst);
};


/*
各项纹理特征的计算
*/

void calcMeanStdDev(const cv::Mat& src, double& mean, double& stddev);

void calcGradientFeatures(const cv::Mat& src, double& gradient_x, double& gradient_y, double& ratio_gx_gy, double& norm_gradient);

void calcDiffStdDevVer(const cv::Mat& src, double& diff_stddev_ver);

void calcDiffStdDevHor(const cv::Mat& src, double& diff_stddev_hor);

void calcEntropy(const cv::Mat& src, double& entropy);

void calcSkewnessKurtosis(const cv::Mat& src, double& skewness, double& kurtosis);

void GrayMagnitude(const cv::Mat& src, cv::Mat& dst, GLCMGrayLevel level);

std::array<std::array<double, 4>, 5> calcGLCMFeatures(const cv::Mat& src, int d, GLCMGrayLevel level);


#endif // !NEXTSOFTWARE_PARTITIONFEATUREETRATION_H

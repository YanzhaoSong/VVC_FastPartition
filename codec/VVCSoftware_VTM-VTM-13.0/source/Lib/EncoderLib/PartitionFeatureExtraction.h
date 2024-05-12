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
������CU������Ϣ�ļ�¼��ֻ��Ҫ��¼LGBM��Ҫ��������ͼ�������� ImageDataManager �ṩ
*/
class CodingData {
public:
  int qp;
  int width;
  int height;
  int size;
  // ���ʹ��ͬһ��ģ�ͶԲ�ͬ�ߴ��CU����Ԥ��ʱ����Щ���ݲ�����

  int QTD;
  int BTD;
  int TTD;
  int MTTD;
  int QTMTD;

  int currIntraMode; // ��¼ 67 ��֡��Ԥ��ģʽ֮һ
  int mrlIdx;
  int ispMode;
  int mtsFlag;
  int lfnstIdx;
  int mipFlag;
  int mipTransposedFlag; // ��� -1 Ϊȱʧֵ

  int isModeVer;
  int intraPredAngleMode;

  int lastSplit;
  int qtBeforeBt;

  double currIntraCost;
  double currIntraFracBits;
  double currIntraDistoration;

  // 2024.16 ����
  int bestPredModeDCT2;
  
public:
  CodingData();
  void set_coding_data(const CodingStructure& cs, const Partitioner& partitioner, const ComprCUCtx& cuECtx);
  std::vector<int> get_coding_data();
  std::vector<double> get_intra_cost();
};

/*
������������
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

  // һЩ��������
  double entropy;
  double skewness; // ƫ��
  double kurtosis; // ���

  // ˮƽ����������֮����ӿ�����
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

  // ��ֱ������
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

  // ˮƽ������
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

  // ��ֱ������
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

  // ����GLCM ������
// 5 �� 4 �У� ����˳��Ϊ contrast, energy, homogeneity, dissimilarity, �Լ�ÿ�������ľ�ֵ��
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
��������Ϣ / �ﾳ��Ϣ
*/
class ContextData {
public:
  int neighAvgQT; // ��� -1 Ϊȱʧֵ
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
* ����ȫ����������Ϣ�������������ݣ����һ����LGBM�����ݽӿ����������
*/
class LGBMFeaturesManager {
public:
  std::unique_ptr<CodingData> coding_data;
  std::unique_ptr<TextureData> texture_data;
  std::unique_ptr<ContextData> context_data;

  int num_features;

  LGBMFeaturesManager();
  void init();
  void get_features(std::vector<double>& dst); // ��ȡ���յ�����LGBM ģ�͵�����
  void features_preprocess(std::map<std::string, double>& src, std::vector<double>& dst);
};


/*
�������������ļ���
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

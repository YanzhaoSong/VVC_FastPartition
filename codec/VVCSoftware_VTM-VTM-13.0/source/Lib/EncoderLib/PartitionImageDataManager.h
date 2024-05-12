#ifndef NEXTSOFTWARE_PARTITIONIMAGEDATAMANAGER_H
#define NEXTSOFTWARE_PARTITIONIMAGEDATAMANAGER_H

#include "CodingStructure.h"

#include "PartitionParams.h"
#include <opencv2/opencv.hpp>
//#include <torch/script.h>


class CTUImageDataManager {
private:
  UnitArea ctuArea_;
  std::unique_ptr<cv::Mat> ctu_pixel_data_; // 存储原始图像数据
  std::unique_ptr<cv::Mat> gradient_x_; // 存储梯度
  std::unique_ptr<cv::Mat> gradient_y_;


public:
  CTUImageDataManager();
  void init_ctu_data(const PelUnitBuf& src, const UnitArea& ctuArea);
  void free_ctu_data();
  std::unique_ptr<cv::Mat>& get_pixel_data();
  std::array<double, 4> getGradientFeatures(int x, int y, int width, int height);
  UnitArea get_ctuArea();
};


class CUImageDataManager {
private:
  UnitArea cuArea_;
  int x_in_ctu_;
  int y_in_ctu_;

  std::unique_ptr<cv::Mat> cu_pixel_data_;

public:
  CUImageDataManager();
  void init_cu_data(const UnitArea& cuArea, CTUImageDataManager& ctu_data);
  void free_cu_data();
  cv::Mat image_preprocess();
  std::unique_ptr<cv::Mat>& get_pixel_data();

};

#endif // !NEXTSOFTWARE_PARTITIONIMAGEDATAMANAGER_H

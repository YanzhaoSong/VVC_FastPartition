#include "PartitionImageDataManager.h"

CTUImageDataManager::CTUImageDataManager()
  : ctuArea_ {}
  , ctu_pixel_data_ {nullptr}
{}

void CTUImageDataManager::init_ctu_data(const PelUnitBuf& src, const UnitArea& ctuArea) {
  ctuArea_ = ctuArea;

  if (ctu_pixel_data_ == nullptr) {
    ctu_pixel_data_ = std::make_unique<cv::Mat>(cv::Mat(ctuArea.lheight(), ctuArea.lwidth(), CV_8UC1)); // 分配内存空间
  }

  for (int i = 0; i < ctuArea.lheight(); ++i) {
    for (int j = 0; j < ctuArea.lwidth(); ++j) {
      (*ctu_pixel_data_).at<uint8_t>(i, j) = src.Y().at(ctuArea.lx() + j, ctuArea.ly() + i) >> 2;
    }
  } // 循环访问，初始化图像数据

  // 初始化图像数据的同时，计算梯度信息，以供后续使用
  if (gradient_x_ == nullptr || gradient_y_ == nullptr) {
    gradient_x_ = std::make_unique<cv::Mat>(cv::Mat(ctuArea.lheight(), ctuArea.lwidth(), CV_16S));
    gradient_y_ = std::make_unique<cv::Mat>(cv::Mat(ctuArea.lheight(), ctuArea.lwidth(), CV_16S));
  }

  cv::Sobel(*ctu_pixel_data_, *gradient_x_, CV_16S, 1, 0, 3);
  cv::Sobel(*ctu_pixel_data_, *gradient_y_, CV_16S, 0, 1, 3);

}


void CTUImageDataManager::free_ctu_data() {
  ctu_pixel_data_.release();
  gradient_x_.release();
  gradient_y_.release();
}

std::unique_ptr<cv::Mat>& CTUImageDataManager::get_pixel_data() {
  return ctu_pixel_data_;
}

std::array<double, 4> CTUImageDataManager::getGradientFeatures(int x, int y, int width, int height) {
  std::array<double, 4> gradient_features{};

  double gradient_x = cv::sum(cv::abs((*gradient_x_)(cv::Rect(x, y, width, height))))[0];
  double gradient_y = cv::sum(cv::abs((*gradient_y_)(cv::Rect(x, y, width, height))))[0];
  double ratio_gx_gy = gradient_x / (gradient_y + 0.00000000001);  // 防止出现除0的情况
  double norm_gradient = (gradient_x + gradient_y) / (width * height);
  
  gradient_features = { gradient_x, gradient_y, ratio_gx_gy, norm_gradient };

  return gradient_features;
}

UnitArea CTUImageDataManager::get_ctuArea() {
  return ctuArea_;
}

/*
CU 级的图像数据方法定义
*/
CUImageDataManager::CUImageDataManager() 
  : cuArea_ {}
  , x_in_ctu_ {0}
  , y_in_ctu_ {0}
  , cu_pixel_data_ {nullptr}
{
}

void CUImageDataManager::init_cu_data(const UnitArea& cuArea, CTUImageDataManager& ctu_data) {
  cuArea_ = cuArea;
  auto ctuArea = ctu_data.get_ctuArea();
  x_in_ctu_ = cuArea_.lx() - ctuArea.lx();
  y_in_ctu_ = cuArea_.ly() - ctuArea.ly();
  
  auto rect = cv::Rect(x_in_ctu_, y_in_ctu_, cuArea_.lwidth(), cuArea_.lheight());
  cu_pixel_data_ = std::make_unique<cv::Mat>((*ctu_data.get_pixel_data())(rect)); // 分配内存，并且初始化。
}

void CUImageDataManager::free_cu_data() {
  cu_pixel_data_.release();
}

cv::Mat CUImageDataManager::image_preprocess() {
  cv::Mat result;  // 由于原数据后续还需使用，因此这里进行的变换不能修改原数据

  // 2023.11.27 added
  //std::string path = std::to_string(cuArea_.lx()) + "_" + std::to_string(cuArea_.ly()) + "_" + std::to_string(cuArea_.lwidth()) + "_" + std::to_string(cuArea_.lheight()) + ".jpg";
  //cv::imwrite(path, *cu_pixel_data_);
  // 2023.11.27 end

  // to_tensor ，会将数据转换到 0，1范围内
  // 归一化到0到1的范围
  //cv::normalize(*cu_pixel_data_, *cu_pixel_data_, 0, 1, cv::NORM_MINMAX);
  //std::cout << (int)cu_pixel_data_->at<uint8_t>(0, 0) << std::endl;

  cu_pixel_data_->convertTo(result, CV_32FC1, 1.0 / 255, 0);  // 注意：只有这种方式才行，其他处理方式不对？

  return result;
}

std::unique_ptr<cv::Mat>& CUImageDataManager::get_pixel_data() {
  return cu_pixel_data_;
}


#include "PartitionUtils.h"
#include <algorithm>
#include <vector>

std::string get_shape(const UnitArea& cuArea) {
  return std::to_string(cuArea.lwidth()) + "x" + std::to_string(cuArea.lheight());
}

bool need_to_prediction(Partitioner& partitioner, const CodingStructure& cs, const UnitArea& cuArea) {
  PartSplit implicitSplit = partitioner.getPartStack().back().checkdIfImplicit ? partitioner.getPartStack().back().implicitSplit : partitioner.getImplicitSplit(cs);
  bool isBoundary = (implicitSplit != CU_DONT_SPLIT); // 如果默认的划分模式不是 CU_DONT_SPLIT， 则为边界

  bool is_valid_shape = (std::find(ShapeList.begin(), ShapeList.end(), get_shape(cuArea)) != ShapeList.end());

  return (is_valid_shape && !isBoundary && partitioner.chType == CHANNEL_TYPE_LUMA);
}

bool is_in(const EncTestModeType& e, const std::vector<EncTestModeType>& v) {
  if (std::find(v.begin(), v.end(), e) != v.end())
    return true;
  return false;
}


std::vector<EncTestModeType> apply_mask(const std::vector<EncTestModeType>& originalVector, const std::vector<bool>& mask) {
  std::vector<EncTestModeType> resultVector;

  // 确保两个向量长度相同
  if (originalVector.size() != mask.size()) {
    std::cerr << "Error: Vector and mask must have the same length." << std::endl;
    return resultVector;
  }

  // 遍历向量和掩码
  for (std::size_t i = 0; i < originalVector.size(); ++i) {
    // 如果掩码不为零，将对应位置的元素添加到结果向量中
    if (mask[i]) {
      resultVector.push_back(originalVector[i]);
    }
  }

  return resultVector;
}

std::vector<bool> generate_mask_k(const std::vector<double>& src, std::size_t k, bool intra) {
  std::vector<bool> mask(src.size(), 1); // 生成同尺寸的全1 vector

  // 创建一个索引向量并对原始向量进行排序
  std::vector<std::size_t> index(src.size());
  for (std::size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  } // 生成 index 的 vector， {0， 1， 2， 3， 4， 5}

  std::sort(index.begin(), index.end(), [&src](std::size_t i, std::size_t j) {
    return src[i] > src[j];
    });

  // 将前 k 个位置设置为 0
  for (std::size_t i = 0; i < k && i < src.size(); ++i) {
    mask[index[i]] = 0;
  }

  // 无论如何， NS 模式都需要进行测试
  if (intra) {
    mask[0] = 0;
  }
 
  return mask;
}

std::vector<bool> generate_mask_multi_label(const std::vector<double>& src, double th) {
  std::vector<bool> mask(src.size(), 1);

  for (int i = 0; i < src.size(); i++)
    if (src[i] >= th)
      mask[i] = 0;

  mask[0] = 0; // 任何情况下，帧内模式都需要进行测试，否则会出现错误？

  return mask;
}

std::vector<bool> generate_mask_with_threshold(const std::vector<double>& src, double threshold) {
  std::vector<bool> mask(src.size(), 1);
  double max_value = *std::max_element(src.begin(), src.end()); // max_element() 函数返回的是指针，指向元素的位置

  for (int i = 0; i <= src.size(); ++i) {
    if ((src[i] / max_value) >= threshold) {
      mask[i] = 0;
    }
  }

  mask[0] = 0; // 由于LGBM 预测是在intra测试之后，因此保留帧内的测试。
  return mask;
}

// 值最大位置的索引
int argmax(const std::vector<double>& vec) {
  if (vec.empty()) {
    std::cerr << "Error: Vector is empty." << std::endl;
    return -1; // 返回一个表示错误的值
  }

  // 使用 std::max_element 获取最大值的迭代器
  auto maxElementIterator = std::max_element(vec.begin(), vec.end());

  // 使用 std::distance 获取最大值的索引
  int index = std::distance(vec.begin(), maxElementIterator);

  return index;
}


// vector 对应位置相乘
std::vector<bool> multiply_vectors(const std::vector<bool>& vector1, const std::vector<bool>& vector2) {
  std::vector<bool> result_vector;

  // 确保两个向量长度相同
  if (vector1.size() != vector2.size()) {
    std::cerr << "Error: Vectors must have the same length." << std::endl;
    return result_vector;
  }

  // 遍历向量并对应位置相乘
  for (std::size_t i = 0; i < vector1.size(); ++i) {
    result_vector.push_back(vector1[i] * vector2[i]);
  }

  return result_vector;
}

bool is_file_exist(const std::string& filepath) {
  FILE* file = std::fopen(filepath.c_str(), "r");
  if (file) {
    std::fclose(file);
    return true;
  }
  return false;
}
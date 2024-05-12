#include "PartitionUtils.h"
#include <algorithm>
#include <vector>

std::string get_shape(const UnitArea& cuArea) {
  return std::to_string(cuArea.lwidth()) + "x" + std::to_string(cuArea.lheight());
}

bool need_to_prediction(Partitioner& partitioner, const CodingStructure& cs, const UnitArea& cuArea) {
  PartSplit implicitSplit = partitioner.getPartStack().back().checkdIfImplicit ? partitioner.getPartStack().back().implicitSplit : partitioner.getImplicitSplit(cs);
  bool isBoundary = (implicitSplit != CU_DONT_SPLIT); // ���Ĭ�ϵĻ���ģʽ���� CU_DONT_SPLIT�� ��Ϊ�߽�

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

  // ȷ����������������ͬ
  if (originalVector.size() != mask.size()) {
    std::cerr << "Error: Vector and mask must have the same length." << std::endl;
    return resultVector;
  }

  // ��������������
  for (std::size_t i = 0; i < originalVector.size(); ++i) {
    // ������벻Ϊ�㣬����Ӧλ�õ�Ԫ����ӵ����������
    if (mask[i]) {
      resultVector.push_back(originalVector[i]);
    }
  }

  return resultVector;
}

std::vector<bool> generate_mask_k(const std::vector<double>& src, std::size_t k, bool intra) {
  std::vector<bool> mask(src.size(), 1); // ����ͬ�ߴ��ȫ1 vector

  // ����һ��������������ԭʼ������������
  std::vector<std::size_t> index(src.size());
  for (std::size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  } // ���� index �� vector�� {0�� 1�� 2�� 3�� 4�� 5}

  std::sort(index.begin(), index.end(), [&src](std::size_t i, std::size_t j) {
    return src[i] > src[j];
    });

  // ��ǰ k ��λ������Ϊ 0
  for (std::size_t i = 0; i < k && i < src.size(); ++i) {
    mask[index[i]] = 0;
  }

  // ������Σ� NS ģʽ����Ҫ���в���
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

  mask[0] = 0; // �κ�����£�֡��ģʽ����Ҫ���в��ԣ��������ִ���

  return mask;
}

std::vector<bool> generate_mask_with_threshold(const std::vector<double>& src, double threshold) {
  std::vector<bool> mask(src.size(), 1);
  double max_value = *std::max_element(src.begin(), src.end()); // max_element() �������ص���ָ�룬ָ��Ԫ�ص�λ��

  for (int i = 0; i <= src.size(); ++i) {
    if ((src[i] / max_value) >= threshold) {
      mask[i] = 0;
    }
  }

  mask[0] = 0; // ����LGBM Ԥ������intra����֮����˱���֡�ڵĲ��ԡ�
  return mask;
}

// ֵ���λ�õ�����
int argmax(const std::vector<double>& vec) {
  if (vec.empty()) {
    std::cerr << "Error: Vector is empty." << std::endl;
    return -1; // ����һ����ʾ�����ֵ
  }

  // ʹ�� std::max_element ��ȡ���ֵ�ĵ�����
  auto maxElementIterator = std::max_element(vec.begin(), vec.end());

  // ʹ�� std::distance ��ȡ���ֵ������
  int index = std::distance(vec.begin(), maxElementIterator);

  return index;
}


// vector ��Ӧλ�����
std::vector<bool> multiply_vectors(const std::vector<bool>& vector1, const std::vector<bool>& vector2) {
  std::vector<bool> result_vector;

  // ȷ����������������ͬ
  if (vector1.size() != vector2.size()) {
    std::cerr << "Error: Vectors must have the same length." << std::endl;
    return result_vector;
  }

  // ������������Ӧλ�����
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
// 2023.11.16 added
// ���ܣ��洢�õ��ĸ��ֹ��ܺ���

#ifndef NEXTSOFTWARE_PARTITIONUTILS_H
#define NEXTSOFTWARE_PARTITIONUTILS_H

#include <cstdio>

#include "Unit.h"
#include "UnitPartitioner.h"
#include "CodingStructure.h"
#include "EncModeCtrl.h"

#include "PartitionParams.h"

std::string get_shape(const UnitArea& cuArea);

bool need_to_prediction(Partitioner& partitioner, const CodingStructure& cs, const UnitArea& cuArea); // �жϵ�ǰ��CU�Ƿ���������Ҫ����Ԥ��ķ�Χ��

// ��ӡ����
template<typename T>
void print_vector(const std::vector<T>& v) {
  for (auto& e : v) {
    std::cout << e << ", ";
  }
  std::cout << std::endl;
}

bool is_in(const EncTestModeType& e, const std::vector<EncTestModeType>& v);

std::vector<EncTestModeType> apply_mask(const std::vector<EncTestModeType>& originalVector, const std::vector<bool>& mask);

std::vector<bool> generate_mask_k(const std::vector<double>& src, std::size_t k, bool intra=true); // ���ݸ����ĸ��ʣ��Լ� k ֵ������mask�����и�����С�� 6 - k ��λ��Ϊ1����Ҫ��ȥ��

std::vector<bool> generate_mask_multi_label(const std::vector<double>& src, double th=0.5);

std::vector<bool> generate_mask_with_threshold(const std::vector<double>& src, double threshold);

int argmax(const std::vector<double>& vec);

std::vector<bool> multiply_vectors(const std::vector<bool>& vector1, const std::vector<bool>& vector2);

bool is_file_exist(const std::string& filepath);


#endif // !NEXTSOFTWARE_PARTITIONUTILS_H

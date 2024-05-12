#include "PartitionCNNManager.h"
#include "PartitionUtils.h"

CNNPredictionManager::CNNPredictionManager(int qp) {
  cnn_params_ = &FastPartitionModeParams[MyFastPartitionMode];
  cnn_params_->qp = qp;
  cnn_params_->shape_list = {
    "32x32",
    //"32x16",
    //"16x32",
    //"32x8",
    //"8x32",
    //"16x16",
  };

  // ��������Ҫ����һ�� static �Ķ�����˱���Ҫ��ʼ��ʱ�ͼ���ģ��
  load_model(); 
}

void CNNPredictionManager::load_model() {
  for (auto& shape : cnn_params_->shape_list) {
    std::string model_path = cnn_params_->cnn_model_root_path + "QP" + std::to_string(cnn_params_->qp) + "_" + shape + ".pt"; // ·����ʽ��models\\cnn_models\\QP37_32x32.pt

    // �����ж�ģ���Ƿ����
    if (!is_file_exist(model_path)) {
      std::cout << "cnn model does not exist: " << model_path << std::endl;
      exit(-1);
    }

    try
    {
      cnn_models_[shape] = torch::jit::load(model_path); // ÿһ���ߴ�洢һ��ģ��
    }
    catch (const c10::Error& e)
    {
      std::cout << "error loading the cnn model." << std::endl;
    }
  }
  // 
}

void CNNPredictionManager::get_input_tensor(const std::unique_ptr<cv::Mat>& src, std::vector<torch::jit::IValue>& dst) {
  dst.clear(); // ����������е����ݡ�

  torch::Tensor input_tensor0 = torch::from_blob((*src).data, { 1, (*src).rows, (*src).cols, 1 }, torch::kFloat);
  input_tensor0 = input_tensor0.permute({ 0, 3, 1, 2 }); // ��ͨ����˳��ת��Ϊ Torch ��Ҫ��ĸ�ʽ
  dst.push_back(input_tensor0);
}

void CNNPredictionManager::predict(const UnitArea& cuArea, const std::unique_ptr<cv::Mat>& src, std::vector<double>& output_prob) {
  std::vector<torch::jit::IValue> input_tensor{};
  get_input_tensor(src, input_tensor);

  std::string shape = get_shape(cuArea);
  auto output = cnn_models_[shape].forward(input_tensor).toTensor();
  //std::cout << output << std::endl;

  // ģ�������δ����softmax��������Ҫ����
  output = output.softmax(1);

  // ����Ƕ��ǩ����Ӧ���� sigmoid
  //output = output.sigmoid();
  //std::cout << "CNN: " << output << std::endl;

  //if (get_shape(cuArea) == "32x16") {
  //  std::cout << output << std::endl;
  //  exit(0);
  //}

  output_prob = std::vector<double> (output.data_ptr<float>(), output.data_ptr<float>() + output.numel()); // �������Ϊfloat�� double �ᱨ��
}

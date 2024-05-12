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

  // 由于我需要定义一个 static 的对象，因此必须要初始化时就加载模型
  load_model(); 
}

void CNNPredictionManager::load_model() {
  for (auto& shape : cnn_params_->shape_list) {
    std::string model_path = cnn_params_->cnn_model_root_path + "QP" + std::to_string(cnn_params_->qp) + "_" + shape + ".pt"; // 路径样式：models\\cnn_models\\QP37_32x32.pt

    // 首先判断模型是否存在
    if (!is_file_exist(model_path)) {
      std::cout << "cnn model does not exist: " << model_path << std::endl;
      exit(-1);
    }

    try
    {
      cnn_models_[shape] = torch::jit::load(model_path); // 每一个尺寸存储一个模型
    }
    catch (const c10::Error& e)
    {
      std::cout << "error loading the cnn model." << std::endl;
    }
  }
  // 
}

void CNNPredictionManager::get_input_tensor(const std::unique_ptr<cv::Mat>& src, std::vector<torch::jit::IValue>& dst) {
  dst.clear(); // 首先清除其中的数据。

  torch::Tensor input_tensor0 = torch::from_blob((*src).data, { 1, (*src).rows, (*src).cols, 1 }, torch::kFloat);
  input_tensor0 = input_tensor0.permute({ 0, 3, 1, 2 }); // 将通道的顺序转换为 Torch 中要求的格式
  dst.push_back(input_tensor0);
}

void CNNPredictionManager::predict(const UnitArea& cuArea, const std::unique_ptr<cv::Mat>& src, std::vector<double>& output_prob) {
  std::vector<torch::jit::IValue> input_tensor{};
  get_input_tensor(src, input_tensor);

  std::string shape = get_shape(cuArea);
  auto output = cnn_models_[shape].forward(input_tensor).toTensor();
  //std::cout << output << std::endl;

  // 模型输出并未进行softmax，这里需要进行
  output = output.softmax(1);

  // 如果是多标签，就应该是 sigmoid
  //output = output.sigmoid();
  //std::cout << "CNN: " << output << std::endl;

  //if (get_shape(cuArea) == "32x16") {
  //  std::cout << output << std::endl;
  //  exit(0);
  //}

  output_prob = std::vector<double> (output.data_ptr<float>(), output.data_ptr<float>() + output.numel()); // 这里必须为float， double 会报错。
}

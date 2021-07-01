//
// Created by dp on 2021/6/29.
//

#include "TorchScript.h"


at::Tensor TorchScript::normPRED(at::Tensor output) {
    at::Tensor max = torch::max(output);
    at::Tensor min = torch::min(output);

    at::Tensor dn = (output-min)/(max-min);
    return dn;
}


cv::Mat TorchScript::inference(std::string imgPath, std::string modelPath){ //std::vector<float> mean_={0}, std::vector<float> std_={0}) {
    //
    cv::Mat image_src = cv::imread(imgPath);

    cv::resize(image_src, image_src, cv::Size(320,320));
    cv::cvtColor(image_src, image_src, cv::COLOR_BGR2RGB);
    //图像转换成tensor的格式
    torch::Tensor tensor_image_src = torch::from_blob(image_src.data, {image_src.cols, image_src.rows, 3}, torch::kByte);//此处必须为torch::kByte的导入形式
    tensor_image_src = tensor_image_src.permute({2,0,1});
    tensor_image_src = tensor_image_src.toType(torch::kFloat);
    tensor_image_src = tensor_image_src.div(255);
    tensor_image_src = tensor_image_src.unsqueeze(0);
    torch::Tensor scr = tensor_image_src.to(torch::kCUDA);

    torch::jit::script::Module model = torch::jit::load(modelPath); //导入torchscript模型

    // 1、如果输出有多个结果，则按照如下的形式
    // 2、如果只有一个输出，可以使用 auto outputs = model.forward({scr}).toTensor(),然后解析outputs;
    auto outputs = model.forward({scr}).toTuple()->elements();

    torch::Tensor pred = outputs[0].toTensor();

    torch::Tensor res_tensor = (pred * torch::ones_like(scr));
    res_tensor = normPRED(res_tensor);
    res_tensor = res_tensor.squeeze().detach();
    res_tensor = res_tensor.mul(255).clamp(0,255).to(torch::kU8); //图像*255，然后拉倒0，255
    res_tensor = res_tensor.to(torch::kCPU);

    //将torch：：Tensor转成cv::Mat
    cv::Mat resultImg(res_tensor.size(1), res_tensor.size(2), CV_8UC3);
    std::memcpy((void *)resultImg.data, res_tensor.cpu().data_ptr(), sizeof(torch::kU8) * res_tensor.numel());

    cv::resize(resultImg, resultImg, cv::Size(image_src.cols, image_src.rows), cv::INTER_LINEAR);
    return resultImg;
}
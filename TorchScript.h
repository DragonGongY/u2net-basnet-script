//
// Created by dp on 2021/6/29.
//
#include "Configs.h"


#ifndef TEST_TORCHSCRIPT_H
#define TEST_TORCHSCRIPT_H

class TorchScript {
    public:
        cv::Mat inference(std::string imgPath, std::string modelPath);
        at::Tensor normPRED(at::Tensor output);
};

#endif //TEST_TORCHSCRIPT_H

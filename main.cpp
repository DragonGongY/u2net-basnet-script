#include "Configs.h"
#include "TorchScript.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
//    std::cout <<torch::cuda::is_availbale() << std::endl;
//    cv::Mat src = cv::imread("/home/dp/Downloads/test.jpg");
//    cv::imshow("src", src);
//    cv::waitKey();

    std::string modelPath = "/media/dp/DATA/huihua_robot/U-2-Net-master/script.pt";
    TorchScript torchscript;
    cv::Mat output = torchscript.inference(
            "/home/dp/Downloads/test/1.jpg",
            modelPath);
    cv::imshow("res", output);
    cv::waitKey();

//    torch::Tensor tensor = torch::ones(3);
//    std::cout << tensor << std::endl;
    return 0;
}

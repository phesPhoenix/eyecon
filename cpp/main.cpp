#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera" << std::endl;
        return -1;
    }

    std::cout << "Camera opened successfully!" << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("EyeCon - Camera Feed", frame);

        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
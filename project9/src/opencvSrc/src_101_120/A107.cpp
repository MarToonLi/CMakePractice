#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>

using namespace std;
using namespace cv;

namespace NA107 {
    // 线程
    void output(int i)
    {
        cout << i << endl;
    }

    void A107()
    {

        for (uint8_t i = 0; i < 4; i++)
        {
            thread t(output, i);
            t.detach();
        }

        getchar(); // 接收终端的字符


        //cv::Sobel();

        return;
    }
}

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>


#pragma execution_character_set("utf-8") 

namespace NA108 {
    // 线程

    void A108_solver()
    {

        int a = 131;
        int c = a * 0.5;

        LOGD("a: {}; c: {};", a, c);


        cv::Mat imgSrc = cv::imread("F:\\Projects\\Opencv-100-Questions\\project1\\OpencvTestImg\\A104.png");

        LOGD("imgSrc.width: {}; imgSrc.height: {}; imgSrc.type: {};", imgSrc.cols, imgSrc.rows, imgSrc.type());

        return;
    }
}

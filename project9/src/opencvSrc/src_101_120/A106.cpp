#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include "A_101_120.h"


using namespace std;
using namespace cv;

namespace NA106 {

    //SimpleBlobDetector
    cv::SimpleBlobDetector::Params pBLOBDetector;
    cv::Mat src, src_ori;
    int iThStep = 10;
    int iMinth = 10;
    int iMaxth = 200;
    int iMinBt = 10;
    int iMinar = 10;
    int iMaxar = 1500;
    int iMinCir = 0;
    int iMinIne = 0;
    int iMinCon = 0;

    void detect()
    {
        pBLOBDetector.thresholdStep = iThStep;
        pBLOBDetector.minThreshold = iMinth;
        pBLOBDetector.maxThreshold = iMaxth;
        pBLOBDetector.minRepeatability = 2;
        pBLOBDetector.minDistBetweenBlobs = iMinBt;
        pBLOBDetector.filterByColor = false;
        pBLOBDetector.blobColor = 0;
        //斑点面积
        pBLOBDetector.filterByArea = false;
        pBLOBDetector.minArea = iMinar;
        pBLOBDetector.maxArea = iMaxar;
        //斑点圆度
        pBLOBDetector.filterByCircularity = false;
        pBLOBDetector.minCircularity = iMinCir * 0.01;
        pBLOBDetector.maxCircularity = (float)3.40282e+038;
        //斑点惯性率
        pBLOBDetector.filterByInertia = false;
        pBLOBDetector.minInertiaRatio = iMinIne * 0.01;
        pBLOBDetector.maxInertiaRatio = (float)3.40282e+038;
        //斑点凸度
        pBLOBDetector.filterByConvexity = false;
        pBLOBDetector.minConvexity = iMinCon * 0.01;
        pBLOBDetector.maxConvexity = (float)3.40282e+038;
        //*用参数创建对象
        cv::Ptr<cv::SimpleBlobDetector> blob = cv::SimpleBlobDetector::create(pBLOBDetector);
        //Ptr<SimpleBlobDetector> blob=SimpleBlobDetector::create();//默认参数创建
        //*blob检测
        vector<cv::KeyPoint> key_points;
        //Mat dst;
        //cvtColor(src, dst, COLOR_RGB2GRAY);

        blob->detect(src, key_points);
        cv::Mat outImg;
        //src.copyTo(outImg);

        //绘制结果
        cv::drawKeypoints(src, key_points, outImg, cv::Scalar(0, 0, 255));   // size即是半径
        cv::imshow("blob", outImg);
        cv::waitKey(0);

    }

    void A106()
    {
        src_ori = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\canny2.png", cv::IMREAD_GRAYSCALE);

        cv::resize(src_ori, src, cv::Size(1000, 1000));

        if (src.empty()) {
            cout << "Cannot load image" << endl;
            return;
        }

        detect();

        //cv::imshow("src", src);
        //cv::namedWindow("Detect window", cv::WINDOW_NORMAL);
        //cv::createTrackbar("最小圆度", "Detect window", &iMinCir, 100, detect);
        //cv::createTrackbar("最小惯性率", "Detect window", &iMinIne, 100, detect);
        //cv::createTrackbar("最大凸度", "Detect window", &iMinCon, 100, detect);
        //cv::createTrackbar("阈值步距", "Detect window", &iThStep, 100, detect);
        //cv::createTrackbar("最小阈值", "Detect window", &iMinth, 255, detect);
        //cv::createTrackbar("最大阈值", "Detect window", &iMaxth, 255, detect);
        //cv::createTrackbar("最小距离", "Detect window", &iMinBt, 255, detect);
        //cv::createTrackbar("最小面积", "Detect window", &iMinar, 1000, detect);
        //cv::createTrackbar("最大面积", "Detect window", &iMaxar, 5000, detect);
    }
}

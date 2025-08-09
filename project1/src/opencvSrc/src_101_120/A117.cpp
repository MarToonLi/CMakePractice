#include <omp.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <thread>
#include <iostream>
#include "A_101_120.h"
#include <stdio.h>
#include <chrono>
#include <future>
#include <algorithm>
#include <typeinfo>

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <utility>


#include <iostream>
#include <thread>
#include <vector>
#include <windows.h>
#include <queue>
#include <list>
#include <mutex>

using namespace std;
using namespace cv;
using namespace ml;

#pragma execution_character_set("utf-8") 



/**********************************
* C++线程知识：
************************************/
namespace NA117 {


    /**********************************
	* Case1: 线程模仿算子计算队列和图像保存队列的处理效率随外部系统应用运行的变化；
    ************************************/

    namespace Case1 {
    
        struct ImgData
        {
            cv::Mat srcImg;
            int index;
        };

        list<ImgData> lImages;


        std::mutex mtx; // 全局互斥量


        // 一个简单的函数，作为线程的入口函数
        void f1(int timeStep)
        {
            int imageIdx = 0;
            cv::Mat src = cv::imread("D:\\1.bmp", 1);  // 读取数据

            clock_t nowTime, endTime, lastTime;
            nowTime = clock();
            endTime = nowTime + 1000 * timeStep;       // 延后 timtStep这么多秒
            lastTime = nowTime;

            int MIN = 5; //ms一个等待间隔，图像大概算作为帧率

            // 当前线程一共会添加N个图像
            // 添加这N个图像的时间是 1000 * timestep ms；
            // 假如图像加载不需要时间，则图像间的加载时间是min； 而N = 1000 * timestep / min;
            do {
                nowTime = clock();

                if (nowTime >= lastTime + MIN) // 如果
                {
                    imageIdx++;
                    ImgData curIData;
                    curIData.srcImg = src.clone();
                    curIData.index = imageIdx;

                    //cout << "input iamgeIdx:" << imageIdx << endl;
                    //infoLogger.TraceKeyInfo("[%s(%s)(%d)]-%s", __FILE__, __FUNCTION__, __LINE__, "关键信息"+ "input iamgeIdx-" +to_string(imageIdx));

                    //LOGD("{}-{}", "input imageIdx", imageIdx);

                    mtx.lock();
                    lImages.push_back(curIData);
                    mtx.unlock();

                    lastTime = nowTime;
                }
            } while (nowTime <= endTime);

            LOGD("image intotal:", imageIdx);
        }

        void f2(int Z)
        {
            int breakTime = 1000;  // 1s以上的等待间隔，即break;
            int processtime = 2000;  // 1s以上的等待间隔，即break;
            clock_t te, ts;
            ts = clock();

            while (1)
            {
                int size = 0;
                if (lImages.size() == 0)
                {
                    te = clock();
                    if (te - ts > breakTime)  //等待超过1000ms，即停止
                        break;
                }
                else
                {
                    mtx.lock();
                    ImgData curIData = lImages.front();
                    lImages.pop_front();
                    mtx.unlock();

                    Mat curImg = curIData.srcImg;
                    int curIdx = curIData.index;



                    LARGE_INTEGER  large_interger;

                    __int64  c1, c2;
                    QueryPerformanceCounter(&large_interger);
                    c1 = large_interger.QuadPart;


                    clock_t ts1, te1;
                    ts1 = clock();

                    Mat kernal = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
                    morphologyEx(curImg, curImg, MORPH_DILATE, Mat());
                    morphologyEx(curImg, curImg, MORPH_DILATE, Mat());
                    putText(curImg, to_string(curIdx), Point(50, 150), 2, 5, Scalar(0, 255, 0), 2, 8, false);
                    //imwrite("resdata\\" + to_string(curIdx) + ".jpg", curImg);

                    //cout << "output image idx:" << curIdx << endl;

                    QueryPerformanceCounter(&large_interger);
                    c2 = large_interger.QuadPart;

                    //int c = c2 - c1;
                    //cout << (c2 - c1) * 1000 / dff << endl;

                    te1 = clock();
                    //LOGD("%s-line%d %s-%d-proc time %d ms %lf ms", __FUNCTION__, __LINE__, "output imageIdx", curIdx, te1 - ts1, (c2 - c1) * 1.0 / 10000);
                    //LOGD("{}-{}-proc time {} ms {} ms", "output imageIdx", curIdx, te1 - ts1, (c2 - c1) * 1.0 / 10000);


                    if (curIdx % 100 == 0)
                    {
                        //cout << curIdx << "  -imageList Size-  " << lImages.size() << endl;
                        LOGD("{} -imageList Size- {} ( {};   {};);", curIdx, lImages.size(), te1 - ts1, (c2 - c1) * 1.0 / 10000);
                    }



                    ts = clock();
                }
            }

            return;
        }
    
    }
    

    void A117_solver()
    {


        list<Case1::ImgData>().swap(Case1::lImages);

        // 使用函数指针创建线程
        thread thread1(Case1::f1, 600);//20s


        // 使用函数对象创建线程
        thread thread2(Case1::f2, 50);

        // 等待所有线程完成
        thread1.join(); // 等待线程 th1 完成
        thread2.join(); // 等待线程 th2 完成


        return;
    }


}

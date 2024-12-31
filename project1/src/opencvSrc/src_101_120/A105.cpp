#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


cv::Mat src1, src_gray, temp_Src;

int maxCorners = 10;
const int MAXTRACKBAR = 25;

const std::string source_window = "Source Image";

void trackFeatures(int, void*);
using namespace cv;
using namespace std;


//! 亚像素goodFeaturesToTrack和cornerSubPix
void A105()
{
    src1 = cv::imread("F:\\Projects\\Opencv-100-Questions\\OpencvTestImg\\canny2.png");
    
    cv::resize(src1, temp_Src, cv::Size(1000, 1000));


    cv::cvtColor(temp_Src, src_gray, cv::COLOR_BGR2GRAY);

    cv::namedWindow(source_window);

    cv::createTrackbar("Max corners:", source_window, &maxCorners, MAXTRACKBAR, trackFeatures);

    trackFeatures(0, 0);

    cv::waitKey();
}

void trackFeatures(int, void*)
{
    maxCorners = MAX(maxCorners, 1);
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;             //! 它乘以最大特征值的乘积作为阈值对图像进行分割
    double minDistance = 10;                //! 像素级别
    int blockSize = 11, gradientSize = 11;  //! blocksize需要参考希望发现的角点区域的大小，太小和太大都不好；
    bool useHarrisDetector = true;          //! 
    double k = 0.04;

    cv::Mat copy = temp_Src.clone();


    cv::goodFeaturesToTrack(src_gray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        cv::Mat(),
        blockSize,
        gradientSize,
        useHarrisDetector,
        k);


    std::cout << "** Number of corners detected: " << corners.size() << "\n";
    int radius = 8;
    for (size_t i = 0; i < corners.size(); i++)
    {
        std::cout << " -- Original Corner Detected [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")\n";
        cv::circle(copy, corners[i], radius, cv::Scalar(0, 255, 0), cv::FILLED);
    }


    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);   //误差0.001 迭代次数40；


    std::vector<cv::Point2f> refinedCorners(corners);                          //! 深拷贝
    cv::cornerSubPix(src_gray, refinedCorners, winSize, zeroZone, criteria);

    for (size_t i = 0; i < refinedCorners.size(); i++)
    {
        std::cout << " -- Refined Corner [" << i << "]  (" << refinedCorners[i].x << "," << refinedCorners[i].y << ")\n";
        cv::circle(copy, refinedCorners[i], radius, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    cv::namedWindow(source_window.c_str());
    cv::imshow(source_window, copy);


    vector<cv::KeyPoint> key_points;

}


// greaterThanPtr：比较两个指针的值大小
template <typename T>
bool greaterThanPtr(const T* ptr1, const T* ptr2) {
    return *ptr1 > *ptr2;
}





/** @brief Same as above, but returns also quality measure of the detected corners.

@param image Input 8-bit or floating-point 32-bit, single-channel image.                                   8UC1  32FC1 单通道
@param corners Output vector of detected corners.                                                          point2f
@param maxCorners Maximum number of corners to return. If there are more corners than are found,           
the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set
and all detected corners are returned.
@param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The           品质的体现指标是特征值
parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
(see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
quality measure less than the product are rejected. For example, if the best corner has the
quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
less than 15 are rejected.
@param minDistance Minimum possible Euclidean distance between the returned corners.                      角点间最小的欧氏距离限制
@param mask Region of interest. If the image is not empty (it needs to have the type
CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
@param cornersQuality Output vector of quality measure of the detected corners.                           //! 这个挺有用的！
@param blockSize Size of an average block for computing a derivative covariation matrix over each
pixel neighborhood. See cornerEigenValsAndVecs .
@param gradientSize Aperture parameter for the Sobel operator used for derivatives computation.           ！用于导数计算的Sobel算子的孔径参数。
See cornerEigenValsAndVecs .
@param useHarrisDetector Parameter indicating whether to use a Harris detector (see #cornerHarris)
or #cornerMinEigenVal.
@param k Free parameter of the Harris detector.                                                           Harris探测器的自由参数。
 */


//https://blog.csdn.net/xdfyoga1/article/details/44175637
void goodFeaturesToTrackxx(cv::Mat image, cv::Mat _corners,
    int maxCorners, double qualityLevel, double minDistance,
    cv::Mat mask, int blockSize,
    bool useHarrisDetector, double harrisK)
{
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);  //对参数有一些基本要求
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

    Mat eig, tmp;   //eig存储每个像素协方差矩阵的最小特征值，tmp用来保存经膨胀后的eig
    if (useHarrisDetector)
        cornerHarris(image, eig, blockSize, 3, harrisK); 
        //blockSize是计算2*2协方差矩阵的窗口大小，sobel算子窗口为3，harrisK是计算Harris角点时需要的值
    else
        cornerMinEigenVal(image, eig, blockSize, 3);  
        //计算每个像素对应的协方差矩阵的最小特征值，保存在eig中

    double maxVal = 0;
    minMaxLoc(eig, 0, &maxVal, 0, 0, mask);                        //maxVal保存了eig的最大值
    threshold(eig, eig, maxVal * qualityLevel, 0, THRESH_TOZERO);  //阈值设置为maxVal乘以qualityLevel，大于此阈值的保持不变，小于此阈值的都设为0

    //! 默认用3*3的核膨胀，膨胀之后，除了局部最大值点和原来相同，其它非局部最大值点被  
    //! 3*3邻域内的最大值点取代，如不理解，可看一下灰度图像的膨胀原理  
    dilate(eig, tmp, Mat());  //tmp中保存了膨胀之后的eig

    Size imgsize = image.size();

    vector<const float*> tmpCorners;  //存放粗选出的角点地址

    // collect list of pointers to features - put them into temporary image 
    for (int y = 1; y < imgsize.height - 1; y++)
    {
        const float* eig_data = (const float*)eig.ptr(y);  //获得eig第y行的首地址
        const float* tmp_data = (const float*)tmp.ptr(y);  //获得tmp第y行的首地址
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for (int x = 1; x < imgsize.width - 1; x++)
        {
            float val = eig_data[x];
            if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))  
                //! val == tmp_data[x]说明这是局部极大值 //? 并不清楚为什么，但记住吧
                tmpCorners.push_back(eig_data + x);  //保存其位置
        }
    }

    //-----------此分割线以上是根据特征值粗选出的角点，我们称之为弱角点----------//
    //-----------此分割线以下还要根据minDistance进一步筛选角点，仍然能存活下来的我们称之为强角点----------//

    //cv::sort(tmpCorners, greaterThanPtr<float>());  // 按特征值降序排列，注意这一步很重要，后面的很多编程思路都是建立在这个降序排列的基础上
    vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    //下面的程序有点稍微难理解，需要自己仔细想想
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);   //向最近的整数取整

        //这里根据cell_size构建了一个矩形窗口grid(虽然下面的grid定义的是vector<vector>，而并不是我们这里说的矩形窗口，但为了便于理解,还是将grid想象成一个grid_width * grid_height的矩形窗口比较好)，除以cell_size说明grid窗口里相差一个像素相当于_image里相差minDistance个像素，至于为什么加上cell_size - 1后面会讲
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width * grid_height);  //vector里面是vector，grid用来保存获得的强角点坐标

        minDistance *= minDistance;  //平方，方面后面计算，省的开根号

        for (i = 0; i < total; i++)     // 刚刚粗选的弱角点，都要到这里来接收新一轮的考验
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);  //tmpCorners中保存了角点的地址，eig.data返回eig内存块的首地址
            int y = (int)(ofs / eig.step);   //角点在原图像中的行
            int x = (int)((ofs - y * eig.step) / sizeof(float));  //在原图像中的列

            bool good = true;  //先认为当前角点能接收考验，即能被保留下来

            int x_cell = x / cell_size;  //x_cell，y_cell是角点（y,x）在grid中的对应坐标
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;  // (y_cell，x_cell）的4邻域像素
            int y1 = y_cell - 1;  //现在知道为什么前面grid_width定义时要加上cell_size - 1了吧，这是为了使得（y,x）在grid中的4邻域像素都存在，也就是说(y_cell，x_cell）不会成为边界像素
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check，再次确认x1,y1,x2或y2不会超出grid边界
            x1 = std::max(0, x1);  //比较0和x1的大小
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            //记住grid中相差一个像素，相当于_image中相差了minDistance个像素
            for (int yy = y1; yy <= y2; yy++)  // 行
            {
                for (int xx = x1; xx <= x2; xx++)  //列
                {
                    vector <Point2f>& m = grid[yy * grid_width + xx];  //引用

                    if (m.size())  //如果(y_cell，x_cell)的4邻域像素，也就是(y,x)的minDistance邻域像素中已有被保留的强角点
                    {
                        for (j = 0; j < m.size(); j++)   //当前角点周围的强角点都拉出来跟当前角点比一比
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;
                            //注意如果(y,x)的minDistance邻域像素中已有被保留的强角点，则说明该强角点是在(y,x)之前就被测试过的，又因为tmpCorners中已按照特征值降序排列（特征值越大说明角点越好），这说明先测试的一定是更好的角点，也就是已保存的强角点一定好于当前角点，所以这里只要比较距离，如果距离满足条件，可以立马扔掉当前测试的角点
                            if (dx * dx + dy * dy < minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }   // 列
            }    //行

        break_out:

            if (good)
            {
                // printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
                //    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
                grid[y_cell * grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                ++ncorners;

                if (maxCorners > 0 && (int)ncorners == maxCorners)  //由于前面已按降序排列，当ncorners超过maxCorners的时候跳出循环直接忽略tmpCorners中剩下的角点，反正剩下的角点越来越弱
                    break;
            }
        }
    }
    else    //除了像素本身，没有哪个邻域像素能与当前像素满足minDistance < 1,因此直接保存粗选的角点
    {
        for (i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
            int y = (int)(ofs / eig.step);   //粗选的角点在原图像中的行
            int x = (int)((ofs - y * eig.step) / sizeof(float));  //在图像中的列

            corners.push_back(Point2f((float)x, (float)y));
            ++ncorners;
            if (maxCorners > 0 && (int)ncorners == maxCorners)
                break;
        }
    }

    //Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);

    /*
    for( i = 0; i < total; i++ )
    {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
        int y = (int)(ofs / eig.step);
        int x = (int)((ofs - y*eig.step)/sizeof(float));
        if( minDistance > 0 )
        {
            for( j = 0; j < ncorners; j++ )
            {
                float dx = x - corners[j].x;
                float dy = y - corners[j].y;
                if( dx*dx + dy*dy < minDistance )
                    break;
            }
            if( j < ncorners )
                continue;
        }
        corners.push_back(Point2f((float)x, (float)y));
        ++ncorners;
        if( maxCorners > 0 && (int)ncorners == maxCorners )
            break;
    }
*/
}
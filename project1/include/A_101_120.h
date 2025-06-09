#pragma once
#include <opencv2/opencv.hpp>
#include "logger.h"
#include <algorithm>
#include "A_101_120_add.h"
#include <windows.h>  // 获取memory
#include <psapi.h>
#include <stdio.h>


#ifndef _A_101_120_
#define _A_101_120_

#define PI  atan(1) * 4

using namespace cv;

void A101();
void A102();
void A103();
void A104();
void A105();

namespace NA106 {
	void A106();
}

namespace NA107 {
	void A107();
}

namespace NA108 {
	void A108_solver();
}


namespace NA109 {
	void A109_solver();
}


namespace NA110 {
	void A110_solver();
}

namespace NA111 {
	int picshadowx(cv::Mat binary, cv::Mat* show);
	void doing(cv::Mat imgori);
	void A111_solver();
}


namespace NA112 {
	void A112_solver();
}

namespace NA113 {
	void A113_solver();
}


namespace NA114 {

	//IncrementalMean local_meaner;


	void A114_solver();

    struct SingleMat
    {
        int camPos = -1;
        std::chrono::steady_clock::time_point starttime;
        int w, h, format;
        char* imgori = nullptr;
        cv::Mat imgrst;
        size_t index;
        int groupsize;
        int _iresult;
        bool _bshow = true;
        std::string sn_fromscanner;
        std::string cam_serial;
        std::vector<std::string> error_message;

        // 默认构造函数
        SingleMat() = default;   // 相当于 SingleMat() {}；

        // 拷贝构造函数（深度拷贝）
        SingleMat(const SingleMat& other)
            : camPos(other.camPos),
            starttime(other.starttime),
            w(other.w),
            h(other.h),
            format(other.format),
            imgrst(other.imgrst.clone()), // 深度拷贝 cv::Mat
            index(other.index),
            groupsize(other.groupsize),
            _iresult(other._iresult),
            _bshow(other._bshow),
            sn_fromscanner(other.sn_fromscanner),
            cam_serial(other.cam_serial),
            error_message(other.error_message)
        {
            // 其他成员函数通过调用其所属数据类型的拷贝构造函数，实现深度拷贝；
            // 深度拷贝 imgori
            if (other.imgori != nullptr && other.w > 0 && other.h > 0)
            {
                size_t size = other.w * other.h * (other.format == 1 ? 1 : 3);
                imgori = new char[size];
                std::memcpy(imgori, other.imgori, size);
            }
            else
            {
                imgori = nullptr;
            }
        }

        // 拷贝赋值运算符（深度拷贝）
        SingleMat& operator=(const SingleMat& other)
        {
            if (this != &other)
            {
                // 释放原有资源
                delete[] imgori;

                // 拷贝基本类型成员
                camPos = other.camPos;
                starttime = other.starttime;
                w = other.w;
                h = other.h;
                format = other.format;
                index = other.index;
                groupsize = other.groupsize;
                _iresult = other._iresult;
                _bshow = other._bshow;

                // 拷贝字符串和容器
                sn_fromscanner = other.sn_fromscanner;
                cam_serial = other.cam_serial;
                error_message = other.error_message;

                // 深度拷贝 cv::Mat
                imgrst = other.imgrst.clone();

                // 深度拷贝 imgori
                if (other.imgori != nullptr && other.w > 0 && other.h > 0)
                {
                    size_t size = other.w * other.h * (other.format == 1 ? 1 : 3);
                    imgori = new char[size];
                    std::memcpy(imgori, other.imgori, size);
                }
                else
                {
                    imgori = nullptr;
                }
            }
            return *this;
        }

        // 移动构造函数
        SingleMat(SingleMat&& other) noexcept  // 移动构造函数，通过窃取临时对象（右值）的资源来构造新对象的构造函数，同时将原对象重置。
            : camPos(other.camPos),
            starttime(other.starttime),
            w(other.w),
            h(other.h),
            format(other.format),
            imgori(other.imgori),
            imgrst(std::move(other.imgrst)),
            index(other.index),
            groupsize(other.groupsize),
            _iresult(other._iresult),
            _bshow(other._bshow),
            sn_fromscanner(std::move(other.sn_fromscanner)),
            cam_serial(std::move(other.cam_serial)),
            error_message(std::move(other.error_message))
        {
            other.imgori = nullptr;
        }

        // 移动赋值运算符
        SingleMat& operator=(SingleMat&& other) noexcept
        {
            if (this != &other)
            {
                delete[] imgori;

                camPos = other.camPos;
                starttime = other.starttime;
                w = other.w;
                h = other.h;
                format = other.format;
                imgori = other.imgori;
                imgrst = std::move(other.imgrst);
                index = other.index;
                groupsize = other.groupsize;
                _iresult = other._iresult;
                _bshow = other._bshow;
                sn_fromscanner = std::move(other.sn_fromscanner);
                cam_serial = std::move(other.cam_serial);
                error_message = std::move(other.error_message);

                other.imgori = nullptr;
            }
            return *this;
        }

        ~SingleMat()
        {
            delete[] imgori;
        }
    };
}


namespace NA115 {

    void A115_solver();
}

#endif 

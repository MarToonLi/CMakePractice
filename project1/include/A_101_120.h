#pragma once
#include <opencv2/opencv.hpp>
#include "logger.h"
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

#endif 

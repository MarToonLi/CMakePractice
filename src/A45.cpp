#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "A_41_50.h"
#include <math.h>
#include <time.h>

using namespace cv;

// ����任
const int ANGLE_T = 180;//�Ƕ�
const int RHO_MAX = 127;

// �����
struct HoughTable {
	int table[RHO_MAX * 2][ANGLE_T];
};

//����任��ͶƱ
HoughTable HoughVote(HoughTable hough_table, cv::Mat img)
{
	int height = img.rows;
	int width = img.cols;
	int rho = 0;
	double angle = 0;
	//��������ͼƬ
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf_s("y:%d,x:%d\n", y, x);
			// ����������ز��Ǳߣ���ֱ������
			if (img.at<uchar>(y, x) != 255) 
				continue;	
			// 0 <= �Ƕ� < 180
			for (int t = 0; t < ANGLE_T; t++) {
				angle = PI / 180 * t;
				//r = xcos�� + ysin��
				rho = (int)(x * cos(angle) + y * sin(angle));
				//����ͶƱ
				hough_table.table[rho + RHO_MAX][t] ++;
			}
		}
	}
	return hough_table;
}

//����任���Ǽ���ֵ����
HoughTable HoughNMS(HoughTable hough_table)
{
    // �����
    HoughTable output_hough_table;

    // ��ʼ�������
    for (int rho = 0; rho < RHO_MAX * 2; rho++) {
        for (int t = 0; t < ANGLE_T; t++)
        {
            output_hough_table.table[rho][t] = 0;
        }
    }

    // top N x, y
    const int N = 30;
    int top_N_rho[N], top_N_t[N], top_N_vote[N];
    int tmp_rho, tmp_t, tmp_vote, tmp_rho2, tmp_t2, tmp_vote2;
    int rho, t;

    for (int n = 0; n < N; n++)
    {
        top_N_rho[n] = -1;
        top_N_t[n] = -1;
        top_N_vote[n] = -1;
    }

    for (int rho = 0; rho < RHO_MAX * 2; rho++) {
        for (int t = 0; t < ANGLE_T; t++) {
            if (hough_table.table[rho][t] == 0) 
                continue;
            
            // �Ա�����
            if (((t - 1) >= 0) && ((rho - 1) >= 0)) 
                if (hough_table.table[rho][t] < hough_table.table[rho - 1][t - 1]) 
                    continue;
             
            // �Ա�����
            if ((rho - 1) >= 0) {
                if (hough_table.table[rho][t] < hough_table.table[rho - 1][t]) {
                    continue;
                }
            }

            // �Ա�����
            if (((t + 1) < ANGLE_T) && ((rho - 1) >= 0)) 
                if (hough_table.table[rho][t] < hough_table.table[rho - 1][t + 1]) 
                    continue;
                
            //�Ա����
            if ((t - 1) >= 0) 
                if (hough_table.table[rho][t] < hough_table.table[rho][t - 1]) 
                    continue;

            // �Ա��ұ�
            if ((t + 1) < ANGLE_T) 
                if (hough_table.table[rho][t] < hough_table.table[rho][t + 1]) 
                    continue;
                
            // �Ա�����
            if (((t - 1) >= 0) && ((rho + 1) < RHO_MAX * 2)) 
                if (hough_table.table[rho][t] < hough_table.table[rho + 1][t - 1]) 
                    continue;

            // �Ա�����
            if ((rho + 1) < RHO_MAX * 2) 
                if (hough_table.table[rho][t] < hough_table.table[rho + 1][t]) 
                    continue;

            // �Ա�����
            if (((t + 1) < ANGLE_T) && ((rho + 1) < RHO_MAX * 2)) 
                if (hough_table.table[rho][t] < hough_table.table[rho + 1][t + 1]) 
                    continue;

            // ѡ��ǰNƱ
            for (int n = 0; n < N; n++) {
                if (top_N_vote[n] <= hough_table.table[rho][t]) {
                    tmp_vote = top_N_vote[n];
                    tmp_rho = top_N_rho[n];
                    tmp_t = top_N_t[n];
                    top_N_vote[n] = hough_table.table[rho][t];
                    top_N_rho[n] = rho;
                    top_N_t[n] = t;

                    for (int m = n + 1; m < N - 1; m++) {
                        tmp_vote2 = top_N_vote[m];
                        tmp_rho2 = top_N_rho[m];
                        tmp_t2 = top_N_t[m];
                        top_N_vote[m] = tmp_vote;
                        top_N_rho[m] = tmp_rho;
                        top_N_t[m] = tmp_t;
                        tmp_vote = tmp_vote2;
                        tmp_rho = tmp_rho2;
                        tmp_t = tmp_t2;
                    }

                    top_N_vote[N - 1] = tmp_vote;
                    top_N_rho[N - 1] = tmp_rho;
                    top_N_t[N - 1] = tmp_t;
                    break;
                }
            }
        }
    }

    // ���ǰN��ͶƱ������
    for (int n = 0; n < N; n++) {
        if (top_N_rho[n] == -1) 
            break;
        rho = top_N_rho[n];
        t = top_N_t[n];
        output_hough_table.table[rho][t] = hough_table.table[rho][t];
    }

    return output_hough_table;
}
void A45(Mat img)
{
	//Canny
	cv::Mat gray = ImgGray(img);
	cv::Mat gaussian = GaussianFilter(gray, 1.4);
	cv::Mat fy = SobelFilterV(gaussian);
	cv::Mat fx = SobelFilterH(gaussian);
	cv::Mat edge = GetEdge(fx, fy);
	cv::Mat angle = GetAngel(fx, fy);
	edge = NMS(angle, edge);
	edge = Histerisis(edge, 100, 30);
	imshow("canny", edge);

	HoughTable hough_table;
    //ͶƱ
	hough_table = HoughVote(hough_table, edge);
	//�Ǽ���ֵ����
    hough_table = HoughNMS(hough_table);

	waitKey(0);
	destroyAllWindows();
}
#ifndef CORE_COMMON
#define CORE_COMMON
#include <opencv2/opencv.hpp>
#include <string>
#include <windows.h>
#include "slogger.h"
namespace wikky_algo
{
#define DLLINTERFACE "1.3"		//change _iresult from bool to index
	struct SingleMat
	{
		int camPos = -1;
		std::chrono::steady_clock::time_point starttime;
		cv::Mat imgori;
		cv::Mat imgrst;
		size_t index;
		int groupsize;
		int _iresult;
		bool _bshow = true;
		std::string sn_fromscanner;
		std::string cam_serial;
		std::vector<std::string> error_message;
	};
	struct CheckParam
	{
		std::string cam_serial;

		int _iThreadX = 145;
		int _iThreadY = 320;
		int _iThreadZ = 2133;
	};



	struct SlotParam
	{
		//阈值分割后的槽区域设为 thresh_slot_area；
		//实际的槽区域设为 real_slot_area；

		// 产品固有性质
		std::string name;
		int _design_height;    // 槽位设计高度(mm)

		// 算法超参数
		int _min_area;         // 默认值1000 槽位区域过滤条件之一：最少的面积大小
		float _max_k;            // 无用
		int slot_blue_thresh;  // 分割slot和blu的阈值
		                       //!? 应该尽量大一些，标准是稍微超过细白线的亮度
		int _height;           // slot_blue_thresh下的标准高度pixel

		int _top_reduce_val;   // thresh_slot_area的上边界距离real_slot_area的上边界的像素距离值
		int _bottom_reduce_val;// thresh_slot_area的下边界距离real_slot_area的下边界的像素距离值


		// 构造函数：在创建对象时初始化成员
		SlotParam(std::string n, 
			int dh,
			int ma, 
			float mk,
			int sbt,
			int h, 
			int trv, 
			int brv
		) {
			name = n;
			_design_height = dh;
			_min_area = ma;
			_max_k = mk;
			slot_blue_thresh = sbt;
			_height = h;
			_top_reduce_val = trv; 
			_bottom_reduce_val = brv;
		}

		SlotParam(
		) {
			name = "default";
			_design_height = 3;
			_min_area = 1000;
			_max_k = 1;
			slot_blue_thresh = 150;
			_height = 510;
			_top_reduce_val = 150;
			_bottom_reduce_val = 110;
		}
	};


	struct WhiteLineParam
	{
		//阈值分割后的槽区域设为 thresh_slot_area；
		//实际的槽区域设为 real_slot_area；

		// 产品固有性质
		std::string name;             // slot名字
		int _width_tol;
		int _height_tol;              // slot区域高度变动范围    10
		int _white_slot_thesh;        // 分割slot和whiteline的阈值  如果太高，会造成细白线的有效像素数量的减少！太少会引入吸盘和槽孔的像素
		int _white_line_thickness;    // 白线的像素高度 30
		int _white_line_length;
		int _white_null_thesh;        // 辨识无料的像素阈值  1500


		// 构造函数：在创建对象时初始化成员
		WhiteLineParam(std::string n,
			int wt,
			int ht,
			int wst,
			int wlt,
			int wll,
			int wnt
		) {
			name = n;
			_width_tol = wt;
			_height_tol = ht;
			_white_slot_thesh = wst;
			_white_line_thickness = wlt;
			_white_line_length = wll;
			_white_null_thesh = wnt;
		};

		WhiteLineParam(
		) {
			name = "CUN11";
			_width_tol = 0;
			_height_tol = 10;
			_white_slot_thesh = 60;
			_white_line_thickness = 30;
			_white_line_length = 4630;
			_white_null_thesh = 1500;
		}
	};





	struct rowInfo
	{
		cv::Point p1 = cv::Point(0, 0);
		cv::Point p2 = cv::Point(0, 0);
		cv::Point p3 = cv::Point(0, 0);

		rowInfo() : p1(cv::Point(0, 0)), p2(cv::Point(0, 0)), p3(cv::Point(0, 0)) { }
	};

	struct ObliqueLine {
		cv::Point p1;
		cv::Point p2;
		double delta_a;
		double delta_b;
		double angle;
	};

	struct ReturnStatus {
		int index;
		std::string en_name;
		std::string cn_name;
		//std::wstring vn_name;

		ReturnStatus(int id, 
			std::string en, 
			std::string cn) {
			index = id; en_name = en; cn_name = cn;
		}
	};


	struct ReturnDatas {
		bool isNull;
		bool isAlgoError;
		float minmax_val;
		float k_val;
		float offset_val;
		int boundary_point_nums;
		std::vector<cv::Point> white_contour;

		ReturnDatas() {};

		ReturnDatas(
			bool isn,
			bool isae,
			float mmv,
			float kv,
			float ov,
			int bpn
		) {
			isNull = isn;
			isAlgoError = isae;
			minmax_val = mmv;
			k_val = kv;
			offset_val = ov;
			boundary_point_nums = bpn;
		};

		ReturnDatas(
			bool isn,
			bool isae,
			float mmv,
			float kv,
			float ov,
			int bpn,
			std::vector<cv::Point> wc
		) {
			isNull = isn;
			isAlgoError = isae;
			minmax_val = mmv;
			k_val = kv;
			offset_val = ov;
			boundary_point_nums = bpn;
			white_contour = wc;
		}
	};



	using TestCallback = std::function<int(SingleMat&, CheckParam*)>;
	using UpdateParam = std::function<void(CheckParam&)>;

	


	///////// add

	class ColorManager {
	public:
		std::map<std::string, cv::Scalar> colorMap;

		void initColorMap();
	};

	std::string d2str(int val, int precision = 3);

	std::string d2str(float val, int precision = 3);

	std::string d2str(double val, int precision = 3);

	std::string vectors2string(std::vector<std::string> vectors);

	std::string vectors2string(std::vector<char *> vectors);

	std::string vectors2string(std::vector<float> vectors);

	std::string vectors2string(std::vector<int64_t> vectors);

	std::string vectors2string(cv::Rect rect);

	bool getHostname(std::string& hostname_str);

	void GetStringSize(HDC hDC, const char* str, int* w, int* h);

	void putTextZH(cv::Mat& dst, const char* str, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline);

	void putTextWStr(cv::Mat& dst, const wchar_t* wstr, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline);


	bool outputParam(SlotParam& param);

	bool outputParam(WhiteLineParam& param);

	bool outputRAMandGPU();

	double calculateDistance(cv::Point p1, cv::Point p2);

	cv::Point getContourCentroid(const std::vector<cv::Point>& contour);

	bool hasIntersection(int a1, int b1, int a2, int b2);

	float minDistanceBetweenContours(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2);

	bool minDistanceBetweenContours(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2, float& minDistX, float& minDistY);



}

#endif // !CORE_COMMON
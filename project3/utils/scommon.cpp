#include <opencv2/opencv.hpp>
#include "scommon.h"

namespace wikky_algo
{

	std::string d2str(int val, int precision) {
		std::ostringstream stream;
		stream << std::fixed << std::setprecision(precision) << val;
		std::string result = stream.str();
		return result;
	}

	std::string d2str(float val, int precision) {
		std::ostringstream stream;
		stream << std::fixed << std::setprecision(precision) << val;
		std::string result = stream.str();
		return result;
	}

	std::string d2str(double val, int precision) {
		std::ostringstream stream;
		stream << std::fixed << std::setprecision(precision) << val;
		std::string result = stream.str();
		return result;
	}


	std::string vectors2string(std::vector<std::string> vectors) {
		if (vectors.size() == 0) { return ""; }
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << vectors[i];
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}

	std::string vectors2string(std::vector<char*> vectors) {
		if (vectors.size() == 0) { return ""; }
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << (vectors[i] ? vectors[i] : "(null)");
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}


	std::string vectors2string(std::vector<float> vectors) {
		if (vectors.size() == 0) { return ""; }
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << (vectors[i] ? vectors[i] : -1);
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}


	std::string vectors2string(std::vector<int64_t> vectors) {
		if (vectors.size() == 0) { return ""; }
		std::ostringstream oss;
		for (size_t i = 0; i < vectors.size(); ++i) {
			oss << (vectors[i] ? vectors[i] : -1);
			if (i != vectors.size() - 1) { oss << "-"; }  // 如果不是最后一个数字，则添加横杠作为分隔符
		}
		std::string vstring = oss.str();

		return vstring;
	}


	std::string vectors2string(cv::Rect rect) {
		std::ostringstream oss;
		oss << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;
		std::string vstring = oss.str();
		return vstring;
	}

	bool getHostname(std::string& hostname_str)
	{
		char hostname_chars[MAX_COMPUTERNAME_LENGTH + 1];
		DWORD size = sizeof(hostname_chars) / sizeof(hostname_chars[0]);
		if (GetComputerNameA(hostname_chars, &size)) {
			hostname_str = hostname_chars;
			return true;
		}
		else { return false; }
	}


    void ColorManager::initColorMap() {
        this->colorMap["black"] = cv::Scalar(0, 0, 0);
        this->colorMap["red"] = cv::Scalar(0, 0, 255);         // ng
        this->colorMap["blue"] = cv::Scalar(255, 0, 0);        
        this->colorMap["green"] = cv::Scalar(0, 255, 0);       // ok
        this->colorMap["yellow"] = cv::Scalar(0, 255, 255);    
        this->colorMap["white"] = cv::Scalar(255, 255, 255);
        this->colorMap["purple"] = cv::Scalar(210, 130, 150);
        this->colorMap["gray"] = cv::Scalar(128, 128, 128);    // mask
        this->colorMap["null"] = cv::Scalar(13, 206, 255);
        this->colorMap["deepYellow"] = cv::Scalar(10, 192, 231);
    }


	void GetStringSize(HDC hDC, const char* str, int* w, int* h)
	{
		SIZE size;
		GetTextExtentPoint32A(hDC, str, (int)strlen(str), &size);
		if (w != 0) *w = size.cx;
		if (h != 0) *h = size.cy;
	}

	void putTextZH(cv::Mat& dst, const char* str, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline)
	{

		CV_Assert(dst.data != 0 && (dst.channels() == 1 || dst.channels() == 3));

		int x, y, r, b;
		if (org.x > dst.cols || org.y > dst.rows) return;
		x = org.x < 0 ? -org.x : 0;
		y = org.y < 0 ? -org.y : 0;

		LOGFONTA lf;
		lf.lfHeight = -fontSize;
		lf.lfWidth = 0;
		lf.lfEscapement = 0;
		lf.lfOrientation = 0;
		lf.lfWeight = 5;
		lf.lfItalic = italic;       //斜体
		lf.lfUnderline = underline; //下划线
		lf.lfStrikeOut = 0;
		lf.lfCharSet = DEFAULT_CHARSET;
		lf.lfOutPrecision = 0;
		lf.lfClipPrecision = 0;
		lf.lfQuality = PROOF_QUALITY;
		lf.lfPitchAndFamily = 0;
		strcpy_s(lf.lfFaceName, fn);

		HFONT hf = CreateFontIndirectA(&lf);
		HDC hDC = CreateCompatibleDC(0);
		HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

		int strBaseW = 0, strBaseH = 0;
		int singleRow = 0;
		char buf[1 << 12];
		strcpy_s(buf, str);
		char* bufT[1 << 12];  // 这个用于分隔字符串后剩余的字符，可能会超出。
		//处理多行
		{
			int nnh = 0;
			int cw, ch;

			const char* ln = strtok_s(buf, "\n", bufT);
			while (ln != 0)
			{
				GetStringSize(hDC, ln, &cw, &ch);
				strBaseW = max(strBaseW, cw);
				strBaseH = max(strBaseH, ch);

				ln = strtok_s(0, "\n", bufT);
				nnh++;
			}
			singleRow = strBaseH;
			strBaseH *= nnh;
		}

		if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
		{
			SelectObject(hDC, hOldFont);
			DeleteObject(hf);
			DeleteObject(hDC);
			return;
		}

		r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
		b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
		org.x = org.x < 0 ? 0 : org.x;
		org.y = org.y < 0 ? 0 : org.y;

		BITMAPINFO bmp = { 0 };
		BITMAPINFOHEADER& bih = bmp.bmiHeader;
		int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

		bih.biSize = sizeof(BITMAPINFOHEADER);
		bih.biWidth = strBaseW;
		bih.biHeight = strBaseH;
		bih.biPlanes = 1;
		bih.biBitCount = 24;
		bih.biCompression = BI_RGB;
		bih.biSizeImage = strBaseH * strDrawLineStep;
		bih.biClrUsed = 0;
		bih.biClrImportant = 0;

		void* pDibData = 0;
		HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);

		CV_Assert(pDibData != 0);
		HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

		//color.val[2], color.val[1], color.val[0]
		SetTextColor(hDC, RGB(255, 255, 255));
		SetBkColor(hDC, 0);
		//SetStretchBltMode(hDC, COLORONCOLOR);

		strcpy_s(buf, str);
		const char* ln = strtok_s(buf, "\n", bufT);
		int outTextY = 0;
		while (ln != 0)
		{
			TextOutA(hDC, 0, outTextY, ln, (int)strlen(ln));
			outTextY += singleRow;
			ln = strtok_s(0, "\n", bufT);
		}
		uchar* dstData = (uchar*)dst.data;
		int dstStep = (int)(dst.step / sizeof(dstData[0]));
		unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
		unsigned char* pStr = (unsigned char*)pDibData + x * 3;
		for (int tty = y; tty <= b; ++tty)
		{
			unsigned char* subImg = pImg + (tty - y) * dstStep;
			unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
			for (int ttx = x; ttx <= r; ++ttx)
			{
				for (int n = 0; n < dst.channels(); ++n) {
					double vtxt = subStr[n] / 255.0;
					int cvv = (int)round(vtxt * color.val[n] + (1 - vtxt) * subImg[n]);
					subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
				}

				subStr += 3;
				subImg += dst.channels();
			}
		}

		SelectObject(hDC, hOldBmp);
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hBmp);
		DeleteDC(hDC);
	}



	//图片写wchar_t宽字符：中英日问越南语等等
	void putTextWStr(cv::Mat& dst, const wchar_t* wstr, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline)
	{
		CV_Assert(dst.data != 0 && (dst.channels() == 1 || dst.channels() == 3));

		int x, y, r, b;
		if (org.x > dst.cols || org.y > dst.rows) return;
		x = org.x < 0 ? -org.x : 0;
		y = org.y < 0 ? -org.y : 0;

		LOGFONTA lf;
		lf.lfHeight = -fontSize;
		lf.lfWidth = 0;
		lf.lfEscapement = 0;
		lf.lfOrientation = 0;
		lf.lfWeight = 5;
		lf.lfItalic = italic;   //斜体
		lf.lfUnderline = underline; //下划线
		lf.lfStrikeOut = 0;
		lf.lfCharSet = DEFAULT_CHARSET;
		lf.lfOutPrecision = 0;
		lf.lfClipPrecision = 0;
		lf.lfQuality = PROOF_QUALITY;
		lf.lfPitchAndFamily = 0;
		strcpy_s(lf.lfFaceName, fn);

		HFONT hf = CreateFontIndirectA(&lf);
		HDC hDC = CreateCompatibleDC(0);
		HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

		int strBaseW = 0, strBaseH = 0;
		int singleRow = 0;
		int len = 1 << 12;
		wchar_t buf[1 << 12];
		wcscpy_s(buf, wstr);
		wchar_t* bufT[1 << 12];  // 这个用于分隔字符串后剩余的字符，可能会超出。
		//处理多行
		{
			int nnh = 0;
			int cw, ch;

			const wchar_t* ln = wcstok_s(buf, L"\n", bufT);
			while (ln != 0)
			{
				SIZE size;
				GetTextExtentPoint32W(hDC, ln, (int)wcslen(ln), &size);
				if (&cw != 0) cw = size.cx;
				if (&ch != 0) ch = size.cy;

				strBaseW = max(strBaseW, cw);
				strBaseH = max(strBaseH, ch);

				ln = wcstok_s(0, L"\n", bufT);
				nnh++;
			}
			singleRow = strBaseH;
			strBaseH *= nnh;
		}

		if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
		{
			SelectObject(hDC, hOldFont);
			DeleteObject(hf);
			DeleteObject(hDC);
			return;
		}

		r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
		b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
		org.x = org.x < 0 ? 0 : org.x;
		org.y = org.y < 0 ? 0 : org.y;

		BITMAPINFO bmp = { 0 };
		BITMAPINFOHEADER& bih = bmp.bmiHeader;
		int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

		bih.biSize = sizeof(BITMAPINFOHEADER);
		bih.biWidth = strBaseW;
		bih.biHeight = strBaseH;
		bih.biPlanes = 1;
		bih.biBitCount = 24;
		bih.biCompression = BI_RGB;
		bih.biSizeImage = strBaseH * strDrawLineStep;
		bih.biClrUsed = 0;
		bih.biClrImportant = 0;

		void* pDibData = 0;
		HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);

		CV_Assert(pDibData != 0);
		HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

		SetTextColor(hDC, RGB(255, 255, 255));
		SetBkColor(hDC, 0);

		wcscpy_s(buf, wstr);
		const wchar_t* ln = wcstok_s(buf, L"\n", bufT);
		int outTextY = 0;
		while (ln != 0)
		{
			TextOutW(hDC, 0, outTextY, ln, (int)wcslen(ln));
			outTextY += singleRow;
			ln = wcstok_s(0, L"\n", bufT);
		}
		uchar* dstData = (uchar*)dst.data;
		int dstStep = (int)(dst.step / sizeof(dstData[0]));
		unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
		unsigned char* pStr = (unsigned char*)pDibData + x * 3;
		for (int tty = y; tty <= b; ++tty)
		{
			unsigned char* subImg = pImg + (tty - y) * dstStep;
			unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
			for (int ttx = x; ttx <= r; ++ttx)
			{
				for (int n = 0; n < dst.channels(); ++n)
				{
					int curSubstr = subStr[n];
					int curSubimg = subImg[n];
					double vtxt = subStr[n] / 255.0;
					int cvv = (int)round(vtxt * color.val[n] + (1 - vtxt) * subImg[n]);
					subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
				}

				subStr += 3;
				subImg += dst.channels();
			}
		}

		SelectObject(hDC, hOldBmp);
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hBmp);
		DeleteDC(hDC);
	}


	bool outputParam(SlotParam& param)
	{
		LOGT("name              : {};", param.name);
		LOGT("slot_blue_thresh  : {};", param.slot_blue_thresh);
		LOGT("_bottom_reduce_val: {};", param._bottom_reduce_val);
		LOGT("_design_height    : {};", param._design_height);
		LOGT("_height           : {};", param._height);
		LOGT("_max_k            : {};", param._max_k);
		LOGT("_min_area         : {};", param._min_area);
		LOGT("_top_reduce_val   : {};", param._top_reduce_val);
		LOGT("\n");
		return true;
	}

	bool outputParam(WhiteLineParam& param)
	{
		LOGT("name                 : {};", param.name);
		LOGT("_width_tol           : {};", param._width_tol);
		LOGT("_height_tol          : {};", param._height_tol);
		LOGT("_white_slot_thesh    : {};", param._white_slot_thesh);
		LOGT("_white_line_thickness: {};", param._white_line_thickness);
		LOGT("_white_line_length   : {};", param._white_line_length);
		LOGT("_white_null_thesh    : {};", param._white_null_thesh);
		LOGT("\n");
		return true;
	}

	bool outputRAMandGPU()
	{
		MEMORYSTATUSEX statex;
		statex.dwLength = sizeof(statex);
		if (GlobalMemoryStatusEx(&statex)) {
			double total = statex.ullTotalPhys / 1024 / 1024;
			double inuse = (statex.ullTotalPhys - statex.ullAvailPhys) / 1024 / 1024;
			LOGW("Memory in use / Total physical memory: {} / {};", inuse, total);
		}
		else {
			LOGW("Failed to get memory status");
		}

		return false;
	}

	// 计算两点之间的欧氏距离
	double calculateDistance(cv::Point p1, cv::Point p2) {
		return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
	}

	// 计算轮廓的质心
	cv::Point getContourCentroid(const std::vector<cv::Point>& contour) {
		cv::Moments m = cv::moments(contour);
		return cv::Point(m.m10 / m.m00, m.m01 / m.m00);  // 计算质心坐标
	}

	bool hasIntersection(int a1, int b1, int a2, int b2) {
		// 判断两个区间是否有交集
		return (std::max)(a1, a2) <= (std::min)(b1, b2);
	}


	float minDistanceBetweenContours(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
		float minDist = FLT_MAX;

		// 遍历第一个轮廓的每个点
		for (const auto& p1 : contour1) {
			// 对每个点，计算它到第二个轮廓的最近点距离
			for (const auto& p2 : contour2) {
				float dist = cv::norm(p1 - p2);  // 计算两点之间的距离
				minDist = (std::min)(minDist, dist);  // 更新最小距离
			}
		}

		return minDist;
	}

	bool minDistanceBetweenContours(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2, float& minDistX, float& minDistY) {
		minDistX = FLT_MAX;
		minDistY = FLT_MAX;

		// 遍历第一个轮廓的每个点
		for (int i = 0; i < contour1.size(); i = i + 5) {
			cv::Point p1 = contour1[i];

			// 对每个点，计算它到第二个轮廓的最近点距离
			for (int j = 0; j < contour2.size(); j = j + 5) {
				cv::Point p2 = contour2[j];

				float distx = cv::norm(abs(p1.x - p2.x));  // 计算两点之间的距离
				float disty = cv::norm(abs(p1.y - p2.y));  // 计算两点之间的距离
				minDistX = (std::min)(minDistX, distx);    // 更新最小距离
				minDistY = (std::min)(minDistY, disty);    // 更新最小距离
			}
		}

		return true;
	}


}


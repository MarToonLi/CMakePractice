#include "yolov5.hpp"
using namespace nvinfer1;


//__global__ void warpaffine_kernel(
//	uint8_t* src, int src_line_size, int src_width,
//	int src_height, float* dst, int dst_width,
//	int dst_height, uint8_t const_value_st,
//	AffineMatrix d2s, int edge) {
//	int position = blockDim.x * blockIdx.x + threadIdx.x;
//	if (position >= edge) return;
//
//	float m_x1 = d2s.value[0];
//	float m_y1 = d2s.value[1];
//	float m_x2 = d2s.value[3];
//	float m_y2 = d2s.value[4];
//
//	int dx = position % dst_width;
//	int dy = position / dst_width;
//	float src_x = m_x1 * dx + m_y1 * dy;
//	float src_y = m_x2 * dx + m_y2 * dy;
//	float c0, c1, c2;
//
//	if (src_x < 0 || src_x + 1 >= src_width || src_y < 0 || src_y + 1 >= src_height) {
//		c0 = const_value_st;
//		c1 = const_value_st;
//		c2 = const_value_st;
//	}
//	else {
//		int x_low = floorf(src_x);
//		int y_low = floorf(src_y);
//		int x_high = x_low + 1;
//		int y_high = y_low + 1;
//		float w1 = (y_high - src_y) * (x_high - src_x);
//		float w2 = (y_high - src_y) * (src_x - x_low);
//		float w3 = (src_y - y_low) * (x_high - src_x);
//		float w4 = (src_y - y_low) * (src_x - x_low);
//		uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
//		uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
//		uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
//		uint8_t* v4 = src + y_high * src_line_size + x_high * 3;
//		c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
//		c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
//		c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
//	}
//
//	// bgr -> rgb
//	float temp = c2;
//	c2 = c0;
//	c0 = temp;
//
//	// normalization
//	c0 /= 255.0f;
//	c1 /= 255.0f;
//	c2 /= 255.0f;
//
//	// rgbrgbrgb -> rrrgggbbb
//	int area = dst_height * dst_width;
//	float* pdst_c0 = dst + dy * dst_width + dx;
//	float* pdst_c1 = pdst_c0 + area;
//	float* pdst_c2 = pdst_c1 + area;
//	*pdst_c0 = c0;
//	*pdst_c1 = c1;
//	*pdst_c2 = c2;
//}


void preprocess(
	uint8_t* src, const int& src_width, const int& src_height,
	float* dst, const int& dst_width, const int& dst_height,
	cudaStream_t stream, float& scale) {

	AffineMatrix s2d, d2s;
	scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);
	s2d.value[0] = scale;
	s2d.value[1] = 0;
	s2d.value[2] = 0;
	s2d.value[3] = 0;
	s2d.value[4] = scale;
	s2d.value[5] = 0;
	cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
	cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
	cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

	memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

	int jobs = dst_height * dst_width;
	int threads = 256;
	int blocks = ceil(jobs / (float)threads);
	//warpaffine_kernel << <blocks, threads, 0, stream >> > (
	//	src, src_width * 3, src_width,
	//	src_height, dst, dst_width,
	//	dst_height, 128, d2s, jobs);
}



static float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
	  (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
	  (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
	  (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
	  (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}

void NMS(std::vector<Detection>& res, float* output, const float& conf_thresh, const float& nms_thresh) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	for (int i = 0; i < output[0]; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));

		int index = 0;
		float cx = det.bbox[index];      //center_x      // 获取目标框的(x1, y1, x2, y2) 
		float cy = det.bbox[index + 1];  //center_y
		float w = det.bbox[index + 2];   //w
		float h = det.bbox[index + 3];   //h
		float xmin = (cx - 0.5 * w) * 1; // xmin
		float ymin = (cy - 0.5 * h) * 1; // ymin
		float xmax = (cx - 0.5 * w) * 1; // xmax
		float ymax = (cy - 0.5 * h) * 1; // ymax

		det.bbox[0] = xmin;
		det.bbox[1] = ymin;
		det.bbox[2] = xmax;
		det.bbox[3] = ymax;

		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}


void NMS_ori(std::vector<Detection>& res, float* output, const float& conf_thresh, const float& nms_thresh) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	for (int i = 0; i < output[0]; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void drawBbox(cv::Mat& img, std::vector<Detection>& res, float& scale, std::map<int, std::string>& Labels) {
	for (size_t j = 0; j < res.size(); j++) {
		float l = res[j].bbox[0] / scale;
		float t = res[j].bbox[1] / scale;
		float r = res[j].bbox[2] / scale;
		float b = res[j].bbox[3] / scale;
		cv::Rect rect = cv::Rect(int(l), int(t), int(r - l), int(b - t));
		std::string name = Labels[(int)res[j].class_id];
		cv::rectangle(img, rect, cv::Scalar(0xFF, 0xFF, 0), 2);
		cv::putText(img, name, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
	}
}


void drawBbox_xyxy(cv::Mat& img, std::vector<Detection>& res, float& scale, std::map<int, std::string>& Labels) {
	for (size_t j = 0; j < res.size(); j++) {
		float l = res[j].bbox[0] / scale;
		float t = res[j].bbox[1] / scale;
		float r = res[j].bbox[2] / scale;
		float b = res[j].bbox[3] / scale;
		cv::Rect rect = cv::Rect(int(l), int(t), int(r - l), int(b - t));
		std::string name = Labels[(int)res[j].class_id];
		cv::rectangle(img, rect, cv::Scalar(0xFF, 0xFF, 0), 2);
		cv::putText(img, name, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
	}
}


std::vector<unsigned char> loadEngineModel(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::binary);        // 以二进制方式读取
	assert(file.is_open() && "load engine model failed!"); // 断言

	file.seekg(0, std::ios::end); // 定位到文件末尾
	size_t size = file.tellg();   // 获取文件大小

	std::vector<unsigned char> data(size); // 创建一个vector，大小为size
	file.seekg(0, std::ios::beg);          // 定位到文件开头
	file.read((char*)data.data(), size);  // 读取文件内容到data中
	file.close();

	return data;
}


/** 模型初始化 */
Onnx_YOLOv5::Onnx_YOLOv5(Configuration config)
{
	try {
		this->objThreshold = config.objThreshold;
		this->nmsThreshold = config.nmsThreshold;
		this->confThresholds = config.confThresholds;
		this->bboxAreaThreshold = config.bboxAreaThreshold;
		std::string model_path = config.modelpath;
		this->inpHeight = 1120;
		this->inpWidth = 1120;
		this->num_classes = 2;


		std::wstring widestr = std::wstring(model_path.begin(), model_path.end());

		/** 配置GPU的options: sessionOptions */
		OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);  // 配置ONNX Runtime使用CUDA加速，0可能代表设备ID，即使用第一个GPU。
		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);                            // 设置ORT的图优化，提升性能（与GPU无关）

		ort_session = new Ort::Session(env, widestr.c_str(), sessionOptions);   // 初始化ONNX Runtime的会话ort_session，加载模型和GPU信息
		size_t numInputNodes = ort_session->GetInputCount();                    // 获取模型输入/输出节点数量
		size_t numOutputNodes = ort_session->GetOutputCount();

		Ort::AllocatorWithDefaultOptions allocator;                                   // 遍历获取所有输入节点名称
		for (int i = 0; i < numInputNodes; i++)                                       // 复制名称到input_names向量
		{
			auto temp_input_name = ort_session->GetInputNameAllocated(i, allocator);
			char* s = temp_input_name.get();
			int le = strlen(s);
			char* inp = new char[le + 1];
			strcpy(inp, s);
			input_names.push_back(inp);
		}
		for (int i = 0; i < numOutputNodes; i++)                                       // 复制名称到output_names向量
		{
			auto temp_output_name = ort_session->GetOutputNameAllocated(i, allocator);
			char* s = temp_output_name.get();
			int le = strlen(s);
			char* inp = new char[le + 1];
			strcpy(inp, s);
			output_names.push_back(inp);
		}

		this->nout = this->num_classes + 5; // wxywh score
		this->num_proposal = 77175;


		LOGI("Basic Model Infor:");
		LOGI("model_path  : {};", model_path);
		LOGI("(H, W)      : {};", this->inpHeight, this->inpWidth);
		LOGI("input_names : {};", wikky_algo::vectors2string(input_names));
		LOGI("output_names: {};", wikky_algo::vectors2string(output_names));
		LOGI("num_classes : {};", this->num_classes);
		LOGI("nout        : {};", this->nout);
		LOGI("\n");
	}
	catch (const cv::Exception& e)
	{
		LOGE("NG_UNDEFINED: e1: {};", e.what());                                    //? NG_ALGO_ERROR
	}
	catch (const std::exception& e)
	{
		LOGE("NG_UNDEFINED: e2: {};", e.what());                                    //? NG_ALGO_ERROR

	}
	catch (...)
	{
		LOGE("NG_UNDEFINED: e3: unkown;");                                          //? NG_ALGO_ERROR
	}
}

/** 输入resize */
cv::Mat Onnx_YOLOv5::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left, cv::Scalar& add_color)  //修改图片大小并填充边界防止失真
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, add_color);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;                         //上部缺失部分
			cv::copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, add_color);
			//上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else { cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA); }
	return dstimg;
}

/** 输入归一化和BR通道交换*/
void Onnx_YOLOv5::normalize_(std::vector<cv::Mat>& frames)  //归一化
{
	int row = this->inpHeight;
	int col = this->inpWidth;
	int chn = frames[0].channels();
	int batch = frames.size();
	this->input_image_.resize(batch * chn * col * row);  // vector大小
    


	if (batch > 0) {
		for (int k = 0; k < batch; k++) {
			for (int c = 0; c < chn; c++){
				for (int i = 0; i < row; i++){
					for (int j = 0; j < col; j++){
						float pix = frames[k].ptr<uchar>(i)[j * chn + 2 - c];  //! BR通道交换
						this->input_image_[row * col * chn * k + c * row * col + i * col + j] = pix / 255.0;					
					}
				}
			}
		}
	}

}


/** nms */
void Onnx_YOLOv5::nms(std::vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(), false);
	auto iou = [](const BoxInfo& box1, const BoxInfo& box2)
	{
		float xx1 = cv::max(box1.x1, box2.x1);
		float yy1 = cv::max(box1.y1, box2.y1);
		float xx2 = cv::min(box1.x2, box2.x2);
		float yy2 = cv::min(box1.y2, box2.y2);
		// 交集
		float w = cv::max(0.0f, xx2 - xx1 + 1);
		float h = cv::max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = cv::max(0.0f, box1.x2 - box1.x1) * cv::max(0.0f, box1.y2 - box1.y1)
			+ cv::max(0.0f, box2.x2 - box2.x1) * cv::max(0.0f, box2.y2 - box2.y1) - inter_area;
		return inter_area / union_area;
	};

	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (remove_flags[j]) continue;
			if (input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i], input_boxes[j]) >= this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}

	int idx_t = 0;
	// remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}


void Onnx_YOLOv5::draw(cv::Mat& imgrst, std::vector<std::vector<BoxInfo>>& output) {
	for (int i = 0; i < output[0].size(); i++) {
		BoxInfo cur_boxinfo = output[0][i];
		int class_idx = cur_boxinfo.label;
		float score = cur_boxinfo.score;

		cv::Rect rect = cv::Rect(cv::Point(int(cur_boxinfo.x1), int(cur_boxinfo.y1)), cv::Point(int(cur_boxinfo.x2), int(cur_boxinfo.y2)));
		std::string name = wikky_algo::d2str(class_idx);
		cv::rectangle(imgrst, rect, cv::Scalar(0xFF, 0xFF, 0), 2);
		cv::putText(imgrst, name, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
	}

}

/** detect */
bool Onnx_YOLOv5::onnx_detect(std::vector<cv::Mat>& frames, std::vector<std::vector<BoxInfo>>& output)
{
	output.clear();


	// resize
	dstImgs.clear();
	cv::Scalar add_color(114, 114, 114);        
	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Size new_shape(this->inpHeight, this->inpWidth);
	for (int i = 0; i < frames.size(); i++) {
		if (frames[i].data == nullptr) { return false;}
		//cv::Mat dstimg = this->letterbox(frames[i], new_shape, add_color, false, true, 32);
		cv::Mat dstimg = this->resize_image(frames[i], &newh, &neww, &padh, &padw, add_color);   //改大小后做padding防失真
		dstImgs.push_back(dstimg);
	}

	// normalize
	this->normalize_(dstImgs);       //归一化
	LOGD("normalize");



	
	// 输入数据的内存分配
	std::array<int64_t, 4> input_shape_{ frames.size(), 3, this->inpHeight, this->inpWidth };  // 定义一个静态数组大小确定， 1,3,640,640
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);       //  创建描述CPU内存分配的MemoryInfo对象。
	// 传递给ORT进行内存分配，确保模型的输入输出在 CPU 上分配和释放内存
	// OrtDeviceAllocator 是分配设备内存的标准分配器类型。它用来控制在 CPU 或 GPU 设备上如何分配内存
	// OrtMemTypeCPU 表示内存类型是用于 CPU 的内存分配。这意味着分配的内存将位于 CPU 上，而不是在 GPU 或其他硬件设备上
	//使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, 
																input_image_.data(), input_image_.size(), 
																input_shape_.data(), input_shape_.size()
	);
	LOGD("CreateTensor");


	// 基于onnxruntime会话进行推理
	std::vector<Ort::Value> ort_outputs;
	ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr },
			                                    &input_names[0],  // 指针类型
			                                    &input_tensor_,   // 指针类型
			                                    1, 
			                                    &output_names[0], // 指针类型
			                                    1);
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();         // 获取 ONNX 模型推理的可修改的底层输出数据指针，并将其转换为 float 类型的指针
	auto type_shape_info = ort_outputs[0].GetTensorTypeAndShapeInfo();   // 获取张量的形状
	size_t element_count = type_shape_info.GetElementCount();            // 直接获取元素总数
	LOGD("Run");


	
	// 后处理
	std::vector<BoxInfo> generate_boxes;        // BoxInfo自定义的结构体
	std::vector<BoxInfo> final_generate_boxes;  // BoxInfo自定义的结构体
	float ratioh = (float)frames[0].rows / newh, ratiow = (float)frames[0].cols / neww;  //原图高和新高比，原图宽与新宽比
	for (int m = 0; m < frames.size(); m++) {

		generate_boxes.clear();
		final_generate_boxes.clear();
		

		// 遍历所有的num_proposal, 可以omp加速
		for (int i = 0; i < num_proposal; ++i)    
		{
			int index = i * nout;                 // nout: x y w h score class1 class2 .....
			float obj_conf = pdata[index + 4];    // 置信度分数
	
			if (obj_conf > this->objThreshold)    // 大于阈值， 首先必须是个非背景物体
			{
				int class_idx = 0;
				float max_class_socre = 0;
				
				for (int k = 0; k < this->num_classes; k++)    // 更新当前框的最大可能的物体类别及其类别分数
				{
					if (pdata[k + index + 5] > max_class_socre)
					{
						max_class_socre = pdata[k + index + 5];
						class_idx = k;
					}
				}

				max_class_socre *= obj_conf;                  // 最大的类别分数*置信度
				                                
				float cx = pdata[index];      //center_x      // 获取目标框的(x1, y1, x2, y2) 
				float cy = pdata[index + 1];  //center_y
				float w = pdata[index + 2];   //w
				float h = pdata[index + 3];   //h
				float xmin = (cx - padw - 0.5 * w) * ratiow; // xmin
				float ymin = (cy - padh - 0.5 * h) * ratioh; // ymin
				float xmax = (cx - padw + 0.5 * w) * ratiow; // xmax
				float ymax = (cy - padh + 0.5 * h) * ratioh; // ymax


				if (xmin < 0 || ymin < 0 || xmax < 0 || ymax < 0) { continue; }    // 异常处理(左闭右开)
				if (xmin >= frames[0].cols || ymin >= frames[0].rows || xmax >= frames[0].cols || ymax >= frames[0].rows) { continue; }

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });  // 收纳box
			}
		}

		
		// nms处理: 去除重合度过高的框
		nms(generate_boxes);


		// 置信度过滤
		for (int i = 0; i < generate_boxes.size(); i++) {
			int class_idx = generate_boxes[i].label;
			float score = generate_boxes[i].score;
			if (score > this->confThresholds[class_idx]) { final_generate_boxes.push_back(generate_boxes[i]); }
		}

		output.push_back(final_generate_boxes);

		pdata += num_proposal * nout;
	}
	LOGD("postprocess");





	
	return true;
}








/** 模型初始化 */
Dnn_YOLOv5::Dnn_YOLOv5(Configuration config)
{
	this->objThreshold = config.objThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->confThresholds = config.confThresholds;
	this->bboxAreaThreshold = config.bboxAreaThreshold;
	std::string model_path = config.modelpath;
	this->inpHeight = 1120;
	this->inpWidth = 1120;
	this->num_classes = 80;
	this->nout = this->num_classes + 5; // wxywh score
	this->num_proposal = 77175;


	LOGI("Basic Model Infor:");
	LOGI("model_path  : {};", model_path);
	LOGI("(H, W)      : {};", this->inpHeight, this->inpWidth);
	LOGI("input_names : {};", wikky_algo::vectors2string(input_names));
	LOGI("output_names: {};", wikky_algo::vectors2string(output_names));
	LOGI("num_classes : {};", this->num_classes);
	LOGI("nout        : {};", this->nout);
	LOGI("\n");

}

/** 输入resize */
cv::Mat Dnn_YOLOv5::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left, cv::Scalar& add_color)  //修改图片大小并填充边界防止失真
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, add_color);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;                         //上部缺失部分
			cv::copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, add_color); //上部填补top大小，下部填补剩余部分，左右不填补
			//? 按道理这里因该写cv::Scalar(114, 114, 114)
		}
	}
	else { cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA); }
	return dstimg;
}

/** 输入归一化且BR通道交换 */
void Dnn_YOLOv5::normalize_(std::vector<cv::Mat>& frames)  //归一化
{
	int row = this->inpHeight;
	int col = this->inpWidth;
	int chn = frames[0].channels();
	int batch = frames.size();
	this->input_image_.resize(batch * chn * col * row);  // vector大小

	if (batch > 0) {
		for (int k = 0; k < batch; k++) {
			for (int c = 0; c < chn; c++) {
				for (int i = 0; i < row; i++) {
					for (int j = 0; j < col; j++) {
						float pix = frames[k].ptr<uchar>(i)[j * chn + 2 - c];                                //! B和G交换通道!
						this->input_image_[row * col * chn * k + c * row * col + i * col + j] = pix / 255.0;
					}
				}
			}
		}
	}
}

/** nms */
void Dnn_YOLOv5::nms(std::vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(), false);
	auto iou = [](const BoxInfo& box1, const BoxInfo& box2)
		{
			float xx1 = cv::max(box1.x1, box2.x1);
			float yy1 = cv::max(box1.y1, box2.y1);
			float xx2 = cv::min(box1.x2, box2.x2);
			float yy2 = cv::min(box1.y2, box2.y2);
			// 交集
			float w = cv::max(0.0f, xx2 - xx1 + 1);
			float h = cv::max(0.0f, yy2 - yy1 + 1);
			float inter_area = w * h;
			// 并集
			float union_area = cv::max(0.0f, box1.x2 - box1.x1) * cv::max(0.0f, box1.y2 - box1.y1)
				+ cv::max(0.0f, box2.x2 - box2.x1) * cv::max(0.0f, box2.y2 - box2.y1) - inter_area;
			return inter_area / union_area;
		};

	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (remove_flags[j]) continue;
			if (input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i], input_boxes[j]) >= this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}

	int idx_t = 0;
	// remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

// 加载网络
int Dnn_YOLOv5::load_dnn_net(cv::dnn::Net& net, std::string model_path, bool is_cuda)
{
	try {
		//? 似乎dnn并不支持yolov5的某些算子?
		// 检查DNN模块是否可用
		auto result = cv::dnn::readNet(model_path);
		// (-215:Assertion failed) splits ＞ 0 && inpShape[axis_rw] % splits == 0 in function ‘getMemoryShapes‘
		// 降低opset版本即可，--opset=11

		if (is_cuda)
		{
			LOGI("Attempty to use CUDA");
			result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		}
		else
		{
			LOGI("Running on CPU");
			result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}
		net = result;
	}
	catch (const cv::Exception& e)
	{
		LOGE("NG_UNDEFINED: e1: {};", e.what());                                    //? NG_ALGO_ERROR
		return -1;
	}
	catch (const std::exception& e)
	{
		LOGE("NG_UNDEFINED: e2: {};", e.what());                                    //? NG_ALGO_ERROR
		return -1;

	}
	catch (...)
	{
		LOGE("NG_UNDEFINED: e3: unkown;");                                          //? NG_ALGO_ERROR
		return -1;
	}
	return 1;
}

/** detect */
bool Dnn_YOLOv5::dnn_detect(std::vector<cv::Mat>& frames, cv::dnn::Net& net, std::vector<std::vector<BoxInfo>>& output)
{
	// https://docs.opencv.org/4.5.5/d4/db9/samples_2dnn_2object_detection_8cpp-example.html




	output.clear();


	// resize
	dstImgs.clear();
	cv::Scalar add_color(114, 114, 114);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Size new_shape(this->inpHeight, this->inpWidth);
	float ratioh = (float)frames[0].rows / newh, ratiow = (float)frames[0].cols / neww;  //原图高和新高比，原图宽与新宽比



	for (int i = 0; i < frames.size(); i++) {
		if (frames[i].data == nullptr) { return false; }
		//cv::Mat dstimg = this->letterbox(frames[i], new_shape, add_color, false, true, 32);
		cv::Mat dstimg = this->resize_image(frames[i], &newh, &neww, &padh, &padw, add_color);          //改大小后做padding防失真
		cv::Mat dstimg2 = cv::dnn::blobFromImage(
			frames[i],          // 输入源图像（原始BGR格式）
			1. / 255.,          // 缩放因子（将像素值从0-255归一化到0-1）
			cv::Size(this->inpHeight, this->inpWidth), // 目标尺寸（模型要求的输入分辨率）
			cv::Scalar(0, 0, 0),      // 均值减法参数（这里未使用，如需使用应填如 Scalar(104, 117, 123)）
			true,             // 交换红蓝通道（BGR→RGB转换，大多数模型需要RGB输入）
			false              // 是否裁剪调整（保持长宽比时为true，直接缩放为false）
		);
		LOGD("dstimg.shape: w:{}, h:{};", dstimg.cols, dstimg.rows);


		//? 目前使用的onnx本就不是基于数据线数据训练出来的，因此其推理结果是否为0不能深究；

		net.setInput(dstimg2);

		//网络计算到指定层（第二个参数指定的层），并返回该层的所有输出
		std::vector<cv::Mat> outputs;

		//cv::Mat detections = net.forward();

		net.forward(outputs, net.getUnconnectedOutLayersNames());  // getUnconnectedOutLayersNames()返回具有未连接输出的层的名称,返回最终输出层
		// warning: setUpNet DNN module was not built with CUDA backend; switching to CPU
		// GPU没被用上，原因是OPENCV在安装的时候，没有安装支持CUDA的版本，所以需要重新编译安装OPENCV


		std::vector<BoxInfo> generate_boxes;        // BoxInfo自定义的结构体
		std::vector<BoxInfo> final_generate_boxes;  // BoxInfo自定义的结构体


		for (size_t k = 0; k < outputs.size(); k++)
		{
			float* pdata = (float*)outputs[k].data;

			// 遍历所有的num_proposal, 可以omp加速
			for (int i = 0; i < num_proposal; ++i)
			{
				int index = i * nout;                 // nout: x y w h score class1 class2 .....
				float obj_conf = pdata[index + 4];    // 置信度分数

				if (obj_conf > this->objThreshold)    // 大于阈值， 首先必须是个非背景物体
				{
					int class_idx = 0;
					float max_class_socre = 0;

					for (int k = 0; k < this->num_classes; k++)    // 更新当前框的最大可能的物体类别及其类别分数
					{
						if (pdata[k + index + 5] > max_class_socre)
						{
							max_class_socre = pdata[k + index + 5];
							class_idx = k;
						}
					}

					max_class_socre *= obj_conf;                  // 最大的类别分数*置信度

					float cx = pdata[index];      //center_x      // 获取目标框的(x1, y1, x2, y2) 
					float cy = pdata[index + 1];  //center_y
					float w = pdata[index + 2];   //w
					float h = pdata[index + 3];   //h
					float xmin = (cx - 0.5 * w) * ratiow; // xmin
					float ymin = (cy - 0.5 * h) * ratioh; // ymin
					float xmax = (cx - 0.5 * w) * ratiow; // xmax
					float ymax = (cy - 0.5 * h) * ratioh; // ymax


					if (xmin < 0 || ymin < 0 || xmax < 0 || ymax < 0) { continue; }    // 异常处理(左闭右开)
					if (xmin >= frames[0].cols || ymin >= frames[0].rows || xmax >= frames[0].cols || ymax >= frames[0].rows) { continue; }

					generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });  // 收纳box
				}

			}
		}


		//std::vector<int> nms_result;
		//cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

		// nms处理: 去除重合度过高的框
		nms(generate_boxes);


		// 置信度过滤
		for (int i = 0; i < generate_boxes.size(); i++) {
			int class_idx = generate_boxes[i].label;
			float score = generate_boxes[i].score;
			if (score > 0.3) { final_generate_boxes.push_back(generate_boxes[i]); }
		}

		output.push_back(final_generate_boxes);



		dstImgs.push_back(dstimg);
	}

	return true;
}




/**********************************
 * @brief 先resize、再减均值、除方差
 *
 * @param src
 * @param dst
 * @return
 *********************************/
void Tensorrt_YOLOv5::preData(cv::Mat& matSrc, cv::Mat& matDst)
{
	cv::resize(matSrc, matSrc, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	cv::Mat matMean(MODEL_HEIGHT, MODEL_WIDTH, CV_32FC3, \
		cv::Scalar(103.53f, 116.28f, 123.675f)); // 均值
	cv::Mat matStd(MODEL_HEIGHT, MODEL_WIDTH, CV_32FC3, \
		cv::Scalar(1.0f, 1.0f, 1.0f)); // 方差
	cv::Mat matF32Img;
	matSrc.convertTo(matF32Img, CV_32FC3);
	matDst = (matF32Img - matMean) / matStd;
	LOGD("--");
}

/** 输入resize */
cv::Mat Tensorrt_YOLOv5::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left, cv::Scalar& add_color)  //修改图片大小并填充边界防止失真
{
	this->inpHeight = 640;
	this->inpWidth = 640;

	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, add_color);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;                         //上部缺失部分
			cv::copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, add_color);
			//上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else { cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA); }
	return dstimg;
}

int Tensorrt_YOLOv5::tensorrt_detect(std::string strTrtSavedPath)
{
	//std::string strTrtSavedPath = "./savedTrt.trt";
	// gLogger
	// gLogger是一个日志类，必须要有，但又不是那么重要，可以自己继承
	IRuntime* runtime = createInferRuntime(gLogger);
	std::ifstream fin(strTrtSavedPath);

	// 1、将文件中的内容读取至cached_engine字符串
	//std::string modelData = "";
	//while (fin.peek() != EOF) { // 使用fin.peek()防止文件读取时无限循环
	//	std::stringstream buffer;
	//	buffer << fin.rdbuf();
	//	modelData.append(buffer.str());
	//}
	//fin.close();

	std::ifstream file_ptr(strTrtSavedPath, std::ios::binary);
	if (!file_ptr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
		//return -1;
	}
	//size_t size = 0;
	//file_ptr.seekg(0, file_ptr.end);	// 将读指针从文件末尾开始移动0个字节
	//size = file_ptr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	//file_ptr.seekg(0, file_ptr.beg);	// 将读指针从文件开头开始移动0个字节
	//char* model_stream = new char[size];
	//file_ptr.read(model_stream, size);
	//file_ptr.close();


	ifstream in(strTrtSavedPath, ios::in | ios::binary);
	if (!file_ptr.is_open()) {
		return -1;
	}

	in.seekg(0, ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> fdata;
	if (length > 0) {
		in.seekg(0, ios::beg);
		fdata.resize(length);

		in.read((char*)&fdata[0], length);
	}
	in.close();



	// 2、 将序列化得到的结果进行反序列化，以执行后续的inference
	ICudaEngine* engine = runtime->deserializeCudaEngine(fdata.data(), fdata.size(), nullptr);
	// inference推断过程

	IExecutionContext* context = engine->createExecutionContext();   // inference推断过程

	int nInputIdx = engine->getBindingIndex("images");     // 获取输入节点索引
	int nOutputIndex = engine->getBindingIndex("output0");  // 获取输出节点索引
	int nNumIndex = engine->getNbBindings();            // 获取总索引个数
	char achInputTensorName[128];
	strcpy(achInputTensorName, engine->getBindingName(nInputIdx));//获取对应索引的节点名
	bool bIsInputTensor = engine->bindingIsInput(nInputIdx); // 是否是输入节点
	Dims tDimInput = engine->getBindingDimensions(nInputIdx); // 获取输入检点的维度
	DataType emInputDataType = engine->getBindingDataType(nInputIdx);// 获取输入数据的类型
	int nMaxBatchSize = engine->getMaxBatchSize();    // 获取最大BatchSize
	int nNumLayers = engine->getNbLayers(); // 获取网络层的个数
	//int nWorkSpaceSize = engine->getWorkspaceSize(); // 获取工作空间的大小, 通常小于设置的值
	TensorLocation emLocattion = engine->getLocation(nInputIdx); // 获取索引对应的tensor在gpu上还是cpu上
	int nBindingSize = engine->getBindingBytesPerComponent(0); //返回元素每个组成部分的字节数
	TensorFormat emTensorFormat = engine->getBindingFormat(nInputIdx); // 返回数据格式

	std::cout << " 输入节点索引 nInputIdx = " << nInputIdx << std::endl;
	std::cout << " 输入节点索引 nOutputIdx = " << nOutputIndex << std::endl;
	std::cout << " 总索引个数 nNumIndex = " << engine->getNbBindings() << std::endl;
	std::cout << " input 节点名 = " << achInputTensorName << std::endl;
	for (int i = 0; i < tDimInput.nbDims; ++i)
	{
		std::cout << " 输入维度 dim[" << i << "] = " << tDimInput.d[i] << std::endl;
	}
	//申请GPU显存
	std::cout << " 输入数据类型为 " << int(emInputDataType) << std::endl;
	//std::cout << " 工作空间大小为 " << nWorkSpaceSize << std::endl;
	std::cout << " 输入数据在 " << int(emLocattion) << std::endl;
	std::cout << " 每个元素空间字节大小：" << nBindingSize << std::endl;
	std::cout << " 输入数据格式： " << int(emTensorFormat) << std::endl;
	std::cout << " 输入数据格式：　" << engine->getBindingFormatDesc(nInputIdx) << std::endl;
	// inference推断过程
	std::cout << " 网络layer的个数：" << nNumLayers << std::endl;
	LOGD("网络layer的个数");

	void* buffers[2] = { NULL, NULL };
	int nBatchSize = 1;
	int nOutputSize = MODEL_OUTPUT_SIZE;
	CHECK(cudaMalloc(&buffers[nInputIdx], nBatchSize * MODEL_CHANNEL * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float)));
	CHECK(cudaMalloc(&buffers[nOutputIndex], nBatchSize * nOutputSize * sizeof(float)));
	// cudaMalloc(&buffers[nInputIdx], nBatchSize * MODEL_CHANNEL * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float));
	// cudaMalloc(&buffers[nOutputIndex], nBatchSize * nOutputSize * sizeof(float));

	// 创建cuda流
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	//cudaStreamCreate(&stream);
	cudaEvent_t start, end; //calculate run time
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&end));
	//cudaEventCreate(&start);
	//cudaEventCreate(&end);

	cv::Mat matBgrImg = cv::imread("D:/58_FGJHAT005TZ000033G-1_DA3180921.png");
	cv::Mat matNormImage;
	preData(matBgrImg, matNormImage); // 减均值除方差


	std::vector<std::vector<cv::Mat>> nChannels;
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(matNormImage, rgbChannels);
	nChannels.push_back(rgbChannels); //  NHWC  转NCHW 

	float* data = (float*)malloc(nBatchSize * MODEL_CHANNEL * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float));
	if (NULL == data)
	{
		printf("malloc error!\n");
		return 0;
	}

	try {
		for (int c = 0; c < 3; c++)
		{
			cv::Mat cur_imag_plane = nChannels[0][c];
			size_t temp = MODEL_HEIGHT * MODEL_WIDTH * sizeof(float);
			memcpy(data + c * MODEL_HEIGHT * MODEL_WIDTH, cur_imag_plane.ptr<float>(0), temp);
		}
	}
	catch (const cv::Exception& e)
	{
		LOGW("e1: {};", e.what());
	}


	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[nInputIdx], data, \
	nBatchSize * MODEL_CHANNEL * MODEL_WIDTH * MODEL_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream));
	//cudaMemcpyAsync(buffers[nInputIdx], data, nBatchSize* MODEL_CHANNEL* MODEL_WIDTH* MODEL_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream);
	//bool bIsSucess = context->execute(nBatchSize, buffers); // 同步执行
	bool bIsSucess = context->enqueueV2(buffers, stream, nullptr); // 同步执行
	if (!bIsSucess)
	{
		std::cerr << " 推断执行失败 " << std::endl;
		return -1;
	}

	context->setName("Vgg16");
	std::cout << "context 名为　" << context->getName() << std::endl;

	tDimInput = context->getStrides(nInputIdx);
	for (int i = 0; i < tDimInput.nbDims; ++i)
	{
		std::cout << " 输入维度 dim[" << i << "] = " << tDimInput.d[i] << std::endl;
	}

	printf("\nend ... TensorRt \n");
	return 0;

}

// A helper function to calculate memory usage
size_t Tensorrt_YOLOv5::get_memory_size(const nvinfer1::Dims& dims, const int32_t elem_size)
{
	return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

int Tensorrt_YOLOv5::tensorrt_detect2(std::string strTrtSavedPath)
{
	
	int c = 0; int j = 0;

	try
	{
		//!
		

	}
	catch (const cv::Exception& e)
	{
		LOGW("c: {}: j: {}:", c, j);
		LOGW("e1: {};", e.what());
	}
	catch (const std::exception& e) {
		LOGW("e2: {}", e.what());
	}


	return 1;
}


#include "yolov5.hpp"




YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->bboxAreaThreshold = config.bboxAreaThreshold;
	std::string model_path = config.modelpath;
	this->inpHeight = 1120;
	this->inpWidth = 1120;
	this->num_classes = 2;

	



	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Ort::Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto temp_input_name = ort_session->GetInputNameAllocated(i, allocator);
		char* s = temp_input_name.get();
		int le = strlen(s);
		char* inp = new char[le + 1];
		strcpy(inp, s);
		input_names.push_back(inp);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		auto temp_output_name = ort_session->GetOutputNameAllocated(i, allocator);
		char* s = temp_output_name.get();
		int le = strlen(s);
		char* inp = new char[le + 1];
		strcpy(inp, s);
		output_names.push_back(inp);
	}

	this->nout = this->num_classes + 5;
	this->num_proposal = 77175;


}



cv::Mat YOLOv5::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left)  //修改图片大小并填充边界防止失真
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
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;  //上部缺失部分
			cv::copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114); //上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::normalize_(std::vector<cv::Mat>& frames)  //归一化
{
	//    img.convertTo(img, CV_32F);
	//cout<<"picture size"<<img.rows<<img.cols<<img.channels()<<endl;
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
						float pix = frames[k].ptr<uchar>(i)[j * chn + 2 - c];
						this->input_image_[row * col * chn * k + c * row * col + i * col + j] = pix / 255.0;					
					}
				}
			}
		}
	}

}

void YOLOv5::nms(std::vector<BoxInfo>& input_boxes)
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

bool YOLOv5::detect(std::vector<cv::Mat>& frames, std::vector<std::vector<BoxInfo>>& output)
{
	output.clear();
	cv::Size new_shape(this->inpHeight, this->inpWidth);
	cv::Scalar add_color(114, 114, 114);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	int width = 0;
	int height = 0;
	int area = 0;
	dstImgs.clear();
	for (int i = 0; i < frames.size(); i++) {
		if (frames[i].data == nullptr) {
			return false;
		}


		//cv::Mat dstimg = this->letterbox(frames[i], new_shape, add_color, false, true, 32);
		cv::Mat dstimg = this->resize_image(frames[i], &newh, &neww, &padh, &padw);   //改大小后做padding防失真
		dstImgs.push_back(dstimg);
	}


	this->normalize_(dstImgs);       //归一化
	

	
	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	std::array<int64_t, 4> input_shape_{ frames.size(), 3, this->inpHeight, this->inpWidth };  //1,3,640,640

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	//使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	// 开始推理
	std::vector<Ort::Value> ort_outputs;


	LOGD("1");
	ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr },
			                                    &input_names[0],
			                                    &input_tensor_, 
			                                    1, 
			                                    &output_names[0], 
			                                    1);   // 开始推理
	LOGD("2");

	
	std::vector<BoxInfo> generate_boxes;  // BoxInfo自定义的结构体
	std::vector<BoxInfo> final_generate_boxes;  // BoxInfo自定义的结构体
	float ratioh = (float)frames[0].rows / newh, ratiow = (float)frames[0].cols / neww;  //原图高和新高比，原图宽与新宽比

	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData

	
	for (int m = 0; m < frames.size(); m++) {

		generate_boxes.clear();
		final_generate_boxes.clear();
		
		for (int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
		{

			int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
			//float obj_conf = pdata[index + 4];  // 置信度分数
			float obj_conf = pdata[index + 4];  // 置信度分数
			

	

			if (obj_conf > this->objThreshold)  // 大于阈值
			{
				int class_idx = 0;
				float max_class_socre = 0;

		

				for (int k = 0; k < this->num_classes; k++)
				{
					if (pdata[k + index + 5] > max_class_socre)
					{
						max_class_socre = pdata[k + index + 5];
						class_idx = k;
					}
				}

				max_class_socre *= obj_conf;   // 最大的类别分数*置信度
				
				
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index + 1];  //y
				float w = pdata[index + 2];  //w
				float h = pdata[index + 3];  //h
				//cout<<cx<<cy<<w<<h<<endl;
				float xmin = (cx - padw - 0.5 * w) * ratiow;
				float ymin = (cy - padh - 0.5 * h) * ratioh;
				float xmax = (cx - padw + 0.5 * w) * ratiow;
				float ymax = (cy - padh + 0.5 * h) * ratioh;

				if (xmin < 0 || ymin < 0 || xmax < 0 || ymax < 0) { continue; }
				if (xmin >= frames[0].cols || ymin >= frames[0].rows || xmax >= frames[0].cols || ymax >= frames[0].rows) { continue; }

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
							
			}

			
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		
		
		nms(generate_boxes);

		std::cout << generate_boxes.size() << std::endl;
		for (int i = 0; i < generate_boxes.size(); i++) {
			int class_idx = generate_boxes[i].label;
			float score = generate_boxes[i].score;

			float tmp_conf = this->confThreshold;
			if (class_idx == 0) { tmp_conf = 0.5f; }

			if (score > tmp_conf) {
				final_generate_boxes.push_back(generate_boxes[i]);
			}
			
		}

		output.push_back(final_generate_boxes);


		pdata += num_proposal * nout;
	}
	
	return true;
}

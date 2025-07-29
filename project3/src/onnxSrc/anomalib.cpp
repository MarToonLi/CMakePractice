#include "anomalib.hpp"
using namespace nvinfer1;


namespace Anomalib
{
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
	Onnx_Anomalib::Onnx_Anomalib(Configuration config)
	{
		try {
			std::string model_path = config.modelpath;
			this->inpHeight = 256;
			this->inpWidth = 256;
			this->num_classes = 2;
			std::wstring widestr = std::wstring(model_path.begin(), model_path.end());


			/** 配置GPU的options: sessionOptions */
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);  // 配置ONNX Runtime使用CUDA加速，0可能代表设备ID，即使用第一个GPU。
			sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);                            // 设置ORT的图优化，提升性能（与GPU无关）

			ort_session = new Ort::Session(env, widestr.c_str(), sessionOptions);   // 初始化ONNX Runtime的会话ort_session，加载模型和GPU信息
			size_t numInputNodes = ort_session->GetInputCount();                    // 获取模型输入/输出节点数量
			size_t numOutputNodes = ort_session->GetOutputCount();

			std::vector<std::string> outputNodeNames;
			std::vector<std::string> inputNodeNames;


			Ort::AllocatorWithDefaultOptions allocator;                                   // 遍历获取所有输入节点名称
			for (int i = 0; i < numInputNodes; i++)                                       // 复制名称到input_names向量
			{
				auto temp_input_name = ort_session->GetInputNameAllocated(i, allocator);
				char* s = temp_input_name.get();
				int le = strlen(s);
				char* inp = new char[le + 1];
				strcpy(inp, s);
				input_names.push_back(inp);

				// 输出节点的相关信息

				std::shared_ptr<char> inputName = std::move(ort_session->GetInputNameAllocated(i, allocator));
				std::string inputNodeName = std::string(inputName.get(), strlen(inputName.get()));
				inputNodeNames.push_back(inputNodeName);


				Ort::TypeInfo inputTypeInfo = ort_session->GetInputTypeInfo(i);
				auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
				ONNXTensorElementDataType inputNodeDataType = input_tensor_info.GetElementType();
				std::vector<int64_t> inputTensorShape = input_tensor_info.GetShape();
				LOGD("inputNode.name: {}; shape: {};", inputNodeName, wikky_algo::vectors2string(inputTensorShape));
			}


			for (int i = 0; i < numOutputNodes; i++)                                       // 复制名称到output_names向量
			{
				auto temp_output_name = ort_session->GetOutputNameAllocated(i, allocator);
				char* s = temp_output_name.get();
				int le = strlen(s);
				char* inp = new char[le + 1];
				strcpy(inp, s);
				output_names.push_back(inp);



				std::shared_ptr<char> outputName = std::move(ort_session->GetOutputNameAllocated(i, allocator));
				std::string outputNodeName = std::string(outputName.get(), strlen(outputName.get()));
				outputNodeNames.push_back(outputNodeName);


				Ort::TypeInfo outputTypeInfo = ort_session->GetOutputTypeInfo(i);
				auto output_tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
				ONNXTensorElementDataType outputNodeDataType = output_tensor_info.GetElementType();
				std::vector<int64_t> outputTensorShape = output_tensor_info.GetShape();
				LOGD("inputNode.name: {}; shape: {};", outputNodeName, wikky_algo::vectors2string(outputTensorShape));

			}

			this->nout = this->num_classes + 5; // wxywh score
			this->num_proposal = 77175;


			LOGD("inputNodeNames.name: {};", wikky_algo::vectors2string(inputNodeNames));
			LOGD("outputNodeNames.name: {};", wikky_algo::vectors2string(outputNodeNames));

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
	cv::Mat Onnx_Anomalib::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left, cv::Scalar& add_color)  //修改图片大小并填充边界防止失真
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



	// 纯normalization
	template<typename T>
	bool BlobFromImage(std::vector<cv::Mat>& imgs, T & iBlob) {
		int batch = imgs.size();
		int channels = imgs[0].channels();
		int imgHeight = imgs[0].rows;
		int imgWidth = imgs[0].cols;

		for (int b = 0; b < batch; b++)
		{
			for (int c = 0; c < channels; c++)
			{
				for (int h = 0; h < imgHeight; h++)
				{
					for (int w = 0; w < imgWidth; w++)
					{
						iBlob[b * c * imgWidth * imgHeight + c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
							(imgs[b].at<cv::Vec3b>(h, w)[c]) / 255.0f);
					}
				}
			}
		}
		return true;
	}




	/** 输入归一化和BR通道交换*/
	void Onnx_Anomalib::normalize_(std::vector<cv::Mat>& frames)  //归一化
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
							float pix = frames[k].ptr<uchar>(i)[j * chn + 2 - c];  //! BR通道交换
							this->input_image_[row * col * chn * k + c * row * col + i * col + j] = pix / 255.0;
						}
					}
				}
			}
		}
	}


	void Onnx_Anomalib::draw(cv::Mat& imgrst, Anomalib_Detection& output) {

		if (imgrst.channels() == 1) {
			cv::cvtColor(imgrst, imgrst, cv::COLOR_GRAY2RGB);
		}

		float alpha = 0.8;


		cv::Mat heat_map = output.anomaly_map;


		cv::threshold(heat_map, heat_map, 0.5, 255, cv::THRESH_TOZERO);

		// 2. 检查热力图数据类型并归一化到 [0, 255]
		cv::Mat heat_map_normalized;
		if (heat_map.depth() == CV_32F) {
			// 浮点型热力图（假设值范围 [0, 1]）
			heat_map.convertTo(heat_map_normalized, CV_8UC1, 255.0);
		}
		else if (heat_map.depth() == CV_8U) {
			// 8 位整型热力图（直接使用或拉伸）
			if (heat_map.type() == CV_8UC1) {
				double min_val, max_val;
				cv::minMaxLoc(heat_map, &min_val, &max_val);
				if (max_val <= 1) {
					heat_map_normalized = heat_map * 255;
				}
				else {
					heat_map_normalized = heat_map.clone();
				}
			}
		}
		else {
			CV_Error(cv::Error::StsUnsupportedFormat, "Heatmap must be CV_32FC1 or CV_8UC1");
		}



		// 3. 调整热力图尺寸与原始图像匹配
		cv::Mat heat_map_resized;
		cv::resize(
			heat_map_normalized,
			heat_map_resized,
			cv::Size(imgrst.cols, imgrst.rows),  // OpenCV 的 Size 是 (width, height)
			cv::INTER_LINEAR
		);

		// 4. 将单通道热力图转为 3 通道（用于颜色映射）
		cv::Mat heat_map_3ch;
		cv::merge(std::vector<cv::Mat>{heat_map_resized, heat_map_resized, heat_map_resized}, heat_map_3ch);

		// 5. 加权融合（alpha * heatmap + beta * image）
		cv::Mat fused_image;
		float beta = 1.0f - alpha;
		cv::addWeighted(
			imgrst,              // 原始图像
			beta,               // 原始图像权重
			heat_map_3ch,       // 热力图（3 通道）
			alpha,              // 热力图权重
			0,                  // 亮度偏移
			fused_image         // 输出
		);

		imgrst = fused_image;
		cv::putText(imgrst,
			wikky_algo::d2str(output.pred_label,2) + "  " + wikky_algo::d2str(output.pred_score, 2),
			cv::Point(20, 10), 1, 1, cv::Scalar(0, 0, 255), 1);
	}



	//!? 目前仅支持单张处理！
	/** detect */
	bool Onnx_Anomalib::onnx_detect(std::vector<cv::Mat>& frames, std::vector<Anomalib_Detection>& output)
	{
		output.clear();


		// resize 成 256
		dstImgs.clear();
		int batch = frames.size();
		int height = frames[0].rows;
		int width = frames[0].cols;
		int channel = frames[0].channels();



		cv::Scalar add_color(114, 114, 114);
		int newh = 0, neww = 0, padh = 0, padw = 0;
		cv::Size new_shape(this->inpHeight, this->inpWidth);
		for (int i = 0; i < frames.size(); i++) {
			if (frames[i].data == nullptr) { return false; }

			// 相机采集的是BGR图像，训练使用RGB图像
			cv::cvtColor(frames[i], frames[i], cv::COLOR_BGR2RGB);


			// resize or 
			cv::Mat dstimg = this->resize_image(frames[i], &newh, &neww, &padh, &padw, add_color);   //改大小后做padding防失真


			// normalize
			dstImgs.push_back(dstimg);
		}



		// normalize
		int blob_size = batch * channel * height * width;
		float* blob = new float[blob_size];
		BlobFromImage(dstImgs, blob);
		LOGD("normalize");




		// 输入数据的内存分配
		std::array<int64_t, 4> input_shape_{ frames.size(), 3, this->inpHeight, this->inpWidth };  // 定义一个静态数组大小确定， 1,3,640,640
		auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);       //  创建描述CPU内存分配的MemoryInfo对象。
		// 传递给ORT进行内存分配，确保模型的输入输出在 CPU 上分配和释放内存
		// OrtDeviceAllocator 是分配设备内存的标准分配器类型。它用来控制在 CPU 或 GPU 设备上如何分配内存
		// OrtMemTypeCPU 表示内存类型是用于 CPU 的内存分配。这意味着分配的内存将位于 CPU 上，而不是在 GPU 或其他硬件设备上
		//使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
		//Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info,
		//	input_image_.data(), input_image_.size(),
		//	input_shape_.data(), input_shape_.size()
		//);

		Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info,
			blob, blob_size,
			input_shape_.data(), input_shape_.size()
		);

		LOGD("CreateTensor");


		// 基于onnxruntime会话进行推理
		std::vector<Ort::Value> ort_outputs;
		ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr },
			&input_names[0],  // 指针类型
			&input_tensor_,   // 指针类型
			input_names.size(),
			&output_names[0], // 指针类型
			output_names.size());


		LOGD("ort_outputs.size: {};", ort_outputs.size());

		Anomalib_Detection anoma_detect_output;

		for (int i = 0; i < ort_outputs.size(); i++)
		{
			std::string outputNodeName = output_names[i];
			Ort::Value& output_value = ort_outputs[i];
			Ort::TensorTypeAndShapeInfo tensor_type_shape_info = ort_outputs[i].GetTensorTypeAndShapeInfo();   // 获取张量的形状
			std::vector<int64_t> outputTensorShape = tensor_type_shape_info.GetShape();                        // 直接获取元素总数
			size_t element_count = tensor_type_shape_info.GetElementCount();            // 直接获取元素总数

			LOGD("inputNode.name: {}; shape: {};", outputNodeName, wikky_algo::vectors2string(outputTensorShape));

			float* pdata = output_value.GetTensorMutableData<float>();         // 获取 ONNX 模型推理的可修改的底层输出数据指针，并将其转换为 float 类型的指针

			if (element_count == 1) 
			{
				LOGD("inputNode.name: {}; val: {};", outputNodeName, wikky_algo::d2str(pdata[0]));
			}


			if (0 == i) { anoma_detect_output.pred_score = pdata[0]; }
			if (1 == i) { anoma_detect_output.pred_label = pdata[0]; }
			if (2 == i) { 
				int _height = outputTensorShape[2];
				int _width = outputTensorShape[3];
				int _channel = 1;
				int _depth = sizeof(float);

				int count = 0;
				std::vector<float> pixels;
				for (int i = 0; i < _height * _width * _channel; i++)
				{
					if (pdata[i] > 0 && count < 10) {
						pixels.push_back(pdata[i]);
						count++;
					}
				}
				LOGD("pixels: {};", wikky_algo::vectors2string(pixels));

				cv::Mat lastimg = cv::Mat(cv::Size(_width, _height), CV_MAKETYPE(CV_32F, _channel), _channel);
				memcpy(lastimg.data, (void*)pdata, _width * _height * _depth * _channel);
				
				anoma_detect_output.anomaly_map = lastimg;
			}
			if (3 == i) { 
				int _height = outputTensorShape[2];
				int _width = outputTensorShape[3];
				int _channel = 1;
				int _depth = 1;

				int count = 0;
				std::vector<float> pixels;
				for (int i = 0; i < _height * _width * _channel; i++)
				{
					if (pdata[i] > 0 && count < 10) {
						pixels.push_back(pdata[i]);
						count++;
					}
				}
				LOGD("pixels: {};", wikky_algo::vectors2string(pixels));


				cv::Mat lastimg = cv::Mat(cv::Size(_width, _height), CV_MAKETYPE(CV_8U, _channel), _channel);
				memcpy(lastimg.data, (void*)pdata, _width * _height * _depth * _channel);
				
				anoma_detect_output.pred_mask = lastimg;
			}
		}

		output.push_back(anoma_detect_output);

		LOGD("postprocess");
		return true;
	}

}





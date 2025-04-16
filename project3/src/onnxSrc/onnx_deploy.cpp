#include "onnx_deploy.h"

bool outputGPU()
{
	LOGD("outputGPU enter.");
	try
	{

		nvmlReturn_t result0;
		unsigned int device_count;
		result0 = nvmlInit();

		result0 = nvmlDeviceGetCount(&device_count);
		if (NVML_SUCCESS != result0)
		{
			std::cout << "Failed to query device count: " << nvmlErrorString(result0);
			return -1;
		}
		std::cout << "Found" << device_count << " device" << std::endl;
		std::stringstream availGPUTxt;

		for (int i = 0; i < device_count; i++)
		{
			nvmlDevice_t device;
			nvmlPciInfo_t pci;
			result0 = nvmlDeviceGetHandleByIndex(i, &device);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get device count: " << nvmlErrorString(result0);
				return false;
			}

			char name[NVML_DEVICE_NAME_BUFFER_SIZE];
			result0 = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get device name: " << nvmlErrorString(result0);
				return false;
			}

			nvmlUtilization_t utilization;
			result0 = nvmlDeviceGetUtilizationRates(device, &utilization);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get utilization rates: " << nvmlErrorString(result0);
				return false;
			}

			nvmlMemory_t memory;
			result0 = nvmlDeviceGetMemoryInfo(device, &memory);
			if (NVML_SUCCESS != result0)
			{
				std::cout << "Failed to get memory info: " << nvmlErrorString(result0);
				return false;
			}

			double total = (float)(memory.total) / 1024.0f / 1024.0f / 1024.0f;
			double isused = (float)(memory.free) / 1024.0f / 1024.0f / 1024.0f;

			LOGD("isused / total =>  {}G / {}G;", wikky_algo::d2str(isused, 3), total);

		}
	}
	catch (...)
	{
		LOGE("GPU error;");
		return false;
	}

	return true;
}




template <typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
	if (ptr) ptr->destroy();
}

bool readEngineFile(const std::string& engineFile, IRuntime*& runtime, ICudaEngine*& engine)
{
	std::ifstream file(engineFile, std::ios::binary);
	if (!file)
	{
		std::cerr << "Failed to open engine file: " << engineFile << std::endl;
		return false;
	}

	file.seekg(0, file.end);
	size_t size = file.tellg();
	file.seekg(0, file.beg);

	std::vector<char> buffer(size);
	file.read(buffer.data(), size);
	file.close();

	std::shared_ptr<IRuntime> runtime_ = nullptr;
	std::shared_ptr<ICudaEngine> engine_;

	static Logger gLogger;

	IBuilder* builder = createInferBuilder(gLogger);


	engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), size, nullptr),
		destroy_nvidia_pointer<ICudaEngine>);

	return true;

}



int yolo_min()
{


	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
	LOGD("OpencvBuildInformation --测试--: {}", cv::getBuildInformation());  //! 查看opencv编译时的选项
	
	bool temp = false;
	try {
		cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
		temp = true;
	}
	catch (...) {
		temp = false;
	}
	LOGD("IsOpenCVWithCUDA: {};", temp);
	LOGD("IsOpenCVWithDNN: {}:", (cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV).empty() ? "NO" : "YES"));
	outputGPU();


	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


	/** 超参数 */
	std::string onnxModelPath = "F:/Projects/CMakePractice/resources/3_2_best.onnx";
	std::string dnnOnnxModelPath = "F:/Projects/yolo_family/yolov5_7.0/pts/yolov5n.onnx";
	//std::string engineModelPath = "F:/Projects/CMakePractice/resources/3_2_best_fp16.plan";
	//std::string engineModelPath = "F:/Projects/CMakePractice/resources/3_2_best.trt";
	//std::string engineModelPath = "F:/Projects/StableDiffusionEO/engine/3_6_best.plan"; 
	std::string engineModelPath = "E://DevelopmentRoute//Produce_Algorithms//resources//yolov8s.engine";
	cv::Mat img = cv::imread("D:/58_FGJHAT005TZ000033G-1_DA3180921.png", 1);
	std::vector<cv::Mat> frames;
	int total_pics_num = 0;
	cv::Mat imgrst;


	LOGI("=============== ONNX Runtime =================");

	/** 模型初始化所需的参数 */
	Configuration onnx_config;
	onnx_config.modelpath = onnxModelPath;
	onnx_config.confThresholds = { 0.3, 0.3, 0.3 };      //? 需要修改成yaml参数
	onnx_config.nmsThreshold = 0.45;
	onnx_config.objThreshold = 0.5;

	/** 模型初始化 */
	//Onnx_YOLOv5* onnx_model;
	//onnx_model = new Onnx_YOLOv5(onnx_config);
	//std::vector<std::vector<BoxInfo>> onnx_output;
	///** 测试效率 */
	//frames.clear();
	//frames.push_back(img);
	//onnx_model->onnx_detect(frames, onnx_output);
	//start = std::chrono::high_resolution_clock::now();
	//for (int i = 0; i < total_pics_num; i++) {
	//	frames.clear();
	//	frames.push_back(img);
	//	onnx_model->onnx_detect(frames, onnx_output);
	//}
	//end = std::chrono::high_resolution_clock::now();
	//duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//LOGD("onnx_output: {};", onnx_output[0].size());
	//LOGD("{} pics time: {}; single pic time: {};\n", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	//
	//imgrst = frames[0].clone();
	//onnx_model->draw(imgrst, onnx_output);
	//cv::resize(imgrst, imgrst, cv::Size(640, 640));
	//cv::imshow("Inference", imgrst);
	//cv::waitKey(0);








	LOGI("=============== Opencv DNN =================");
	Configuration dnn_config;
	dnn_config.modelpath = dnnOnnxModelPath;
	dnn_config.confThresholds = { 0.3, 0.3, 0.3 };      //? 需要修改成yaml参数
	dnn_config.nmsThreshold = 0.45;
	dnn_config.objThreshold = 0.5;

	Dnn_YOLOv5* dnn_model;
	dnn_model = new Dnn_YOLOv5(dnn_config);
	std::vector<std::vector<BoxInfo>> dnn_output;

	cv::dnn::Net dnn_net;
	int _R = dnn_model->load_dnn_net(dnn_net, dnn_config.modelpath, true);
	if (_R == 1) {
		LOGI("DNN Model has been Successfilly loaded.");
		frames.clear();
		frames.push_back(img);
		dnn_model->dnn_detect(frames, dnn_net, dnn_output);

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			frames.clear();
			frames.push_back(img);
			dnn_model->dnn_detect(frames, dnn_net, dnn_output);
		}
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGD("dnn_output: {};", dnn_output[0].size());
		LOGD("{} pics time: {}; single pic time: {};\n", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));
	}








	LOGI("=============== TensorRT =================");
	Tensorrt_YOLOv5* tensorrt_model;
	tensorrt_model = new Tensorrt_YOLOv5();

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1; i++) {
		tensorrt_model->tensorrt_detect2(engineModelPath);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	LOGD("dnn_output: {};", dnn_output[0].size());
	LOGD("{} pics time: {}; single pic time: {};\n", total_pics_num, duration.count(), duration.count() / (float)(total_pics_num));



	return 1;
}


#include "tensorrtrunner.h"

TensorRTRunner::TensorRTRunner()
{

}

TensorRTRunner::~TensorRTRunner()
{

}

TensorRTRunner::TensorRTRunner(const RunnerParams& params) :
	_params(params)
{

}

TensorRTRunner::TensorRTRunner(TensorRTRunner&& temp) noexcept
{
	_params = std::move(temp._params);
	_engine = std::move(temp._engine);
	_runtime = std::move(temp._runtime);

	temp._engine.reset();
	temp._runtime.reset();
}

TensorRTRunner& TensorRTRunner::operator=(TensorRTRunner&& temp) noexcept
{
	_params = std::move(temp._params);
	_engine = std::move(temp._engine);
	_runtime = std::move(temp._runtime);

	temp._engine.reset();
	temp._runtime.reset();

	return *this;
}

bool TensorRTRunner::build(const std::function<bool(IOptimizationProfile*)>& optimization)
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	std::vector<char> plan_data;
	bool load_engine_success = false;
	if (load_engine(plan_data))
	{
		_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
		if (!_runtime)
		{
			load_engine_success = false;
		}

		_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
			_runtime->deserializeCudaEngine(plan_data.data(), plan_data.size()), samplesCommon::InferDeleter());
		if (!_engine)
		{
			load_engine_success = false;
		}

		if (_runtime && _engine)
		{
			sample::gLogInfo << "Loaded model from .engine file." << std::endl;
			load_engine_success = true;
		}
	}

	if (!load_engine_success)
	{
		auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
		if (!network)
		{
			return false;
		}

		auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!config)
		{
			return false;
		}

		auto parser
			= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
		if (!parser)
		{
			return false;
		}

		if (optimization)
		{
			nvinfer1::IOptimizationProfile* optimization_profile = builder->createOptimizationProfile();
			bool ret = optimization(optimization_profile);
			if (!ret)
			{
				return false;
			}
			config->addOptimizationProfile(optimization_profile);
		}

		auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();

		auto constructed = construct_network(builder, network, config, parser, timingCache);
		if (!constructed)
		{
			return false;
		}

		// CUDA stream used for profiling by the builder.
		auto profileStream = samplesCommon::makeCudaStream();
		if (!profileStream)
		{
			return false;
		}
		config->setProfileStream(*profileStream);

		SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
		if (!plan)
		{
			return false;
		}

		if (save_engine_to_plan_file(plan.get()))
		{
			sample::gLogInfo << "Saved engine as plan file" << std::endl;
		}

		if (timingCache != nullptr && !_params.timingCacheFile.empty())
		{
			samplesCommon::updateTimingCacheFile(
				sample::gLogger.getTRTLogger(), _params.timingCacheFile, timingCache.get(), *builder);
		}

		_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
		if (!_runtime)
		{
			return false;
		}

		_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
			_runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
		if (!_engine)
		{
			return false;
		}
	}

	uint32_t nireq = _params.nContexts;
	for (uint32_t i = 0; i < nireq; ++i)
	{
		_execution_contexts.emplace_back(_engine->createExecutionContext());
		_idle_contexts.push(_execution_contexts[i].get());
	}

	return true;
}

void TensorRTRunner::get(std::vector<Tensor<float>>& result)
{
	std::unique_lock<std::mutex> inference_results_lock(_inference_results_mut);
	_inference_results_cond.wait(inference_results_lock, [&]() { return !_inference_results.empty(); });
	result = _inference_results.front();
	_inference_results.pop();
}

std::vector<int> TensorRTRunner::get_input_shape(uint32_t index) const
{
	std::vector<int> vect{};

	nvinfer1::Dims dims = _engine->getTensorShape(get_input_tensor_name(index).c_str());
	for (int i = 0; i < dims.nbDims; ++i)
	{
		vect.push_back(dims.d[i]);
	}
	return vect;
}

std::vector<int> TensorRTRunner::get_output_shape(uint32_t index) const
{
	std::vector<int> vect{};

	nvinfer1::Dims dims = _engine->getTensorShape(get_output_tensor_name(index).c_str());
	for (int i = 0; i < dims.nbDims; ++i)
	{
		vect.push_back(dims.d[i]);
	}
	return vect;
}

nvinfer1::DataType TensorRTRunner::get_tensor_data_type(const std::string& name) const
{
	nvinfer1::DataType data_type = _engine->getTensorDataType(name.c_str());
	return data_type;
}

std::string TensorRTRunner::get_input_tensor_name(uint32_t index) const
{
	std::vector<std::string> input_names;

	int numIOTensors = _engine->getNbIOTensors();

	for (int i = 0; i < numIOTensors; ++i) {
		const char* tensorName = _engine->getIOTensorName(i);

		if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
			input_names.push_back(tensorName);
		}
	}

	return input_names[index];
}

std::string TensorRTRunner::get_output_tensor_name(uint32_t index) const
{
	std::vector<std::string> output_names;

	int numIOTensors = _engine->getNbIOTensors();

	for (int i = 0; i < numIOTensors; ++i) {
		const char* tensorName = _engine->getIOTensorName(i);

		if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT) {
			output_names.push_back(tensorName);
		}
	}

	return output_names[index];
}

size_t TensorRTRunner::get_number_of_inputs() const
{
	// Step 1: Get the number of I/O tensors
	int numIOTensors = _engine->getNbIOTensors();
	int numInputLayers = 0;

	// Step 2: Loop through all tensors and count the inputs
	for (int i = 0; i < numIOTensors; ++i) {
		// Get the tensor name at index i
		const char* tensorName = _engine->getIOTensorName(i);

		// Step 3: Check if this tensor is an input
		if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
			numInputLayers++;
		}
	}

	return numInputLayers;
}

size_t TensorRTRunner::get_number_of_outputs() const
{
	// Step 1: Get the number of I/O tensors
	int numIOTensors = _engine->getNbIOTensors();
	int numOutputLayers = 0;

	// Step 2: Loop through all tensors and count the outputs
	for (int i = 0; i < numIOTensors; ++i) {
		// Get the tensor name at index i
		const char* tensorName = _engine->getIOTensorName(i);

		// Step 3: Check if this tensor is an output
		if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT) {
			numOutputLayers++;
		}
	}

	return numOutputLayers;
}

bool TensorRTRunner::construct_network(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
{
	auto parsed = parser->parseFromFile(locateFile(_params.onnxFileName, _params.dataDirs).c_str(),
		static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}

	if (_params.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (_params.bf16)
	{
		config->setFlag(BuilderFlag::kBF16);
	}
	if (_params.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
	}
	if (_params.timingCacheFile.size())
	{
		timingCache = samplesCommon::buildTimingCacheFromFile(
			sample::gLogger.getTRTLogger(), *config, _params.timingCacheFile, sample::gLogError);
	}

	samplesCommon::enableDLA(builder.get(), config.get(), _params.dlaCore);

	return true;
}

bool TensorRTRunner::verify_output(const samplesCommon::BufferManager& buffers, const nvinfer1::IExecutionContext* context)
{
	auto n_outputs = get_number_of_outputs();
	std::vector<Tensor<float>> output_vect;
	for (uint32_t i = 0; i < n_outputs; ++i)
	{
		std::string tensor_name = get_output_tensor_name(i);
		size_t output_size = 1;
		std::vector<int> output_dims = dims_to_vector(context->getTensorShape(tensor_name.c_str()));
		for (int i = 0; i < output_dims.size(); ++i)
		{
			assert(output_dims[i] != -1);
			output_size *= output_dims[i];
		}

		float* output = static_cast<float*>(buffers.getHostBuffer(get_output_tensor_name(i)));

		Tensor<float> out;
		out.name = tensor_name;
		out.dimensions = std::move(output_dims);
		out.data = std::vector<float>(output, output + output_size);

		output_vect.push_back(std::move(out));
	}
	std::lock_guard<std::mutex> inference_results_lock(_inference_results_mut);
	_inference_results.push(std::move(output_vect));
	_inference_results_cond.notify_one();
	return true;
}

bool TensorRTRunner::load_engine(std::vector<char>& data) {
	std::istringstream stream(_params.onnxFileName);
	std::string file_name;
	std::getline(stream, file_name, '.');
	std::string engineFilePath = file_name + ".engine";
	std::ifstream file(engineFilePath, std::ios::binary);
	if (!file) {
		//throw std::runtime_error("Could not open engine file: " + engineFilePath);
		sample::gLogError << "Could not open engine file: " + engineFilePath << std::endl;
		sample::gLogInfo << "Will attempt to create engine file from .onnx model." << std::endl;
		return false;
	}

	file.seekg(0, std::ifstream::end);
	size_t fileSize = file.tellg();
	file.seekg(0, std::ifstream::beg);

	std::vector<char> engineData(fileSize);
	file.read(engineData.data(), fileSize);
	file.close();

	data = engineData;
	return true;
}

bool TensorRTRunner::save_engine_to_plan_file(nvinfer1::IHostMemory* plan)
{
	std::istringstream stream(_params.onnxFileName);
	std::string file_name;
	std::getline(stream, file_name, '.');
	std::string planFilePath = file_name + ".engine";
	std::ofstream planFile(planFilePath, std::ios::binary);
	planFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
	planFile.close();
	return true;
}

nvinfer1::Dims TensorRTRunner::vector_to_dims(const std::vector<int>& vec) const
{
	assert(vec.size() <= 8);
	nvinfer1::Dims dims;
	dims.nbDims = vec.size();
	for (int i = 0; i < dims.nbDims; ++i)
	{
		dims.d[i] = vec[i];
	}

	return dims;
}

std::vector<int> TensorRTRunner::dims_to_vector(const nvinfer1::Dims& dims) const
{
	std::vector<int> vec;
	vec.reserve(8);
	for (int i = 0; i < dims.nbDims; ++i)
	{
		vec.push_back(dims.d[i]);
	}
	return vec;
}

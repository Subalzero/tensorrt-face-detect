#include "tensorrtrunner.h"

TensorRTRunner::TensorRTRunner()
{

}

TensorRTRunner::~TensorRTRunner()
{

}

TensorRTRunner::TensorRTRunner(const RunnerParams& params):
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

bool TensorRTRunner::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	_network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
	if (!_network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser
		= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*_network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();

	auto constructed = construct_network(builder, _network, config, parser, timingCache);
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

	SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*_network, *config) };
	if (!plan)
	{
		return false;
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

	uint32_t nireq = 2;
	for (uint32_t i = 0; i < nireq; ++i)
	{
		_execution_contexts.emplace_back(_engine->createExecutionContext());
		_idle_contexts.push(_execution_contexts[i].get());
	}

	return true;
}

void TensorRTRunner::process(const std::vector<float>& input)
{
	// Create RAII buffer manager object
	samplesCommon::BufferManager buffers(_engine);

	std::unique_lock<std::mutex> idle_context_lock(_idle_contexts_mut);
	_idle_contexts_cond.wait(idle_context_lock, [&]() { return !_idle_contexts.empty(); });
	nvinfer1::IExecutionContext* context = _idle_contexts.front();
	_idle_contexts.pop();
	idle_context_lock.unlock();

	for (int32_t i = 0, e = _engine->getNbIOTensors(); i < e; i++)
	{
		auto const name = _engine->getIOTensorName(i);
		context->setTensorAddress(name, buffers.getDeviceBuffer(name));
	}

	process_input(buffers, input);

	// Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	context->executeV2(buffers.getDeviceBindings().data());

	// Memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();
}

bool TensorRTRunner::construct_network(SampleUniquePtr<nvinfer1::IBuilder>& builder, std::shared_ptr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
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

bool TensorRTRunner::process_input(const samplesCommon::BufferManager& buffers, const std::vector<float>& input)
{
	nvinfer1::Dims input_dims = _network->getInput(0)->getDimensions();

	const int inputH = input_dims.d[2];
	const int inputW = input_dims.d[3];
	const int inputC = input_dims.d[1];

	const int nElems = inputH * inputW * inputC;

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(_network->getInput(0)->getName()));

	for (int i = 0; i < nElems; ++i)
	{
		hostDataBuffer[i] = input[i];
	}

	return true;
}


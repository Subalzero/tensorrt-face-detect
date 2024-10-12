#ifndef TENSORRT_RUNNER_H
#define TENSORRT_RUNNER_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vector>
#include <queue>
#include <concepts>
#include <utility>
#include <typeinfo>
#include <unordered_map>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

struct Tensor
{
    std::string name;
    std::vector<int> shape;
    std::vector<unsigned char> data;
    nvinfer1::DataType data_type;

    // Constructor
    Tensor()
        : data_type(DataType::kFLOAT) {}

    Tensor(const std::string& name, 
        const std::vector<int>& shape, 
        std::vector<unsigned char>&& data, 
        const nvinfer1::DataType& data_type)
        : name(name), shape(shape), data(std::move(data)), data_type(data_type) {}

    Tensor(const std::string& name, 
        const std::vector<int>& shape, 
        const std::vector<unsigned char>& data, 
        const nvinfer1::DataType& data_type)
        : name(name), shape(shape), data(data), data_type(data_type) {}
};

struct RunnerParams
{
    int32_t batchSize{ 1 };              //!< Number of inputs in a batch
    int32_t dlaCore{ -1 };               //!< Specify the DLA core to run network on.
    bool int8{ false };                  //!< Allow runnning the network in Int8 mode.
    bool fp16{ false };                  //!< Allow running the network in FP16 mode.
    bool bf16{ false };                  //!< Allow running the network in BF16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string timingCacheFile; //!< Path to timing cache file
    std::string onnxFileName;  //!< REQUIRED: Model file in ONNX format.
    uint32_t nContexts{ 1 };  //!< Number of contexts tensorrt will generate
    bool dynamic{ false };
};

class TensorRTRunner
{
public:
    TensorRTRunner();
    ~TensorRTRunner();

    TensorRTRunner(const RunnerParams& params);

    TensorRTRunner(const TensorRTRunner& copy) = delete;
    TensorRTRunner& operator=(const TensorRTRunner& copy) = delete;

    TensorRTRunner(TensorRTRunner&& temp) noexcept;
    TensorRTRunner& operator=(TensorRTRunner&& temp) noexcept;

    bool build(const std::function<bool(IOptimizationProfile*)>& optimization = {});

    void process(const std::vector<Tensor>& input_tensors);

    void process_async(const std::vector<Tensor>& input_tensors);

    void get(std::vector<Tensor>& result);

    std::vector<int> get_input_shape(uint32_t index) const;
    std::vector<int> get_output_shape(uint32_t index) const;
    nvinfer1::DataType get_tensor_data_type(const std::string& name) const;
    constexpr size_t get_tensor_data_size(const nvinfer1::DataType& data_type) const;
    std::string get_input_tensor_name(uint32_t index) const;
    std::string get_output_tensor_name(uint32_t index) const;
    size_t get_number_of_inputs() const;
    size_t get_number_of_outputs() const;

private:
    RunnerParams _params;

    std::shared_ptr<nvinfer1::IRuntime> _runtime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> _engine; //!< The TensorRT engine used to run the network

    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> _execution_contexts;
    std::queue<nvinfer1::IExecutionContext*> _idle_contexts;
    std::queue<nvinfer1::IExecutionContext*> _busy_contexts;
    std::queue<std::vector<Tensor>> _inference_results;

    std::mutex _idle_contexts_mut;
    std::mutex _busy_contexts_mut;
    std::mutex _inference_results_mut;

    std::mutex _engine_mut;

    std::condition_variable _idle_contexts_cond;
    std::condition_variable _busy_contexts_cond;
    std::condition_variable _inference_results_cond;

    bool construct_network(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool process_input(const samplesCommon::BufferManager& buffers, nvinfer1::IExecutionContext* context, const std::vector<Tensor>& tensors);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verify_output(const samplesCommon::BufferManager& buffers, const nvinfer1::IExecutionContext* context);

    //!
    //! \brief Load plan file
    //! 
    bool load_engine(std::vector<char>& data);

    //!
    //! \brief Save engine to .plan file
    //! 
    bool save_engine_to_plan_file(nvinfer1::IHostMemory* plan);

    nvinfer1::Dims vector_to_dims(const std::vector<int>& vec) const;

    std::vector<int> dims_to_vector(const nvinfer1::Dims& dims) const;
};

#endif // TENSORRT_RUNNER_H
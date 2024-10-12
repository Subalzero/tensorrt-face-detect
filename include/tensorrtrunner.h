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

inline constexpr int typename_to_code(const char* type_name)
{
    if (std::string_view(type_name) == "float")                 return 0;
    else if (std::string_view(type_name) == "char")             return 2;
    else if (std::string_view(type_name) == "int")              return 3;
    else if (std::string_view(type_name) == "bool")             return 4;
    else if (std::string_view(type_name) == "unsigned char")    return 5;
    else if (std::string_view(type_name) == "long long")        return 8;
    else if (std::string_view(type_name) == "short")            return 9;
    else                                                        return -1;
}

// Concept to check if a type is integral or float
template <typename T>
concept BasicType = std::integral<T> || std::floating_point<T>;

// Concept to check if a type is a std::vector
template<typename T>
concept Vector = requires(T a) {
    typename T::value_type;  // Checks that T has a nested value_type
    { a.begin() } -> std::same_as<typename T::iterator>;  // Checks for begin iterator
    { a.data() } -> std::convertible_to<void*>; // Checks for data() method
}&& BasicType<typename T::value_type>;

template <typename T>
concept TensorLike = requires(T a)
{
    typename T::data_type; // Checks that T has a nested data_type
    { a.name } -> std::convertible_to<std::string>; // Check if name member exists
    { a.dimensions } -> std::convertible_to<std::vector<int>>; // Check if dimensions member exists
    { std::remove_cvref_t<decltype(a.data)>{} } -> Vector; // Checks if data is Vector type
};

//! Tensor Object
template <BasicType T>
struct Tensor
{
    std::string name;
    std::vector<int> dimensions;
    std::vector<T> data;

    using data_type = T;

    // Constructor
    Tensor() {}

    Tensor(const std::string& name, const std::vector<int>& dimensions, std::vector<T>&& data)
        : name(name), dimensions(dimensions), data(std::move(data)) {}

    Tensor(const std::string& name, const std::vector<int>& dimensions, const std::vector<T>& data)
        : name(name), dimensions(dimensions), data(data) {}
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

    template <TensorLike... TensorType>
    void process(const TensorType&... input_tensors)
    {
        // Create RAII buffer manager object
        std::unique_lock<std::mutex> engine_lock(_engine_mut);
        samplesCommon::BufferManager buffers(_engine);
        engine_lock.unlock();

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

        process_input(buffers, context, input_tensors...);

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        std::unique_lock<std::mutex> busy_contexts_lock(_busy_contexts_mut);
        _busy_contexts.push(context);
        _busy_contexts_cond.notify_one();
        busy_contexts_lock.unlock();

        context->executeV2(buffers.getDeviceBindings().data());

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();

        verify_output(buffers, context);

        busy_contexts_lock.lock();
        _busy_contexts_cond.wait(busy_contexts_lock, [&]() { return !_busy_contexts.empty(); });
        _busy_contexts.pop();
        busy_contexts_lock.unlock();

        idle_context_lock.lock();
        _idle_contexts.push(context);
        _idle_contexts_cond.notify_one();
        idle_context_lock.unlock();
    }

    template <TensorLike... TensorType>
    void process_async(const TensorType&... input_tensors)
    {
        std::thread(&TensorRTRunner::process<TensorType...>, this, input_tensors...).detach();
    }

    void get(std::vector<Tensor<float>>& result);

    std::vector<int> get_input_shape(uint32_t index) const;
    std::vector<int> get_output_shape(uint32_t index) const;
    nvinfer1::DataType get_tensor_data_type(const std::string& name) const;
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
    std::queue<std::vector<Tensor<float>>> _inference_results;

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
    template <TensorLike... TensorType>
    bool process_input(const samplesCommon::BufferManager& buffers, nvinfer1::IExecutionContext* context, const TensorType&... tensors)
    {
        auto n_inputs = get_number_of_inputs();
        std::size_t count = sizeof...(tensors);
        assert(count == n_inputs);
        int32_t i = 0;
        ([&]
            {
                std::string tensor_name = tensors.name;
                context->setInputShape(tensor_name.c_str(), vector_to_dims(tensors.dimensions));

                int32_t type_code = static_cast<int32_t>(get_tensor_data_type(tensor_name));
                const char* type_name = typeid(typename TensorType::data_type).name();

                assert(type_code == typename_to_code(type_name));

                size_t input_size = 1;
                std::vector<int> input_dims = dims_to_vector(context->getTensorShape(tensor_name.c_str()));
                for (int i = 0; i < input_dims.size(); ++i)
                {
                    assert(input_dims[i] != -1);
                    input_size *= input_dims[i];
                }

                auto hostDataBuffer = buffers.getHostBuffer(get_input_tensor_name(i));

                memcpy(hostDataBuffer, tensors.data.data(), input_size * sizeof(typename TensorType::data_type));

                ++i;
            } (), ...);

        return true;
    }

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
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

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

// Concept to check if a type is a std::vector
template<typename T>
concept Vector = requires(T a) {
    typename T::value_type;  // Checks that T has a nested value_type
    { a.begin() } -> std::same_as<typename T::iterator>;  // Checks for begin iterator
    { a.data() } -> std::convertible_to<void*>; // Checks for data() method
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

    bool build();

    template <Vector... Vecs>
    void process(const Vecs&... vecs)
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

        process_input(buffers, vecs...);

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        std::unique_lock<std::mutex> busy_contexts_lock(_busy_contexts_mut);
        _busy_contexts.push(context);
        _busy_contexts_cond.notify_one();
        busy_contexts_lock.unlock();

        context->executeV2(buffers.getDeviceBindings().data());

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();

        verify_output(buffers);

        busy_contexts_lock.lock();
        _busy_contexts_cond.wait(busy_contexts_lock, [&]() { return !_busy_contexts.empty(); });
        _busy_contexts.pop();
        busy_contexts_lock.unlock();

        idle_context_lock.lock();
        _idle_contexts.push(context);
        _idle_contexts_cond.notify_one();
        idle_context_lock.unlock();
    }

    template <Vector... Vecs>
    void process_async(const Vecs&... vecs)
    {
        std::thread(&TensorRTRunner::process<Vecs...>, this, vecs...).detach();
    }

    void get(std::vector<std::vector<float>>& result);

    std::vector<size_t> get_input_shape(uint32_t index) const;
    std::vector<size_t> get_output_shape(uint32_t index) const;
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
    std::queue<std::vector<std::vector<float>>> _inference_results;

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
    template <Vector... Vecs>
    bool process_input(const samplesCommon::BufferManager& buffers, const Vecs&... vecs)
    {
        auto n_inputs = get_number_of_inputs();
        std::size_t count = sizeof...(vecs);
        assert(count == n_inputs);
        int32_t i = 0;
        ([&]
            {
                size_t input_size = 0;
                std::vector<size_t> input_dims = get_input_shape(i);
                input_size = input_dims[0];
                for (int i = 1; i < input_dims.size(); ++i)
                {
                    input_size *= input_dims[i];
                }

                auto hostDataBuffer = buffers.getHostBuffer(get_input_tensor_name(i));

                memcpy(hostDataBuffer, vecs.data(), input_size * sizeof(typename Vecs::value_type));

                ++i;
            } (), ...);

        return true;
    }

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verify_output(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Load plan file
    //! 
    bool load_engine(std::vector<char>& data);

    //!
    //! \brief Save engine to .plan file
    //! 
    bool save_engine_to_plan_file(nvinfer1::IHostMemory* plan);
};

#endif // TENSORRT_RUNNER_H
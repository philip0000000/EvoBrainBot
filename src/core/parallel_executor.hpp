#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace evobrain::detail {

// Returns a usable logical-processor count even when the platform reports none.
inline std::size_t available_execution_threads() noexcept
{
    const unsigned int reported = std::thread::hardware_concurrency();
    return reported == 0 ? 1 : static_cast<std::size_t>(reported);
}

// Reuses a fixed standard-library worker set across simulation and brain batches.
class ParallelExecutor {
public:
    ParallelExecutor()
    {
        const std::size_t worker_count = available_execution_threads() - 1;
        workers_.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            workers_.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~ParallelExecutor()
    {
        {
            std::lock_guard lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_available_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    // Runs independent indices with the caller participating as one worker.
    template <typename Function>
    void for_each_index(const std::size_t count, const std::size_t thread_count,
        Function&& function)
    {
        if (count == 0) return;
        const std::size_t participating_workers =
            std::min({thread_count > 0 ? thread_count - 1 : 0, workers_.size(), count - 1});
        if (participating_workers == 0) {
            for (std::size_t index = 0; index < count; ++index) function(index);
            return;
        }

        // Jobs are serialized because the executor owns one shared callable and index range.
        std::lock_guard invocation_lock(invocation_mutex_);
        {
            std::lock_guard state_lock(state_mutex_);
            task_ = std::forward<Function>(function);
            task_count_ = count;
            next_index_.store(0, std::memory_order_relaxed);
            active_worker_count_ = participating_workers;
            unfinished_workers_ = participating_workers;
            first_exception_ = nullptr;
            ++generation_;
        }
        work_available_.notify_all();
        run_available_indices();

        std::unique_lock state_lock(state_mutex_);
        work_completed_.wait(state_lock, [this] { return unfinished_workers_ == 0; });
        const std::exception_ptr failure = first_exception_;
        task_ = {};
        state_lock.unlock();
        if (failure) std::rethrow_exception(failure);
    }

private:
    // Claims remaining indices atomically so each item is evaluated exactly once.
    void run_available_indices()
    {
        for (;;) {
            const std::size_t index = next_index_.fetch_add(1, std::memory_order_relaxed);
            if (index >= task_count_) return;
            try {
                task_(index);
            } catch (...) {
                std::lock_guard lock(state_mutex_);
                if (!first_exception_) first_exception_ = std::current_exception();
            }
        }
    }

    // Waits for new job generations and participates only when selected.
    void worker_loop(const std::size_t worker)
    {
        std::size_t observed_generation = 0;
        for (;;) {
            std::unique_lock lock(state_mutex_);
            work_available_.wait(lock, [this, &observed_generation] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            const bool participates = worker < active_worker_count_;
            lock.unlock();
            if (!participates) continue;

            run_available_indices();
            lock.lock();
            --unfinished_workers_;
            if (unfinished_workers_ == 0) work_completed_.notify_one();
        }
    }

    std::mutex invocation_mutex_;
    std::mutex state_mutex_;
    std::condition_variable work_available_;
    std::condition_variable work_completed_;
    std::vector<std::thread> workers_;
    std::function<void(std::size_t)> task_;
    std::atomic<std::size_t> next_index_ {0};
    std::size_t task_count_ = 0;
    std::size_t active_worker_count_ = 0;
    std::size_t unfinished_workers_ = 0;
    std::size_t generation_ = 0;
    std::exception_ptr first_exception_;
    bool stopping_ = false;
};

// Returns the process-wide executor so repeated batches reuse worker threads.
inline ParallelExecutor& parallel_executor()
{
    static ParallelExecutor executor;
    return executor;
}

} // namespace evobrain::detail

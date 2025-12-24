#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <chrono>

namespace perturblab {
namespace kernel {

/**
 * @brief Simple progress tracker for multi-threaded operations.
 * 
 * Uses a raw pointer to an array of size_t (one per thread) for lock-free updates.
 */
class ProgressTracker {
public:
    ProgressTracker(size_t nthreads)
        : nthreads_(nthreads), progress_(nthreads, 0) {
        raw_ptr_ = progress_.data();
    }

    size_t* ptr() { return raw_ptr_; }

    size_t aggregate() const {
        size_t sum = 0;
        for (size_t i = 0; i < nthreads_; ++i) {
            sum += progress_[i];
        }
        return sum;
    }

    size_t nthreads() const { return nthreads_; }

private:
    size_t nthreads_;
    std::vector<size_t> progress_;
    size_t* raw_ptr_;
};

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    #ifndef STDOUT_FILENO
    #define STDOUT_FILENO _fileno(stdout)
    #endif
#elif defined(__APPLE__) || defined(__linux__)
    #include <sys/ioctl.h>
    #include <unistd.h>
#endif

/**
 * @brief Helper to get the current terminal width.
 */
static size_t get_terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return static_cast<size_t>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
    }
#elif defined(__APPLE__) || defined(__linux__)
    #ifdef TIOCGWINSZ
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
            return static_cast<size_t>(w.ws_col);
        }
    #endif
#endif
    return 80;
}

/**
 * @brief Calculate appropriate bar width based on terminal size and text length.
 */
static size_t calculate_bar_width(const std::string& desc, const std::string& unit) {
    size_t term_width = get_terminal_width();
    size_t fixed_length = desc.length() + (desc.empty() ? 0 : 1) + 
                         2 + 1 + 3 + 2 + 12 + 6 + unit.length() + 2;
    
    if (term_width <= fixed_length + 10) {
        return 10;
    }
    return term_width - fixed_length;
}

/**
 * @brief A professional CLI progress bar, inspired by tqdm.
 */
class ProgressBar {
public:
    ProgressBar(size_t total,
              const std::string& desc = "",
              size_t width = 0,
              const std::string& unit = "it",
              bool use_unicode = true)
        : total_(std::max<size_t>(1, total)),
          desc_(desc),
          width_(width == 0 ? std::max<size_t>(10, calculate_bar_width(desc, unit) - 10) : std::max<size_t>(1, width)),
          unit_(unit),
          use_unicode_(use_unicode),
          start_time_(std::chrono::steady_clock::time_point::min()) {}

    void update(size_t current) {
        if (current > total_) current = total_;
        auto now = std::chrono::steady_clock::now();
        if (start_time_ == std::chrono::steady_clock::time_point::min()) {
            start_time_ = now;
        }
        
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        if (elapsed < 1e-9) elapsed = 1e-9;

        const double progress = static_cast<double>(current) / static_cast<double>(total_);
        const double exact_blocks = progress * static_cast<double>(width_);
        size_t full_blocks = static_cast<size_t>(exact_blocks);
        double frac = std::clamp(exact_blocks - static_cast<double>(full_blocks), 0.0, 1.0);
        if (full_blocks > width_) { full_blocks = width_; frac = 0.0; }

        std::string bar;
        bar.reserve(width_ * (use_unicode_ ? 3 : 1));
        size_t cells = 0;
        const char* full_sym = use_unicode_ ? "█" : "#";
        for (size_t i = 0; i < full_blocks && cells < width_; ++i) {
            bar += full_sym; ++cells;
        }
        if (cells < width_ && frac > 0.0) {
            bar += use_unicode_ ? partial_block(frac) : ">";
            ++cells;
        }
        while (cells < width_) { bar += " "; ++cells; }

        const double rate = static_cast<double>(current) / elapsed;
        std::string remain_text;
        if (current < total_ && rate > 1e-12) {
            const double remain = (static_cast<double>(total_ - current) / rate);
            remain_text = fmt_time(remain);
        } else if (current < total_) {
            remain_text = "--:--";
        } else {
            remain_text = "00:00";
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            std::ostringstream oss;
            if (!desc_.empty()) oss << desc_ << " ";
            oss << "[" << bar << "] "
                << fmt_time(elapsed) << "<" << remain_text << ", "
                << std::fixed << std::setprecision(2)
                << rate << " " << unit_ << "/s";

#ifdef _WIN32
            std::cout << "\r";
            static size_t last_length = 0;
            std::string output = oss.str();
            if (output.length() < last_length) {
                std::cout << std::string(last_length, ' ') << "\r";
            }
            std::cout << output << std::flush;
            last_length = output.length();
#else
            std::cout << "\r\033[2K" << oss.str() << std::flush;
#endif
        }
    }

    inline void complete() {
        completed_ = true;
        if (start_time_ == std::chrono::steady_clock::time_point::min()) {
            start_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);
        }
        update(total_);
        std::cout << std::endl;
    }

    inline void reset_start_time() {
        start_time_ = std::chrono::steady_clock::time_point::min();
    }

    inline bool is_completed() const { return completed_; }

private:
    static std::string fmt_time(double sec) {
        if (sec < 0) sec = 0;
        long long t = static_cast<long long>(sec + 0.5);
        long long d = t / 86400; t %= 86400;
        long long h = t / 3600;  t %= 3600;
        long long m = t / 60;    long long s = t % 60;
        std::ostringstream os;
        os << std::setfill('0');
        if (d > 0) {
            os << d << "d " << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        } else if (h > 0) {
            os << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        } else {
            os << std::setw(2) << m << ":" << std::setw(2) << s;
        }
        return os.str();
    }

    static std::string partial_block(double frac) {
        if (frac >= 7.0/8.0) return "▉";
        if (frac >= 6.0/8.0) return "▊";
        if (frac >= 5.0/8.0) return "▋";
        if (frac >= 4.0/8.0) return "▌";
        if (frac >= 3.0/8.0) return "▍";
        if (frac >= 2.0/8.0) return "▎";
        if (frac >= 1.0/8.0) return "▏";
        return " ";
    }

    size_t total_, width_;
    std::string desc_, unit_;
    bool use_unicode_;
    std::chrono::steady_clock::time_point start_time_;
    static std::mutex mu_;
    bool completed_ = false;
};

/**
 * @brief Wrapper for managing multiple progress bar stages.
 */
class BarWrapper {
private:
    std::vector<ProgressBar> bars_;
    std::vector<size_t> stage_totals_;
    ProgressTracker tracker_;
    std::thread update_thread_;
    std::atomic<bool> stop_flag_{false};
    size_t current_stage_{0};
    size_t cumulative_total_{0};
    
public:
    BarWrapper(std::vector<ProgressBar> bars, std::vector<size_t> stage_totals, int nthreads) 
        : bars_(std::move(bars)), stage_totals_(std::move(stage_totals)), tracker_(nthreads) {
        if (bars_.size() != stage_totals_.size()) {
            throw std::invalid_argument("Number of bars must match number of stage totals");
        }
    }
    
    BarWrapper(std::vector<ProgressBar> bars, int nthreads) 
        : bars_(std::move(bars)), tracker_(nthreads) {
        stage_totals_.resize(bars_.size(), SIZE_MAX);
    }

    void start(size_t time_interval = 100) {
        stop_flag_ = false;
        current_stage_ = 0;
        cumulative_total_ = 0;
        
        update_thread_ = std::thread([this, time_interval]() {
            while (!stop_flag_ && current_stage_ < bars_.size()) {
                size_t sum = tracker_.aggregate();
                while (current_stage_ < bars_.size() && 
                       sum >= cumulative_total_ + stage_totals_[current_stage_]) {
                    bars_[current_stage_].complete();
                    cumulative_total_ += stage_totals_[current_stage_];
                    current_stage_++;
                    if (current_stage_ < bars_.size()) {
                        bars_[current_stage_].reset_start_time();
                    }
                }
                if (current_stage_ < bars_.size()) {
                    size_t current_progress = sum - cumulative_total_;
                    bars_[current_stage_].update(current_progress);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(time_interval));
            }
        });
    }
    
    void stop() {
        stop_flag_ = true;
        if (update_thread_.joinable()) {
            update_thread_.join();
        }
        for (size_t i = current_stage_; i < bars_.size(); ++i) {
            if (!bars_[i].is_completed()) {
                bars_[i].complete();
            }
        }
    }
    
    ~BarWrapper() {
        stop();
    }
    
    size_t* get_progress_ptr() {
        return tracker_.ptr();
    }
};

} // namespace kernel
} // namespace perturblab

// Per-call scratch overhead of parallel_reduce_sum on the CUDA path.
//
//   nvcc -std=c++23 -O3 -arch=native -DXPU_CUDA -I. bench/reduce_scratch.cu -o build-cuda/bench_reduce
//   ./build-cuda/bench_reduce
//
//   caller   : old API, scratch allocated once by the caller (baseline)
//   xpu      : xpu::parallel_reduce_sum, cub env overload -> cudaMallocAsync/cudaFreeAsync per call
//              (xpu raises the default pool's release threshold so syncs don't return memory to the OS)
//   malloc   : cudaMalloc/cudaFree per call (naive per-call allocation, for reference)
//
// "async" enqueues iters reductions then syncs once (throughput);
// "each" syncs after every reduction (latency; would trim a default-configured pool at each sync).

#include <xpu/buffer.hpp>
#include <xpu/launch.hpp>

#include <chrono>
#include <cstdio>

struct value {
  __host__ __device__
  auto operator()(const xpu::array<std::size_t, 1uz>& index) const -> float {
    return static_cast<float>(index[0] & 7uz);
  }
};

enum class mode { caller, xpu, malloc };

template <mode m>
auto run(std::size_t n, int iters, bool sync_each, float* out) -> double {
  const auto range{xpu::range<1uz>{{0uz}, {n}, {1uz}}};
  const auto input{xpu::detail::reduction_input<float>(range, value{})};
  const auto count{static_cast<std::ptrdiff_t>(n)};

  auto bytes{0uz};
  xpu::cu_check(cub::DeviceReduce::Sum(nullptr, bytes, input, out, count));
  auto* scratch{static_cast<void*>(nullptr)};
  xpu::cu_check(cudaMalloc(&scratch, bytes));

  const auto once{[&] {
    if constexpr (m == mode::caller) {
      xpu::cu_check(cub::DeviceReduce::Sum(scratch, bytes, input, out, count));
    } else if constexpr (m == mode::xpu) {
      xpu::parallel_reduce_sum(range, out, value{});
    } else {
      auto* temp{static_cast<void*>(nullptr)};
      auto temp_bytes{0uz};
      xpu::cu_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, out, count));
      xpu::cu_check(cudaMalloc(&temp, temp_bytes));
      xpu::cu_check(cub::DeviceReduce::Sum(temp, temp_bytes, input, out, count));
      xpu::cu_check(cudaFree(temp));
    }
    if (sync_each) { xpu::cu_check(cudaDeviceSynchronize()); }
  }};

  for (auto i{0}; i < 10; ++i) { once(); }
  xpu::cu_check(cudaDeviceSynchronize());

  const auto start{std::chrono::steady_clock::now()};
  for (auto i{0}; i < iters; ++i) { once(); }
  xpu::cu_check(cudaDeviceSynchronize());
  const auto stop{std::chrono::steady_clock::now()};

  xpu::cu_check(cudaFree(scratch));

  return std::chrono::duration<double, std::micro>(stop - start).count() / iters;
}

int main() {
  auto out{xpu::buffer<float>{1uz}};

  std::printf("%12s %6s %10s %10s %10s   (us/call)\n", "n", "sync", "caller", "xpu", "malloc");

  for (const auto n : {1uz << 10, 1uz << 16, 1uz << 20, 1uz << 24, 1uz << 28}) {
    const auto iters{n >= (1uz << 24) ? 200 : 2000};

    for (const auto sync_each : {false, true}) {
      const auto caller{run<mode::caller>(n, iters, sync_each, out.data())};
      const auto xpu{run<mode::xpu>(n, iters, sync_each, out.data())};
      const auto naive{run<mode::malloc>(n, iters, sync_each, out.data())};

      std::printf(
        "%12zu %6s %10.2f %10.2f %10.2f\n",
        n, sync_each ? "each" : "end", caller, xpu, naive
      );
    }
  }
}

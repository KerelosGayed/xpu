#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/math.hpp>

#if defined(XPU_CUDA)
  #include <cub/device/device_for.cuh>
  #include <cub/device/device_reduce.cuh>
  #include <cuda/std/array>
  #include <thrust/iterator/counting_iterator.h>
  #include <thrust/iterator/transform_iterator.h>
#else
  #include <array>
#endif

namespace xpu {

#if defined(XPU_CUDA) && defined(__CUDACC__)

namespace detail {

[[nodiscard]]
inline auto device_SMs() -> unsigned int {
  static const auto cached{[] {
      auto device{0}, sms{0};
      xpu::cu_check(cudaGetDevice(&device));
      xpu::cu_check(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, device
      ));
      const auto multiprocessors{xpu::detail::checked_cast<unsigned int>(sms)};

      return multiprocessors;
    }()
  };

  return cached;
}

// the default pool trims to 0 bytes at every sync, so stream-ordered scratch would
// go back to the OS after each synchronized call (~0.8ms per reduce); keep it cached
inline auto retain_default_pool() -> void {
  static const auto once{[] {
      auto device{0};
      auto pool{cudaMemPool_t{}};
      auto threshold{std::numeric_limits<std::uint64_t>::max()};
      xpu::cu_check(cudaGetDevice(&device));
      xpu::cu_check(cudaDeviceGetDefaultMemPool(&pool, device));
      xpu::cu_check(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &threshold
      ));

      return true;
    }()
  };

  static_cast<void>(once);
}

template <typename Kernel> [[nodiscard]]
inline auto wave_blocks(Kernel kernel, dim3 threads, std::size_t smem = 0uz) -> unsigned int {
  const auto thread_budget{xpu::detail::checked_mul(
    xpu::detail::checked_mul(threads.x, threads.y), threads.z
  )};
  auto blocks_per_SM{0};

  cu_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_SM, kernel, xpu::detail::checked_cast<int>(thread_budget), smem
  ));

  const auto blocks{xpu::detail::checked_mul(
    xpu::detail::device_SMs(),
    xpu::detail::checked_cast<unsigned int>(blocks_per_SM)
  )};

  return blocks;
}

[[nodiscard]]
inline constexpr auto num_blocks(std::size_t size, unsigned int threads) noexcept -> unsigned int {
  const auto zero_threads{threads == 0u};

  if (zero_threads) {
    xpu::detail::checked_error("zero launch threads");
  }

  const auto blocks{xpu::detail::checked_cast<unsigned int>(
    xpu::ceiling_div<std::size_t>(size, threads)
  )};

  return blocks;
}

template <typename Kernel> [[nodiscard]]
inline auto blocks_for(Kernel kernel, unsigned int threads, std::size_t size) -> unsigned int {
  const auto zero_threads{threads == 0u};

  if (zero_threads) {
    xpu::detail::checked_error("zero launch threads");
  }

  const auto needed_blocks{xpu::ceiling_div<std::size_t>(size, threads)};
  const auto available_blocks{xpu::detail::wave_blocks(kernel, threads)};
  const auto blocks{xpu::detail::checked_cast<unsigned int>(
    xpu::min(needed_blocks, static_cast<std::size_t>(available_blocks))
  )};

  return blocks;
}

} // namespace xpu::detail

[[nodiscard]] XPU_DEVICE_ONLY
inline auto linear_index() noexcept -> std::size_t {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] XPU_DEVICE_ONLY
inline auto linear_stride() noexcept -> std::size_t {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

template <int Dims> struct Coord;
template <> struct Coord<1> { std::size_t x{}; };
template <> struct Coord<2> { std::size_t x{}, y{}; };
template <> struct Coord<3> { std::size_t x{}, y{}, z{}; };

template <int Dims> __device__ [[nodiscard]]
inline auto global_index() noexcept -> Coord<Dims> {
  auto id{Coord<Dims>{}};
  
  if constexpr (Dims >= 1) {
    id.x = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  }
  if constexpr (Dims >= 2) {
    id.y = static_cast<std::size_t>(blockDim.y) * blockIdx.y + threadIdx.y;
  }
  if constexpr (Dims >= 3) {
    id.z = static_cast<std::size_t>(blockDim.z) * blockIdx.z + threadIdx.z;
  }

  return id;
}

inline constexpr auto block_per_dim(std::size_t size, unsigned int dim_threads) noexcept -> unsigned int {
  const auto blocks{xpu::detail::num_blocks(size, dim_threads)};

  return blocks;
}

template <int Dims> __device__ [[nodiscard]]
inline auto global_stride() noexcept -> Coord<Dims> {
  auto stride{Coord<Dims>{}};
  
  if constexpr (Dims >= 1) {
    stride.x = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  }
  if constexpr (Dims >= 2) {
    stride.y = static_cast<std::size_t>(blockDim.y) * gridDim.y;
  }
  if constexpr (Dims >= 3) {
    stride.z = static_cast<std::size_t>(blockDim.z) * gridDim.z;
  }

  return stride;
}

#endif

using xstd::array;

template <std::size_t dims>
struct [[nodiscard]] range {
  xpu::array<std::size_t, dims> begin;
  xpu::array<std::size_t, dims> end;
  xpu::array<std::size_t, dims> step;
};

namespace detail {

template <std::size_t dims>
inline constexpr auto num_itrs(
  const xpu::range<dims>& range
) noexcept -> std::size_t {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto total{1uz};

  for (auto d{0uz}; d < dims; ++d) {
    const auto empty_dimension{range.step[d] == 0uz || range.begin[d] >= range.end[d]};

    if (empty_dimension) {
      const auto empty_count{0uz};

      return empty_count;
    }
  }

  for (auto d{0uz}; d < dims; ++d) {
    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};
    total = xpu::detail::checked_mul(total, count);
  }

  return total;
}

template <std::size_t dims> XPU_CUDA_CALLABLE
inline constexpr auto itr_index(
  const xpu::range<dims>& range,
  std::size_t linear
) noexcept -> xpu::array<std::size_t, dims> {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto index{xpu::array<std::size_t, dims>{}};

  for (auto d{dims}; d-- > 1uz;) {
    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};
    const auto coordinate{linear % count};

    index[d] = range.begin[d] + coordinate * range.step[d];
    linear /= count;
  }

  index[0] = range.begin[0] + linear * range.step[0];

  return index;
}

template <typename F, typename T, std::size_t dims>
concept sum_contribution = requires(
  const std::remove_reference_t<F>& contribution,
  const xpu::array<std::size_t, dims>& index
) {
  { contribution(index) } -> std::same_as<T>;
};

template <std::size_t dims, arithmetic T, typename F>
struct reduction_contribution {
  xpu::range<dims> range;
  F contribution;

  [[nodiscard]] XPU_CUDA_CALLABLE
  auto operator()(std::size_t linear) const -> T {
    const auto index{xpu::detail::itr_index(range, linear)};
    const auto value{T{contribution(index)}};

    return value;
  }
};

#if defined(XPU_CUDA)
template <arithmetic T, std::size_t dims, typename F>
inline auto reduction_input(
  const xpu::range<dims>& range,
  F&& contribution
) {
  using function_t = std::decay_t<F>;

  const auto transform{reduction_contribution<dims, T, function_t>{
    range, function_t{std::forward<F>(contribution)}
  }};

  const auto indices{thrust::counting_iterator<std::size_t>{0uz}};
  const auto input{thrust::make_transform_iterator(indices, transform)};

  return input;
}

template <std::size_t dims, typename F>
struct parallel_for_function {
  xpu::range<dims> range;
  F fcn;

  XPU_DEVICE_ONLY
  auto operator()(std::size_t linear) -> void {
    const auto index{xpu::detail::itr_index(range, linear)};

    fcn(index);
  }
};
#endif

#if !defined(XPU_CUDA)
template <bool dense, std::size_t dimension, std::size_t dims>
inline auto parallel_for_coordinate(
  const xpu::range<dims>& range,
  std::size_t iteration
) -> std::size_t {
  if constexpr (dense) {
    return iteration;
  } else {
    return range.begin[dimension] + iteration * range.step[dimension];
  }
}

template <bool dense, std::size_t dims, typename F, typename... Coordinates>
XPU_FORCE_INLINE auto parallel_for_nested(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  F& fcn,
  Coordinates... outer
) -> void {
  constexpr auto dimension{sizeof...(Coordinates)};

  if constexpr (dimension == dims) {
    fcn(xpu::array<std::size_t, dims>{outer...});
  } else if constexpr (dimension + 1uz == dims) {
    #pragma omp simd
    for (auto iteration = 0uz; iteration < counts[dimension]; ++iteration) {
      parallel_for_nested<dense>(
        range, counts, fcn, outer..., parallel_for_coordinate<dense, dimension>(range, iteration)
      );
    }
  } else {
    for (auto iteration{0uz}; iteration < counts[dimension]; ++iteration) {
      parallel_for_nested<dense>(
        range, counts, fcn, outer..., parallel_for_coordinate<dense, dimension>(range, iteration)
      );
    }
  }
}

template <bool dense, std::size_t dims, typename F>
inline auto parallel_for_cpu(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims> counts,
  F& fcn
) -> void {
  if constexpr (!std::is_copy_constructible_v<F>) {
    auto shared{[&fcn](auto&& index) {
      fcn(std::forward<decltype(index)>(index));
    }};

    parallel_for_cpu<dense>(range, counts, shared);
  } else if constexpr (dims == 1uz) {
    #pragma omp parallel for simd firstprivate(counts, fcn)
    for (auto iteration = 0uz; iteration < counts[0]; ++iteration) {
      parallel_for_nested<dense>(
        range, counts, fcn, parallel_for_coordinate<dense, 0uz>(range, iteration)
      );
    }
  } else {
    #pragma omp parallel for firstprivate(counts, fcn)
    for (auto iteration = 0uz; iteration < counts[0]; ++iteration) {
      parallel_for_nested<dense>(
        range, counts, fcn, parallel_for_coordinate<dense, 0uz>(range, iteration)
      );
    }
  }
}
#endif

template <std::size_t dims, typename F>
inline auto parallel_for_impl(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
#if defined(XPU_CUDA)
  const auto total{xpu::detail::num_itrs(range)};
  if (total == 0uz) { return; }

  using function_t = std::decay_t<F>;

  const auto operation{parallel_for_function<dims, function_t>{
    range,
    function_t{std::forward<F>(fcn)}
  }};

  xpu::cu_check(cub::DeviceFor::Bulk(total, operation));
#else
  auto dense{true};
  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    const auto empty_dimension{range.step[dimension] == 0uz || range.begin[dimension] >= range.end[dimension]};
    if (empty_dimension) { return; }

    dense = dense && range.begin[dimension] == 0uz && range.step[dimension] == 1uz;
  }

  auto counts{range.end};
  auto total{1uz};

  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    if (!dense) {
      const auto delta{range.end[dimension] - range.begin[dimension]};
      counts[dimension] = xpu::ceiling_div(delta, range.step[dimension]);
    }

    total = xpu::detail::checked_mul(total, counts[dimension]);
  }

  if (dense) {
    parallel_for_cpu<true>(range, counts, fcn);
  } else {
    parallel_for_cpu<false>(range, counts, fcn);
  }
#endif
}

#if !defined(XPU_CUDA)
template <bool unit_step, std::size_t dims, arithmetic T, typename F, typename... Coordinates>
XPU_FORCE_INLINE auto parallel_reduce_nested(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  const F& contribution,
  T& total_sum,
  Coordinates... outer
) -> void {
  constexpr auto dimension{sizeof...(Coordinates)};

  if constexpr (dimension + 1uz == dims) {
    // GCC won't vectorize an omp simd body that constructs the index array itself
    const auto step{unit_step ? 1uz : range.step[dimension]};
    const auto contribution_at{[&](std::size_t iteration) {
      return contribution(xpu::array<std::size_t, dims>{
        outer..., range.begin[dimension] + iteration * step
      });
    }};

    #pragma omp simd reduction(+ : total_sum)
    for (auto iteration = 0uz; iteration < counts[dimension]; ++iteration) {
      total_sum += contribution_at(iteration);
    }
  } else {
    for (auto iteration{0uz}; iteration < counts[dimension]; ++iteration) {
      parallel_reduce_nested<unit_step>(
        range, counts, contribution, total_sum, outer...,
        parallel_for_coordinate<false, dimension>(range, iteration)
      );
    }
  }
}

template <bool unit_step, arithmetic T, std::size_t dims, typename F> [[nodiscard]]
inline auto parallel_reduce_cpu(
  const xpu::range<dims>& range,
  std::size_t total,
  const F& contribution
) -> T {
  constexpr auto grain{8192uz};
  auto counts{range.end};

  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    const auto delta{range.end[dimension] - range.begin[dimension]};
    counts[dimension] = xpu::ceiling_div(delta, range.step[dimension]);
  }

  // boxes of ~grain iterations: dims before split fixed, split blocked, dims after split whole
  auto split{0uz}, outer{1uz}, inner{total / counts[0]};
  for (; inner > grain; inner /= counts[++split]) {
    outer *= counts[split];
  }

  const auto block{xpu::min(counts[split], xpu::max(grain / inner, 1uz))};
  const auto blocks{xpu::ceiling_div(counts[split], block)};
  const auto boxes{outer * blocks};
  auto total_sum{T{}};

  #pragma omp parallel for reduction(+ : total_sum) firstprivate(contribution) if(boxes > 1uz)
  for (auto box = 0uz; box < boxes; ++box) {
    auto box_range{range};
    auto box_counts{counts};
    const auto first{box % blocks * block};

    box_range.begin[split] += first * range.step[split];
    box_counts[split] = xpu::min(block, counts[split] - first);

    for (auto dimension{split}, rest{box / blocks}; dimension-- > 0uz;) {
      box_range.begin[dimension] += rest % counts[dimension] * range.step[dimension];
      box_counts[dimension] = 1uz;
      rest /= counts[dimension];
    }

    parallel_reduce_nested<unit_step>(box_range, box_counts, contribution, total_sum);
  }

  return total_sum;
}
#endif

template <std::size_t dims, arithmetic T, typename F>
inline auto parallel_reduce_sum_impl(
  const xpu::range<dims>& range,
  std::size_t total,
  T* output_ptr,
  F&& contribution
) -> void {
#if defined(XPU_CUDA)
  const auto input{reduction_input<T>(range, std::forward<F>(contribution))};
  const auto count{xpu::detail::checked_cast<std::ptrdiff_t>(total)};

  // cub owns the scratch: cudaMallocAsync/cudaFreeAsync from the device's default pool
  xpu::detail::retain_default_pool();
  xpu::cu_check(cub::DeviceReduce::Sum(input, output_ptr, count));
#else
  if (range.step[dims - 1uz] == 1uz) {
    *output_ptr = parallel_reduce_cpu<true, T>(range, total, contribution);
  } else {
    *output_ptr = parallel_reduce_cpu<false, T>(range, total, contribution);
  }
#endif
}

} // namespace xpu::detail

template <std::size_t dims, typename F>
inline auto parallel_for(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  detail::parallel_for_impl(range, std::forward<F>(fcn));
}

template <std::size_t dims, arithmetic T, typename F>
  requires detail::sum_contribution<F, T, dims>
inline auto parallel_reduce_sum(
  const xpu::range<dims>& range,
  T* output_ptr,
  F&& contribution
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  const auto total{xpu::detail::num_itrs(range)};
  const auto empty{total == 0uz};

  if (empty) {
#if defined(XPU_CUDA)
    xpu::cu_check(cudaMemsetAsync(output_ptr, 0, sizeof(T)));
#else
    *output_ptr = T{};
#endif

    return;
  }

  detail::parallel_reduce_sum_impl(
    range, total, output_ptr, std::forward<F>(contribution)
  );
}

} // namespace xpu

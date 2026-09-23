#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/launch.hpp>
#include <xpu/memory.hpp>

#include <array>
#include <limits>

template <typename T>
struct sum_values {
  [[nodiscard]] XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>&) -> T {
    const auto value{T{-1}};

    return value;
  }

  [[nodiscard]] XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>& index) const -> T {
    const auto value{static_cast<T>(index[0] + index[1] % 7uz)};

    return value;
  }
};

struct nonconst_sum {
  XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>&) -> int;
};

template <typename T, typename F>
concept supports_sum = requires(const xpu::range<2uz>& range, T* output, F contribution) {
  xpu::parallel_reduce_sum(range, output, contribution);
};

static_assert(supports_sum<int, sum_values<int>>);
static_assert(supports_sum<float, sum_values<float>>);
static_assert(supports_sum<double, sum_values<double>>);
static_assert(!supports_sum<int, nonconst_sum>);
static_assert(!supports_sum<int, sum_values<double>>);
static_assert(!supports_sum<double, sum_values<float>>);

template <typename T>
inline auto run_sum_type() -> int {
  constexpr auto columns{std::array{0uz, 1uz, 3uz, 4097uz}};
  auto contribution{sum_values<T>{}};
  auto output{xpu::buffer<T>{1uz}};

  for (const auto count : columns) {
    const auto range = xpu::range<2uz>{
      {2uz, 1uz}, {6uz, 1uz + 3uz * count}, {2uz, 3uz}
    };
    auto expected{T{}};

    for (auto row{2uz}; row < 6uz; row += 2uz) {
      for (auto column{1uz}; column < 1uz + 3uz * count; column += 3uz) {
        expected += static_cast<T>(row + column % 7uz);
      }
    }

    const auto initial{T{91}};
    xpu::copy_n(output.data(), &initial, 1uz);

    for (auto repeat{0uz}; repeat < 2uz; ++repeat) {
      xpu::parallel_reduce_sum(range, output.data(), contribution);

      auto actual{T{}};
      xpu::copy_n(&actual, output.data(), 1uz);
      const auto incorrect_sum{actual != expected};

      if (incorrect_sum) {
        const auto failure{test::fail("reduction did not overwrite output with the expected sum")};

        return failure;
      }
    }
  }

  const auto success{0};

  return success;
}

struct sum_coordinates {
  template <std::size_t dims> [[nodiscard]] XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, dims>& index) const -> int {
    auto value{0};

    for (const auto coordinate : index) {
      value += static_cast<int>(coordinate);
    }

    return value;
  }
};

template <std::size_t dims>
inline auto check_coordinate_sum(const xpu::range<dims>& range, int expected) -> int {
  const auto contribution{sum_coordinates{}};
  auto output{xpu::buffer<int>{1uz}};

  xpu::parallel_reduce_sum(range, output.data(), contribution);

  auto actual{0};
  xpu::copy_n(&actual, output.data(), 1uz);
  const auto incorrect_sum{actual != expected};

  if (incorrect_sum) {
    const auto failure{test::fail("reduction visited incorrect range coordinates")};

    return failure;
  }

  const auto success{0};

  return success;
}

inline auto run_sum_cases() -> int {
  const xpu::range<1uz> line{{3uz}, {10uz}, {2uz}};
  const xpu::range<1uz> long_line{{0uz}, {8193uz}, {1uz}};
  const xpu::range<1uz> long_stepped_line{{1uz}, {16386uz}, {2uz}};
  const xpu::range<3uz> volume{{1uz, 2uz, 3uz}, {4uz, 7uz, 8uz}, {2uz, 3uz, 2uz}};
  const xpu::range<2uz> square{{0uz, 0uz}, {17uz, 31uz}, {1uz, 1uz}};
  const xpu::range<2uz> large_square{{0uz, 0uz}, {64uz, 64uz}, {1uz, 1uz}};
  const xpu::range<4uz> dense_four{
    {0uz, 0uz, 0uz, 0uz}, {2uz, 3uz, 5uz, 7uz}, {1uz, 1uz, 1uz, 1uz}
  };
  const xpu::range<4uz> stepped_four{
    {1uz, 2uz, 3uz, 4uz}, {5uz, 11uz, 9uz, 13uz}, {2uz, 3uz, 3uz, 3uz}
  };
  const xpu::range<2uz> thin_square{{0uz, 0uz}, {1uz, 20000uz}, {1uz, 1uz}};
  const xpu::range<3uz> thin_volume{{0uz, 0uz, 0uz}, {3uz, 5uz, 9000uz}, {1uz, 1uz, 1uz}};
  const xpu::range<3uz> thin_stepped_volume{{1uz, 0uz, 3uz}, {3uz, 2uz, 30000uz}, {1uz, 1uz, 3uz}};
  const xpu::range<3uz> middle_split{{0uz, 0uz, 0uz}, {4uz, 300uz, 40uz}, {1uz, 1uz, 1uz}};
  const xpu::range<3uz> empty_later{
    {0uz, 0uz, 0uz},
    {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), 0uz},
    {1uz, 1uz, 1uz}
  };

  if (const auto failure{check_coordinate_sum(line, 24)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(volume, 126)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(long_line, 33558528)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(long_stepped_line, 67125249)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(square, 12121)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(large_square, 258048)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(dense_four, 1365)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(stepped_four, 666)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(thin_square, 199990000)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(thin_volume, 607837500)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(thin_stepped_volume, 600019992)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(middle_split, 8184000)}; failure) {

    return failure;
  }

  if (const auto failure{check_coordinate_sum(empty_later, 0)}; failure) {

    return failure;
  }

  if (const auto failure{run_sum_type<int>()}; failure) {

    return failure;
  }

  if (const auto failure{run_sum_type<float>()}; failure) {

    return failure;
  }

  const auto result{run_sum_type<double>()};

  return result;
}

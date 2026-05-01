#pragma once
// page_size_dispatch.h — Compile-time page size dispatch utility.
//
// Converts a runtime page_size value into a compile-time template parameter.
// Supports: 64K, 128K, 256K, 512K, 1M, 2M.
//
// Usage:
//   dispatch_page_size(p.page_size, [&](auto ps_tag) {
//       constexpr unsigned PS = decltype(ps_tag)::value;
//       auto kernel_fn = my_kernel<PS>;
//       kernel_fn<<<grid, block, smem, stream>>>(args...);
//   });

#include <cstdint>
#include <type_traits>

template<typename Fn>
inline void dispatch_page_size(uint32_t page_size, Fn&& fn) {
    switch (page_size) {
        case 65536:   fn(std::integral_constant<unsigned int, 65536>{}); break;
        case 131072:  fn(std::integral_constant<unsigned int, 131072>{}); break;
        case 262144:  fn(std::integral_constant<unsigned int, 262144>{}); break;
        case 524288:  fn(std::integral_constant<unsigned int, 524288>{}); break;
        case 1048576: fn(std::integral_constant<unsigned int, 1048576>{}); break;
        default:      fn(std::integral_constant<unsigned int, 2097152>{}); break;
    }
}

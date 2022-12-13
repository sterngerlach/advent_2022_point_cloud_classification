
// data_conversion.hpp

#ifndef DATA_CONVERSION_HPP
#define DATA_CONVERSION_HPP

#include <cstdint>

#include <ap_int.h>

union conv32_t
{
  std::uint32_t u32;
  int i32;
  float f;
};

// Interpret float as std::uint32_t
inline std::uint32_t FloatToU32(const float f)
{
  conv32_t conv;
  conv.f = f;
  return conv.u32;
}

// Interpret std::uint32_t as float
inline float U32ToFloat(const std::uint32_t u32)
{
  conv32_t conv;
  conv.u32 = u32;
  return conv.f;
}

// Interpret int as std::uint32_t
inline std::uint32_t IntToU32(const int i32)
{
  conv32_t conv;
  conv.i32 = i32;
  return conv.u32;
}

// Interpret std::uint32_t as int
inline int U32ToInt(const std::uint32_t u32)
{
  conv32_t conv;
  conv.u32 = u32;
  return conv.i32;
}

// Interpret float as int
inline int FloatToInt(const float f)
{
  conv32_t conv;
  conv.f = f;
  return conv.i32;
}

// Interpret int as float
inline float IntToFloat(const int i32)
{
  conv32_t conv;
  conv.i32 = i32;
  return conv.f;
}

// Interpret ap_fixed<TBits, I> as std::uint32_t
template <typename T, int TBits>
inline std::uint32_t FixedToU32(const T f)
{
  ap_uint<32> x = 0;
  x.range(TBits - 1, 0) = f.range(TBits - 1, 0);
  return x.to_uint();
}

// Interpret std::uint32_t as ap_fixed<TBits, I>
template <typename T, int TBits>
inline T U32ToFixed(const std::uint32_t u)
{
  ap_uint<32> x = u;
  T f = 0;
  f.range(TBits - 1, 0) = x.range(TBits - 1, 0);
  return f;
}

#endif // DATA_CONVERSION_HPP

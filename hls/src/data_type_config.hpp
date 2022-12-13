
// data_type_config.hpp

#ifndef DATA_TYPE_CONFIG_HPP
#define DATA_TYPE_CONFIG_HPP

#include <ap_fixed.h>
#include <ap_int.h>

// Value types
template <int _AP_W, int _AP_I>
using ap_fixed_sat = ap_fixed<
  _AP_W, _AP_I, ap_q_mode::AP_TRN, ap_o_mode::AP_SAT, 0>;

#ifndef USE_FLOAT
// Data width for the values
#ifndef VALUE_BIT_WIDTH
#warning `VALUE_BIT_WIDTH` is not defined (default: 32)
constexpr const int kValueBitWidth = 32;
#elif VALUE_BIT_WIDTH > 32
#error `VALUE_BIT_WIDTH` should be less than or equal to 32
#else
constexpr const int kValueBitWidth = VALUE_BIT_WIDTH;
#endif // VALUE_BIT_WIDTH

// Integer width for the values
#ifndef VALUE_INT_WIDTH
#warning `VALUE_INT_WIDTH` is not defined (default: 16)
constexpr const int kValueIntWidth = 16;
#else
constexpr const int kValueIntWidth = VALUE_INT_WIDTH;
#endif // VALUE_INT_WIDTH

// Data width for the network parameters
#ifndef PARAM_BIT_WIDTH
#warning `PARAM_BIT_WIDTH` is not defined (default: 32)
constexpr const int kParamBitWidth = 32;
#elif PARAM_BIT_WIDTH > 32
#error `PARAM_BIT_WIDTH` should be less than or equal to 32
#else
constexpr const int kParamBitWidth = PARAM_BIT_WIDTH;
#endif // PARAM_BIT_WIDTH

// Integer width for the network parameters
#ifndef PARAM_INT_WIDTH
#warning `PARAM_INT_WIDTH` is not defined (default: 16)
constexpr const int kParamIntWidth = 16;
#else
constexpr const int kParamIntWidth = PARAM_INT_WIDTH;
#endif // PARAM_INT_WIDTH

// Data type for values (layer inputs, outputs, and intermediate results)
using value_t = ap_fixed_sat<kValueBitWidth, kValueIntWidth>;
// Data type for network parameters
using param_t = ap_fixed_sat<kParamBitWidth, kParamIntWidth>;
#else
// Data type for values (layer inputs, outputs, and intermediate results)
using value_t = float;
// Data type for network parameters
using param_t = float;
#endif // USE_FLOAT

#endif // DATA_TYPE_CONFIG_HPP

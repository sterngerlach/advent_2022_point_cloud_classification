
// vector_utils.hpp

#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP

// Set the `N`D vector to zero (`x` = 0)
template <typename T, int N>
inline void VectorNdSetZero(T x[N])
{
  for (int i = 0; i < N; ++i) {
    x[i] = T(0);
  }
}

// Set the `N`D vector to zero (`x` = 0)
// template <typename T, int N>
// inline void VectorNdSetZero(T (&x)[N])
// {
//   for (int i = 0; i < N; ++i) {
//     x[i] = T(0);
//   }
// }

#endif // VECTOR_UTILS_HPP

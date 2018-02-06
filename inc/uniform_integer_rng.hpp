#ifndef BGM_UNIFORM_INTEGER_RNG_HPP_
#define BGM_UNIFORM_INTEGER_RNG_HPP_

#ifdef USE_GLOG
#include "glog/logging.h"
#else
#include <cassert>
#endif // USE_GLOG

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>

namespace bgm
{

template <typename Integer>
class UniformIntegerRNG
{
 public:
  static std::shared_ptr<UniformIntegerRNG<Integer> > GetInstance();

  Integer Random(Integer min = std::numeric_limits<Integer>::min(),
                 Integer max = std::numeric_limits<Integer>::max());
  template <typename Iterator>
  void Random(Iterator& dst_begin, int num = 1,
              Integer min = std::numeric_limits<Integer>::min(),
              Integer max = std::numeric_limits<Integer>::max());
  template <typename Iterator>
  void Random(Iterator& dst_begin, Iterator& dst_end,
              Integer min = std::numeric_limits<Integer>::min(),
              Integer max = std::numeric_limits<Integer>::max());

 private:
  UniformIntegerRNG();

  std::function<Integer(void)> rng_;
  std::mutex mutex_;

  static std::shared_ptr<UniformIntegerRNG<Integer> > instance_;
};

template <typename Integer>
std::shared_ptr<UniformIntegerRNG<Integer> > UniformIntegerRNG<Integer>::instance_;

// template functions

template <typename Integer>
inline std::shared_ptr<UniformIntegerRNG<Integer> > 
    UniformIntegerRNG<Integer>::GetInstance() {
  if (!instance_)
    instance_.reset(new UniformIntegerRNG<Integer>());
  return instance_;
}

template <typename Integer>
Integer UniformIntegerRNG<Integer>::Random(Integer min, Integer max) {
#ifdef USE_GLOG
  CHECK_LE(min, max);
#else
  assert(min <= max);
#endif // USE_GLOG
  
  if (min == max)
    return min;

  mutex_.lock();
  Integer random = (rng_() % (max - min)) + min;
  mutex_.unlock();
  return random;
}

template <typename Integer>
template <typename Iterator>
inline void UniformIntegerRNG<Integer>::Random(
    Iterator& dst_begin, int num, Integer min, Integer max) {
#ifdef USE_GLOG
  CHECK_GE(num, 0);
  CHECK_LE(min, max);
#else
  assert(num >= 0);
  assert(min <= max);
#endif // USE_GLOG

  for (int n = num; n--;)
    *dst_begin++ = Random(min, max);
}

template <typename Integer>
template <typename Iterator>
void UniformIntegerRNG<Integer>::Random(
    Iterator& dst_begin, Iterator& dst_end, Integer min, Integer max) {
#ifdef USE_GLOG
  CHECK(dst_begin < dst_end);
  CHECK_LE(min, max);
#else
  assert(dst_begin < dst_end);
  assert(min <= max);
#endif // USE_GLOG
  for (auto iter = dst_begin; iter != dst_end; iter++) {
    *iter = Random(min, max);
  }
}

template <typename Integer>
UniformIntegerRNG<Integer>::UniformIntegerRNG() {
  std::uniform_int_distribution<Integer> dist;
  rng_ = std::bind(dist,
                   std::default_random_engine(
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count()));
}

} // namespace bgm

#endif // !BGM_UNIFORM_INTEGER_RNG_HPP_

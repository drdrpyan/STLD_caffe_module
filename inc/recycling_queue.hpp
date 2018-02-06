#ifndef TLR_RECYCLING_QUEUE_HPP_
#define TLR_RECYCLING_QUEUE_HPP_

#include <deque>
#include <list>
#include <memory>

namespace bgm
{

template <typename Elem>
class RecyclingQueue
{
 public:
  const std::unique_ptr<Elem>* Top() const;
  std::unique_ptr<Elem>* Top();
  
  void Push(Elem* elem);
  void Pop();

  Elem* GetEmptyBin(bool allow_new = true);

  int Size() const;

 private:
  std::deque<std::unique_ptr<Elem> > full_;
  std::list<std::unique_ptr<Elem> > empty_;
};

// template inline functions
template <typename Elem>
inline const std::unique_ptr<Elem>* RecyclingQueue<Elem>::Top() const {
  return full_.empty() ? nullptr : &(full_.front());
}

template <typename Elem>
inline std::unique_ptr<Elem>* RecyclingQueue<Elem>::Top() {
  return const_cast<std::unique_ptr<Elem>*>(
      static_cast<const RecyclingQueue<Elem>*>(this)->Top());
}

template <typename Elem>
inline void RecyclingQueue<Elem>::Push(Elem* elem) {
  //full_.push_back(std::unique_ptr<Elem>(elem));
  full_.emplace_back(elem);
}

template <typename Elem>
inline int RecyclingQueue<Elem>::Size() const {
  return full_.size();
}

// template functions
template <typename Elem>
void RecyclingQueue<Elem>::Pop() {
  if (full_.empty())
    return;

  Elem* empty_elem = full_.front().release();
  full_.pop_front();

  empty_.emplace_back(empty_elem);
  //empty_.push_back(std::unique_ptr<Elem>(empty_elem));
}

template <typename Elem>
Elem* RecyclingQueue<Elem>::GetEmptyBin(bool allow_new) {
  if (empty_.empty()) {
    return allow_new ? new Elem : nullptr;
  }
  else {
    Elem* empty_elem = empty_.front().release();
    empty_.pop_front();
    return empty_elem;
  }
}





} // namespace bgm
#endif // !TLR_RECYCLING_QUEUE_HPP_

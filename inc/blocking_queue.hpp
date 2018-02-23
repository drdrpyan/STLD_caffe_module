//#ifndef BGM_UTIL_BLOCKING_QUEUE_HPP_
//#define BGM_UTIL_BLOCKING_QUEUE_HPP_
//
//#include "boost/shared_ptr.hpp"
//
//#include <queue>
//#include <string>
//
//namespace bgm {
//
//template<typename T>
//class BlockingQueue {
// public:
//  explicit BlockingQueue();
//
//  void push(const T& t);
//
////  bool try_pop(T* t);
//
//  // This logs a message if the threads needs to be blocked
//  // useful for detecting e.g. when data feeding is too slow
//  T pop(const std::string& log_on_wait = "");
//
////  bool try_peek(T* t);
////
////  // Return element without removing it
////  T peek();
////
////  size_t size() const;
//
// protected:
//  /**
//   Move synchronization fields out instead of including boost/thread.hpp
//   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
//   Linux CUDA 7.0.18.
//   */
//  class sync;
//
////  std::queue<T> queue_;
//  boost::shared_ptr<sync> sync_;
//
//////DISABLE_COPY_AND_ASSIGN(BlockingQueue);
////  private:
////    BlockingQueue(const BlockingQueue&); 
////    BlockingQueue& operator=(const BlockingQueue&);
//};
//
//// template functions
//template<typename T>
//class BlockingQueue<T>::sync {
// public:
//  mutable boost::mutex mutex_;
//  boost::condition_variable condition_;
//};
//
//template<typename T>
//BlockingQueue<T>::BlockingQueue()
//    : sync_(new sync()) {
//}
//
//template<typename T>
//void BlockingQueue<T>::push(const T& t) {
//  boost::mutex::scoped_lock lock(sync_->mutex_);
//  queue_.push(t);
//  lock.unlock();
//  sync_->condition_.notify_one();
//}
//
////template<typename T>
////bool BlockingQueue<T>::try_pop(T* t) {
////  boost::mutex::scoped_lock lock(sync_->mutex_);
////
////  if (queue_.empty()) {
////    return false;
////  }
////
////  *t = queue_.front();
////  queue_.pop();
////  return true;
////}
//
//template<typename T>
//T BlockingQueue<T>::pop(const std::string& log_on_wait) {
//  boost::mutex::scoped_lock lock(sync_->mutex_);
//
//  while (queue_.empty()) {
//    if (!log_on_wait.empty()) {
//      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
//    }
//    sync_->condition_.wait(lock);
//  }
//
//  T t = queue_.front();
//  queue_.pop();
//  return t;
//}
//
////template<typename T>
////bool BlockingQueue<T>::try_peek(T* t) {
////  boost::mutex::scoped_lock lock(sync_->mutex_);
////
////  if (queue_.empty()) {
////    return false;
////  }
////
////  *t = queue_.front();
////  return true;
////}
////
////template<typename T>
////T BlockingQueue<T>::peek() {
////  boost::mutex::scoped_lock lock(sync_->mutex_);
////
////  while (queue_.empty()) {
////    sync_->condition_.wait(lock);
////  }
////
////  return queue_.front();
////}
////
////template<typename T>
////size_t BlockingQueue<T>::size() const {
////  boost::mutex::scoped_lock lock(sync_->mutex_);
////  return queue_.size();
////}
//
//}  // namespace bgm
//
//#endif // !BGM_UTIL_BLOCKING_QUEUE_HPP_

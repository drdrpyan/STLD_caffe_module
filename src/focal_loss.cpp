#include "focal_loss.hpp"

#include "glog/logging.h"

#include <algorithm>
#include <limits>

namespace bgm
{
//template <typename Dtype>
//void FocalLoss<Dtype>::SigmoidRegressionFocalLoss(
//    Dtype value, Dtype target, Dtype alpha, Dtype gamma, 
//    Dtype* loss, Dtype* diff) const {
//  const Dtype sig_value = Sigmoid(value);
//  
//  RegressionFocalLoss(sig_value, target, alpha, gamma, loss, diff);
//  
//  const Dtype sig_diff = sig_value - (sig_value*sig_value); /* s(1-s) */
//  (*diff) *= sig_diff;  
//}
//
//template <typename Dtype>
//void FocalLoss<Dtype>::RegressionFocalLoss(Dtype prob, Dtype target,
//                                           Dtype alpha, Dtype gamma,
//                                           Dtype* loss, Dtype* diff) const {
//  CHECK_GE(prob, 0);
//  CHECK_LE(prob, 1);
//  CHECK_GE(target, 0);
//  CHECK_LE(target, 1);
//
//  Dtype abs_sub = std::abs(prob - target);
//  BaseFocalLoss(1 - abs_sub, alpha, gamma, loss, diff);
//  if (prob > target)
//    (*diff) = -(*diff);
//}
//
//template <typename Dtype>
//void FocalLoss<Dtype>::SoftmaxFocalLoss(const std::vector<Dtype>& value, 
//                                        int target, Dtype alpha, Dtype gamma,
//                                        std::vector<Dtype>* loss, 
//                                        std::vector<Dtype>* diff) const {
//  CHECK_GE(target, 0);
//  CHECK_LT(target, value.size());
//  CHECK(loss);
//  CHECK(diff);
//  
//  std::vector<Dtype> softmax;
//  Softmax(value, &softmax);
//
//  loss->resize(value.size());
//  diff->resize(value.size());
//
//  for (int i = 0; i < softmax.size(); ++i) {
//    Dtype base_diff;
//    BaseFocalLoss(softmax[i], alpha, gamma, &((*loss)[i]), &base_diff);
//    
//    Dtype softmax_diff = -(softmax[i] * softmax[i]) - ((i != target) ? softmax[i] : 0);
//    (*diff)[i] = base_diff * softmax_diff;
//  }
//}
//
//template <typename Dtype>
//void FocalLoss<Dtype>::BaseFocalLoss(Dtype prob, Dtype alpha, Dtype gamma,
//                                     Dtype* loss, Dtype* diff) const {
//  CHECK_GE(prob, 0);
//  CHECK_LE(prob, 1);
//  CHECK_GT(alpha, 0);
//  CHECK_GE(gamma, 0);
//  CHECK(loss);
//  CHECK(diff);
//  
//  const Dtype not_prob = 1 - prob;
//  const Dtype gamma_pow = std::pow(not_prob, gamma);
//  const Dtype log_prob = std::log(std::max(prob, std::numeric_limits<Dtype>::min()));
//  
//  *loss = (-alpha) * gamma_pow * log_prob;
//  *diff = (alpha*gamma_pow)*((gamma*log_prob / not_prob) - (1 / prob));
//}
//
//template <typename Dtype>
//void FocalLoss<Dtype>::Softmax(const std::vector<Dtype>& value,
//                               std::vector<Dtype>* softmax) const {
//  CHECK(!value.empty());
//  CHECK(softmax);
//
//  softmax->resize(value.size());
//  Dtype exp_sum = 0;
//  for (int i = 0; i < value.size(); ++i) {
//    Dtype exp_value = std::exp(value[i]);
//    exp_sum += exp_value;
//    (*softmax)[i] = exp_value;
//  }
//
//  for (int i = 0; i < value.size(); ++i)
//    ((*softmax)[i]) /= exp_sum;
//}


} // namespace bgm
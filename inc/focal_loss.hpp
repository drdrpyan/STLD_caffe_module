#ifndef BGM_FOCAL_LOSS_HPP_
#define BGM_FOCAL_LOSS_HPP_

#include <vector>

#include "glog/logging.h"

namespace bgm
{

template <typename Dtype>
class FocalLoss
{
 public:
  void SigmoidRegressionFocalLoss(Dtype value, Dtype target,
                                  Dtype alpha, Dtype gamma,
                                  Dtype* loss, Dtype* diff) const;
  void RegressionFocalLoss(Dtype prob, Dtype target,
                           Dtype alpha, Dtype gamma,
                           Dtype* loss, Dtype* diff) const;
  void SoftmaxFocalLoss(const std::vector<Dtype>& value, int target,
                        Dtype alpha, Dtype gamma,
                        Dtype* loss, 
                        std::vector<Dtype>* diff) const;
  void SoftmaxFocalLoss(const std::vector<Dtype>& value, int target,
                        Dtype alpha, Dtype gamma,
                        std::vector<Dtype>* prob,
                        Dtype* loss, 
                        std::vector<Dtype>* diff) const;
 private:
  void BaseFocalLoss(Dtype prob, Dtype alpha, Dtype gamma,
                     Dtype* loss, Dtype* diff) const;
  void Softmax(const std::vector<Dtype>& value,
               std::vector<Dtype>* softmax) const;
  Dtype Sigmoid(Dtype value) const;
  
}; // class FocalLoss

// inline functions
template <typename Dtype>
inline Dtype FocalLoss<Dtype>::Sigmoid(Dtype value) const {
  return 1 / (1 + std::exp(-value));
}

// template functions
template <typename Dtype>
void FocalLoss<Dtype>::SigmoidRegressionFocalLoss(
    Dtype value, Dtype target, Dtype alpha, Dtype gamma, 
    Dtype* loss, Dtype* diff) const {
  const Dtype sig_value = Sigmoid(value);
  
  RegressionFocalLoss(sig_value, target, alpha, gamma, loss, diff);
  
  const Dtype sig_diff = sig_value - (sig_value*sig_value); /* s(1-s) */
  (*diff) *= sig_diff;  
}

template <typename Dtype>
void FocalLoss<Dtype>::RegressionFocalLoss(Dtype prob, Dtype target,
                                           Dtype alpha, Dtype gamma,
                                           Dtype* loss, Dtype* diff) const {
  CHECK_GE(prob, 0);
  CHECK_LE(prob, 1);
  CHECK_GE(target, 0);
  CHECK_LE(target, 1);

  Dtype abs_sub = std::abs(prob - target);
  BaseFocalLoss(1 - abs_sub, alpha, gamma, loss, diff);
  if (prob > target)
    (*diff) = -(*diff);
}

template <typename Dtype>
void FocalLoss<Dtype>::SoftmaxFocalLoss(const std::vector<Dtype>& value, 
                                        int target, Dtype alpha, Dtype gamma,
                                        Dtype* loss, 
                                        std::vector<Dtype>* diff) const {
  CHECK_GE(target, 0);
  CHECK_LT(target, value.size());
  CHECK(loss);
  CHECK(diff);
  
  std::vector<Dtype> softmax;
  Softmax(value, &softmax);

  diff->resize(value.size());

  Dtype base_diff;
  BaseFocalLoss(softmax[target], alpha, gamma, loss, &base_diff);
  for(int i=0; i<softmax.size(); ++i)
    if(i != target)
      (*diff)[i] = base_diff * (-softmax[i]) * softmax[target];
  else
      (*diff)[i] = base_diff * softmax[i] * (1 - softmax[i]);
}

template <typename Dtype>
void FocalLoss<Dtype>::SoftmaxFocalLoss(const std::vector<Dtype>& value,
                                        int target,
                                        Dtype alpha, Dtype gamma,
                                        std::vector<Dtype>* prob,
                                        Dtype* loss,
                                        std::vector<Dtype>* diff) const {
  CHECK_GE(target, 0);
  CHECK_LT(target, value.size());
  CHECK(prob);
  CHECK(loss);
  CHECK(diff);
  
  Softmax(value, prob);

  diff->resize(value.size());

  Dtype base_diff;
  BaseFocalLoss((*prob)[target], alpha, gamma, loss, &base_diff);
  for(int i=0; i<prob->size(); ++i)
    if(i != target)
      (*diff)[i] = base_diff * (-(*prob)[i]) * (*prob)[target];
  else
      (*diff)[i] = base_diff * (*prob)[i] * (1 - (*prob)[i]);
}

template <typename Dtype>
void FocalLoss<Dtype>::BaseFocalLoss(Dtype prob, Dtype alpha, Dtype gamma,
                                     Dtype* loss, Dtype* diff) const {
  CHECK_GE(prob, 0);
  CHECK_LE(prob, 1);
  CHECK_GT(alpha, 0);
  CHECK_GE(gamma, 0);
  CHECK(loss);
  CHECK(diff);
  
  const Dtype nonzero_prob = std::max(prob, 
                                      std::numeric_limits<Dtype>::min());
  const Dtype nonzero_not_prob = std::max(1 - nonzero_prob, 
                                          std::numeric_limits<Dtype>::min());
  const Dtype gamma_pow = std::pow(nonzero_not_prob, gamma);
  const Dtype log_prob = std::log(nonzero_prob);
  
  *loss = (-alpha) * gamma_pow * log_prob;
  *diff = (alpha*gamma_pow)*((gamma*log_prob / nonzero_not_prob) - (1 / nonzero_prob));

  CHECK(!isnan(*loss));
  CHECK(!isinf(*loss));
  CHECK(!isnan(*diff));
  CHECK(!isinf(*diff));
}

template <typename Dtype>
void FocalLoss<Dtype>::Softmax(const std::vector<Dtype>& value,
                               std::vector<Dtype>* softmax) const {
  CHECK(!value.empty());
  CHECK(softmax);

  Dtype max_val = *(std::max_element(value.cbegin(), value.cend()));

  softmax->resize(value.size());
  Dtype exp_sum = 0;
  for (int i = 0; i < value.size(); ++i) {
    Dtype exp_value = std::exp(value[i] - max_val);
    exp_sum += exp_value;
    (*softmax)[i] = exp_value;
  }

  for (int i = 0; i < value.size(); ++i)
    ((*softmax)[i]) /= exp_sum;
}

} // namespace bgm

#endif // !BGM_FOCAL_LOSS_HPP_

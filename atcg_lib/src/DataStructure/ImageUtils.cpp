#include <DataStructure/Image.h>

namespace atcg
{

namespace detail
{
torch::Tensor reducdeTensor(const torch::Tensor& tensor, const std::string& reduction)
{
    if(reduction == "mean")
    {
        return torch::mean(tensor, {-1});
    }
    else if(reduction == "norm")
    {
        return torch::norm(tensor, 2, -1);
    }
    else if(reduction == "sum")
    {
        return torch::sum(tensor, {-1});
    }
    else
    {
        return tensor;
    }
}
}    // namespace detail

torch::Tensor
Utils::AEMap(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction)
{
    TORCH_CHECK_EQ(ground_truth.size(0), prediction.size(0));
    TORCH_CHECK_EQ(ground_truth.size(1), prediction.size(1));
    TORCH_CHECK_EQ(ground_truth.size(2), prediction.size(2));
    TORCH_CHECK_EQ(ground_truth.size(3), prediction.size(3));

    torch::Tensor gt_reduced         = detail::reducdeTensor(ground_truth, channel_reduction);
    torch::Tensor prediction_reduced = detail::reducdeTensor(prediction, channel_reduction);

    return torch::abs(gt_reduced - prediction_reduced);
}

torch::Tensor Utils::relAEMap(const torch::Tensor& ground_truth,
                              const torch::Tensor& prediction,
                              const std::string& channel_reduction,
                              const float delta)
{
    auto ae_map = AEMap(ground_truth, prediction, channel_reduction);

    torch::Tensor gt_reduced = detail::reducdeTensor(ground_truth, channel_reduction);

    return ae_map / (torch::abs(gt_reduced) + delta);
}

torch::Tensor
Utils::SEMap(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction)
{
    auto ae_map = AEMap(ground_truth, prediction, channel_reduction);

    return ae_map * ae_map;
}

torch::Tensor Utils::relSEMap(const torch::Tensor& ground_truth,
                              const torch::Tensor& prediction,
                              const std::string& channel_reduction,
                              const float delta)
{
    auto ae_map              = AEMap(ground_truth, prediction, channel_reduction);
    torch::Tensor gt_reduced = detail::reducdeTensor(ground_truth, channel_reduction);

    return (ae_map * ae_map) / (gt_reduced * gt_reduced + delta);
}

torch::Tensor
Utils::MAE(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction)
{
    return torch::mean(AEMap(ground_truth, prediction, channel_reduction), {1, 2});
}

torch::Tensor Utils::relMAE(const torch::Tensor& ground_truth,
                            const torch::Tensor& prediction,
                            const std::string& channel_reduction,
                            const float delta)
{
    return torch::mean(relAEMap(ground_truth, prediction, channel_reduction, delta), {1, 2});
}

torch::Tensor
Utils::MSE(const torch::Tensor& ground_truth, const torch::Tensor& prediction, const std::string& channel_reduction)
{
    return torch::mean(relSEMap(ground_truth, prediction, channel_reduction), {1, 2});
}


torch::Tensor Utils::relMSE(const torch::Tensor& ground_truth,
                            const torch::Tensor& prediction,
                            const std::string& channel_reduction,
                            const float delta)
{
    return torch::mean(relSEMap(ground_truth, prediction, channel_reduction, delta), {1, 2});
}
}    // namespace atcg

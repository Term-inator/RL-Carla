import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape([input.shape[0], -1])


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 512 # 7 * 7 * 64
        # nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(value=0.0))
        # 预测模型
        self.predictor = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=self.input_size,
            #     out_channels=32,
            #     kernel_size=8,
            #     stride=4),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=4,
            #     stride=2),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=64,
            #     out_channels=64,
            #     kernel_size=3,
            #     stride=1),
            # nn.LeakyReLU(),
            # Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # 随机网络
        self.target = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=self.input_size,
            #     out_channels=32,
            #     kernel_size=8,
            #     stride=4),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=4,
            #     stride=2),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=64,
            #     out_channels=64,
            #     kernel_size=3,
            #     stride=1),
            # nn.LeakyReLU(),
            # Flatten(),
            nn.Linear(feature_output, 512)
        )

        # for index, param in enumerate(self.target.parameters()):
        #     param.stop_gradient = True  # 随机网络不需要梯度更新参数
        #     self.target.parameters()[index] = param.sign() * param.abs().sqrt(2)

        for index, target_param in enumerate(self.target.parameters()):
            target_param.data.copy_(
                target_param.sign() * target_param.abs().sqrt()
            )

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


# 如何计算内在奖励
def compute_intrinsic_reward(rnd, next_obs):
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

    return intrinsic_reward.cpu().detach().numpy().item()

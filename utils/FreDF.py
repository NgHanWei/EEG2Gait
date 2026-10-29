import torch
import torch.nn as nn
import torch.fft as fft


# 定义损失函数
class FreDFLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(FreDFLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()  # 时间域的MSE损失

    def forward(self, pred, target):
        # 1. 计算时间域损失
        time_loss = self.mse_loss(pred, target)

        # 2. 进行傅里叶变换，转换到频域
        pred_freq = fft.fft(pred.to(torch.float32), dim=-1)  # 对最后一个维度做FFT
        target_freq = fft.fft(target.to(torch.float32), dim=-1)

        # 3. 计算频域损失 (这里用L1损失，因为频域数值差异较大，L2可能不稳定)
        freq_loss = torch.mean(torch.abs(pred_freq - target_freq))

        # 4. 加权组合两个损失
        total_loss = self.alpha * freq_loss + (1 - self.alpha) * time_loss
        return total_loss


class GradientNormalizationLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(GradientNormalizationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 1. 计算时间域损失
        time_loss = self.mse_loss(pred, target)

        # 2. 计算损失的梯度
        grad = torch.autograd.grad(time_loss, pred, create_graph=True)[0]
        grad_norm = torch.norm(grad, p=2) + self.epsilon  # 防止梯度为0

        # 3. 使用梯度归一化调整损失
        normalized_loss = time_loss / grad_norm

        return normalized_loss.mean()

class FreqGradientLoss(nn.Module):
    def __init__(self, alpha=0.3, epsilon=1e-5):
        super(FreqGradientLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()  # 时间域的MSE损失

    def forward(self, pred, target):
        # 1. 计算时间域损失
        time_loss = self.mse_loss(pred, target)

        # 2. 进行傅里叶变换，转换到频域
        pred_freq = fft.fft(pred.to(torch.float32), dim=-1)  # 对最后一个维度做FFT
        target_freq = fft.fft(target.to(torch.float32), dim=-1)

        # 3. 计算频域损失 (这里用L1损失，因为频域数值差异较大，L2可能不稳定)
        freq_loss = torch.mean(torch.abs(pred_freq - target_freq))

        # 4. 计算时间域损失的梯度
        grad = torch.autograd.grad(time_loss, pred, create_graph=True)[0]
        grad_norm = torch.norm(grad, p=2) + self.epsilon  # 防止梯度为0

        # 5. 使用梯度归一化调整时间域损失
        normalized_time_loss = time_loss / grad_norm

        # 6. 加权组合频域损失和归一化的时间域损失
        total_loss = self.alpha * freq_loss + (1 - self.alpha) * normalized_time_loss
        # print('tmp', self.alpha * freq_loss / (1 - self.alpha) * normalized_time_loss)

        return total_loss.mean()


# 定义只有MSE和Encouraging Loss的损失函数
class MSEWithEncouragingLoss(nn.Module):
    def __init__(self, beta=0.1, epsilon=1e-5):
        super(MSEWithEncouragingLoss, self).__init__()
        self.beta = beta  # Encouraging Loss 的权重
        self.epsilon = epsilon  # 避免log(0)的数值不稳定性
        self.mse_loss = nn.MSELoss()  # 时间域的MSE损失

    def encouraging_loss(self, loss):
        """
        计算Encouraging Loss的额外奖励部分:
        - loss: 输入的原始损失 (可以是时间域的MSE损失)
        - 返回: 加入了Encouraging部分的损失
        """
        reward = torch.log(1 - torch.exp(-loss) + self.epsilon)  # 增加奖励项
        return loss + self.beta * reward  # 加入奖励项后新的损失

    def forward(self, pred, target):
        # 1. 计算时间域损失
        time_loss = self.mse_loss(pred, target)

        # 2. 加入Encouraging部分
        total_loss = self.encouraging_loss(time_loss)

        return total_loss


class FreDFLossWithEncouraging(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, epsilon=1e-5):
        super(FreDFLossWithEncouraging, self).__init__()
        self.alpha = alpha  # 频域和时间域损失的权重
        self.beta = beta  # Encouraging Loss 的权重
        self.epsilon = epsilon  # 避免log(0)的数值不稳定性
        self.mse_loss = nn.MSELoss()  # 时间域的MSE损失

    def encouraging_loss(self, loss):
        """
        计算Encouraging Loss的额外奖励部分:
        - loss: 输入的原始损失 (可以是时间域或频域的损失)
        - 返回: 加入了Encouraging部分的损失
        """
        reward = torch.log(1 - torch.exp(-loss) + self.epsilon)  # 增加奖励项
        # print('mse loss:', loss)
        # print('reward:', reward)
        return loss + self.beta * reward  # 加入奖励项后新的损失

    def forward(self, pred, target):
        # 1. 计算时间域损失
        time_loss = self.mse_loss(pred, target)
        time_loss = self.encouraging_loss(time_loss)  # 加入Encouraging部分

        # 2. 进行傅里叶变换，转换到频域
        pred_freq = fft.fft(pred.to(torch.float32), dim=-1)  # 对最后一个维度做FFT
        target_freq = fft.fft(target.to(torch.float32), dim=-1)

        # 3. 计算频域损失 (这里用L1损失，因为频域数值差异较大，L2可能不稳定)
        freq_loss = torch.mean(torch.abs(pred_freq - target_freq))
        # print('freq loss:', freq_loss)
        freq_loss = self.encouraging_loss(freq_loss)  # 加入Encouraging部分
        # print('freq loss after encouraging:', freq_loss)
        # print('time loss', time_loss)

        # 4. 加权组合两个损失
        total_loss = self.alpha * freq_loss + (1 - self.alpha) * time_loss
        return total_loss



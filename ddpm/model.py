"""M3 Tiny DiT：核心计算已由学习者完成，保留 M3-1 至 M3-5b 的实现提示。

形状记号：B=批大小，C=通道数，S=图片边长，P=patch 边长，
N=(S/P)^2=token 数，F=P*P*C=每块像素数，D=隐藏维度。
"""

import torch
from torch import Tensor, nn


def _validate_layout(image_size: int, patch_size: int, in_channels: int) -> None:
    """功能：检查正方形图片与 patch 的几何配置，不执行像素转换。

    参数：
        image_size：图片边长 S，必须为正。
        patch_size：patch 边长 P，必须为正并能整除 S。
        in_channels：输入通道数 C，必须为正；MNIST 为 1。
    返回：
        None。非法配置抛出 ValueError。
    """
    if image_size <= 0 or patch_size <= 0 or in_channels <= 0:
        raise ValueError("image_size, patch_size, and in_channels must be positive")
    if image_size % patch_size != 0:
        raise ValueError("patch_size must divide image_size")


def patchify(images: Tensor, patch_size: int = 4) -> Tensor:
    """功能：将正方形图片拆成不重叠的 patch，并展平为 token 像素向量。

    参数：
        images：[B,C,S,S] 图片张量；保持原 dtype、device 与梯度连接。
        patch_size：每块的边长 P，默认 4；S 必须可被 P 整除。
    返回：
        [B,N,F] 张量，N=(S/P)^2，F=P*P*C。
        token 按 patch 网格先行后列排列；块内特征按像素行、像素列、通道排列。
        例如双通道时，同一像素的两个通道值相邻，而非整块按通道分组。
    """
    if images.ndim != 4 or images.shape[0] <= 0 or images.shape[2] != images.shape[3]:
        raise ValueError("images must have nonempty square shape [B,C,S,S]")
    _validate_layout(images.shape[2], patch_size, images.shape[1])
    # M3-1：先把 H/W 分别拆成「网格轴 × 块内像素轴」，再重新排列各轴。
    # 目标轴顺序为 [B,网格行,网格列,块内行,块内列,C]，最后合并成 [B,N,F]。
    # 提示：reshape、permute；permute 后不要假设张量仍连续，可继续使用 reshape。
    # 此函数只重排像素，不做 Linear 投影，也不添加位置或时间信息。
    batch_size, channels, image_size, _ = images.shape
    grid_size = image_size // patch_size
    return (
        images.reshape(batch_size, channels, grid_size, patch_size, grid_size, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch_size, grid_size * grid_size, patch_size * patch_size * channels)
    )


def unpatchify(
    patches: Tensor, image_size: int = 28, patch_size: int = 4, in_channels: int = 1
) -> Tensor:
    """功能：将 patch 像素向量按 patchify 的排列约定还原为图片。

    参数：
        patches：[B,N,F] 张量；此处 F 是 P*P*C，不是 Transformer 的隐藏维度 D。
        image_size：输出图片边长 S，默认 28。
        patch_size：patch 边长 P，必须与拆分时一致，默认 4。
        in_channels：输出通道数 C，默认 1；噪声预测与输入图片通道数一致。
    返回：
        [B,C,S,S] 张量，保持 patches 的 dtype、device 和梯度连接。
    """
    _validate_layout(image_size, patch_size, in_channels)
    expected = ((image_size // patch_size) ** 2, patch_size * patch_size * in_channels)
    if patches.ndim != 3 or patches.shape[0] <= 0 or tuple(patches.shape[1:]) != expected:
        raise ValueError("patches must have shape [B,(S/P)^2,P*P*C]")
    # M3-2：逆转 M3-1 的轴变换，先恢复网格与块内轴，再还原空间 H/W。
    # 请勿只把 [B,N,F] 直接 reshape 成 [B,C,S,S]，那会打乱像素位置。
    batch_size = patches.shape[0]
    grid_size = image_size // patch_size
    return (
        patches.reshape(batch_size, grid_size, grid_size, patch_size, patch_size, in_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch_size, in_channels, image_size, image_size)
    )


def timestep_embedding(t: Tensor, dim: int = 128, max_period: float = 10000.0) -> Tensor:
    """功能：把离散时间步转换为固定的正弦/余弦特征，本函数没有可学习权重。

    参数：
        t：[B] int64 非负时间步；沿用 M2 的原始下标，不除以扩散总步数 T。
        dim：输出维度 D，要求为正偶数且至少为 2，默认 128。
        max_period：控制频率跨度的基数，必须大于 1，默认 10000；不是最大合法 t。
    返回：
        [B,D] float32 张量，与 t 同设备；前 D/2 维为 cos，后 D/2 维为 sin。
        第 j 个频率为 exp(-log(max_period)*j/(D/2))，j=0,...,D/2-1。
    """
    if t.ndim != 1 or t.numel() == 0 or t.dtype != torch.int64 or t.min().item() < 0:
        raise ValueError("t must be a nonempty, nonnegative int64 vector")
    if dim < 2 or dim % 2 != 0 or max_period <= 1:
        raise ValueError("dim must be positive and even; max_period must exceed 1")
    # M3-3：生成 D/2 个频率，让 [B,1] 的时间步与 [1,D/2] 频率广播相乘。
    # 对结果分别求 cos/sin，再沿特征轴拼接；不要在此函数创建 nn.Linear。
    # 提示：arange、exp、log、cos、sin、cat；显式使用 t.device 和 float32。
    half_dim = dim // 2
    indices = torch.arange(half_dim, device=t.device, dtype=torch.float32)
    frequencies = torch.exp(
        -torch.log(torch.tensor(max_period, device=t.device, dtype=torch.float32))
        * indices
        / half_dim
    )
    angles = t.to(dtype=torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


class TransformerBlock(nn.Module):
    """带两条残差支路的 Pre-LayerNorm 双向自注意力 block；不含位置/时间编码。"""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4, mlp_ratio: int = 4) -> None:
        """功能：定义本 block 的归一化、注意力与 MLP 子模块。

        参数：
            hidden_size：每个 token 的特征维度 D，必须可被 num_heads 整除。
            num_heads：注意力头数，默认 4；每头维度为 D/num_heads。
            mlp_ratio：MLP 中间层宽度相对 D 的倍率，默认 4，必须为正整数。
        返回：
            None；实例具有可训练的 norm1、attention、norm2、mlp 模块。
        """
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0 or mlp_ratio <= 0:
            raise ValueError("Invalid hidden_size, num_heads, or mlp_ratio")
        self.hidden_size = hidden_size
        # M3-4a：定义上述四个模块，norm1/norm2 都沿最后一个 D 维归一化。
        # attention 使用 nn.MultiheadAttention，设置 batch_first=True、dropout=0.0。
        # mlp 使用 Linear(D,D*mlp_ratio) → GELU → Linear(D*mlp_ratio,D)。
        # 子模块要赋给 self 才能注册参数；本阶段不必自己重写多头注意力内部计算。
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """功能：依次执行「归一化→自注意力→残差」和「归一化→MLP→残差」。

        参数：
            tokens：[B,N,D] token 特征；各图片应独立计算注意力，不能混合 batch。
        返回：
            [B,N,D] 更新后的特征，形状不变；每个 token 可关注本图的所有 patch。
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,N,hidden_size]")
        # M3-4b：在归一化后的同一组 token 上构造 Q、K、V，执行双向自注意力。
        # MultiheadAttention 返回 (特征,注意力权重)；need_weights=False 可省去权重输出。
        # 残差连接要保留支路前的 tokens；第二条支路使用第一次残差相加后的特征。
        # 不使用 causal mask，也不把 N 轴当成 batch 轴。
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class TinyDiT(nn.Module):
    """M3 的像素空间噪声预测器：图片与时间步输入，输出同形状的 epsilon 预测。"""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
    ) -> None:
        """功能：定义图片投影、位置条件、时间条件、block 堆叠和输出头。

        参数：
            image_size：正方形图片边长 S，默认 28；不是 token 数。
            patch_size：patch 边长 P，默认 4；28/4 得到 7×7=49 个 token。
            in_channels：图片通道数 C，默认 1；也决定预测噪声的通道数。
            hidden_size：隐藏维度 D，默认 128；要求为偶数并可被 num_heads 整除。
            depth：Transformer block 数量，默认 4，必须为正。
            num_heads：每个 block 的注意力头数，默认 4。
            mlp_ratio：每个 block 的 MLP 扩展倍率，默认 4。
        返回：
            None；得到可通过 model(x_t,t) 调用的 nn.Module。
        """
        super().__init__()
        _validate_layout(image_size, patch_size, in_channels)
        if hidden_size < 2 or hidden_size % 2 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be even and divisible by positive num_heads")
        if depth <= 0 or mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        # M3-5a：按以下名称注册各模块，便于检查、保存和后续复用。
        # patch_projection：Linear(F,D)，把原始 patch 像素投影到隐藏空间。
        # pos_embed：可学习的 [1,N,D] nn.Parameter，按位置广播到 B 张图。
        #   用小幅随机值初始化以区分位置；不要每次 forward 都重新生成。
        # time_mlp：Linear(D,4D) → SiLU → Linear(4D,D)，处理固定时间编码。
        # blocks：nn.ModuleList，包含 depth 个独立的 TransformerBlock，不能用普通 list。
        # final_norm：LayerNorm(D)；output_projection：带 bias 的 Linear(D,F)。
        # M3 保留普通 Linear 的默认初始化；adaLN-Zero 等初始化留到 M7。
        self.patch_projection = nn.Linear(self.patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_size) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, self.patch_dim)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """功能：结合带噪图片、位置和时间步信息，预测每个像素的噪声。

        参数：
            x_t：[B,C,S,S] float32 带噪图片，C/S 与构造参数一致；无需裁剪像素。
            t：[B] 非负 int64 时间步，与 x_t 同设备；batch 内可以各不相同。
                本模型不绑定日程长度 T，调用方负责不超过所用日程的范围。
        返回：
            epsilon_pred：[B,C,S,S] 噪声预测，与 x_t 同形状；不是干净图片。
            不加 sigmoid/tanh 或 clamp，因为噪声可以超出图片范围。
        """
        if (
            x_t.ndim != 4
            or tuple(x_t.shape[1:]) != (self.in_channels, self.image_size, self.image_size)
            or x_t.dtype != torch.float32
            or x_t.shape[0] <= 0
        ):
            raise ValueError("x_t must match the configured [B,C,S,S] float32 shape")
        if t.shape != (x_t.shape[0],) or t.dtype != torch.int64 or t.device != x_t.device:
            raise ValueError("t must be int64 [B] on x_t.device")
        if t.min().item() < 0:
            raise ValueError("t must be nonnegative")
        # M3-5b：串联 M3-1 至 M3-4 的组件。
        # 图片路径：[B,C,S,S] → patchify → [B,N,F] → patch_projection → [B,N,D]。
        # 时间路径：t → timestep_embedding(D) → time_mlp → [B,D]，增加 token 单例轴。
        # 将位置 [1,N,D] 与时间 [B,1,D] 条件加到图片 token 上，再依次经过各 block。
        # 输出路径：final_norm → output_projection 得 [B,N,F] → unpatchify。
        # 不要把 [B,N,D] 直接还原图片，也不要调用 M2 的 q_sample 来替代预测网络。
        tokens = self.patch_projection(patchify(x_t, self.patch_size))
        time_condition = self.time_mlp(timestep_embedding(t, self.hidden_size)).unsqueeze(1)
        tokens = tokens + self.pos_embed + time_condition
        for block in self.blocks:
            tokens = block(tokens)
        predicted_patches = self.output_projection(self.final_norm(tokens))
        return unpatchify(
            predicted_patches,
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

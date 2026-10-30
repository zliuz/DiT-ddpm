"""M3 独立验收：核心 TODO 未完成时明确失败，不跳过或隐式代写网络。"""

import pytest
import torch

from ddpm.model import TinyDiT, TransformerBlock, patchify, timestep_embedding, unpatchify


def test_patchify_known_pixel_order() -> None:
    """功能：用有编号的像素检查块顺序。参数：无。返回：None。"""
    images = torch.arange(16).reshape(1, 1, 4, 4)
    expected = torch.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]])
    assert torch.equal(patchify(images, 2), expected)


def test_patchify_channel_is_last_inside_patch() -> None:
    """功能：检查双通道块内特征顺序。参数：无。返回：None。"""
    images = torch.arange(32).reshape(1, 2, 4, 4)
    actual = patchify(images, 2)
    assert actual.shape == (1, 4, 8)
    assert torch.equal(actual[0, 0], torch.tensor([0, 16, 1, 17, 4, 20, 5, 21]))


def test_unpatchify_known_pixels_without_using_patchify() -> None:
    """功能：独立检查还原函数，避免两个相反错误互相抵消。参数：无。返回：None。"""
    patches = torch.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]])
    actual = unpatchify(patches, image_size=4, patch_size=2, in_channels=1)
    assert torch.equal(actual, torch.arange(16).reshape(1, 1, 4, 4))


def test_patch_round_trip_keeps_values_and_gradients() -> None:
    """功能：检查批量、多通道往返与梯度连接。参数：无。返回：None。"""
    images = torch.arange(96, dtype=torch.float32).reshape(3, 2, 4, 4).requires_grad_()
    restored = unpatchify(patchify(images, 2), image_size=4, patch_size=2, in_channels=2)
    assert torch.equal(restored, images)
    restored.sum().backward()
    assert torch.equal(images.grad, torch.ones_like(images))


@pytest.mark.parametrize("patch_size", [0, 3])
def test_patchify_invalid_size(patch_size: int) -> None:
    """功能：检查非法块尺寸。参数：patch_size 为零或不能整除边长 4。返回：None。"""
    with pytest.raises(ValueError):
        patchify(torch.zeros(1, 1, 4, 4), patch_size)


def test_time_embedding_at_zero() -> None:
    """功能：检查 t=0 的 cos/sin 排列。参数：无。返回：None。"""
    actual = timestep_embedding(torch.tensor([0]), dim=4)
    assert actual.dtype == torch.float32 and actual.shape == (1, 4)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_time_embedding_reference_frequencies() -> None:
    """功能：用频率 1、0.1 的手算值检查编码。参数：无。返回：None。"""
    actual = timestep_embedding(torch.tensor([1]), dim=4, max_period=100)
    expected = torch.tensor([[0.5403023, 0.9950042, 0.8414710, 0.0998334]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dim", [0, 3])
def test_time_embedding_invalid_dimension(dim: int) -> None:
    """功能：检查编码维度限制。参数：dim 为零或奇数。返回：None。"""
    with pytest.raises(ValueError):
        timestep_embedding(torch.tensor([0]), dim=dim)


def test_block_zero_branches_keep_residual() -> None:
    """功能：两支路权重归零时应保留输入残差。参数：无。返回：None。"""
    block = TransformerBlock(hidden_size=8, num_heads=2, mlp_ratio=2)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    tokens = torch.randn(2, 5, 8)
    torch.testing.assert_close(block(tokens), tokens)


def test_block_does_not_mix_images_in_batch() -> None:
    """功能：修改第二张图不能影响第一张图。参数：无。返回：None。"""
    torch.manual_seed(0)
    block = TransformerBlock(hidden_size=8, num_heads=2, mlp_ratio=2).eval()
    tokens = torch.randn(2, 5, 8)
    changed = tokens.clone()
    changed[1] = torch.randn_like(changed[1]) * 5
    with torch.no_grad():
        torch.testing.assert_close(block(tokens)[0], block(changed)[0])


def test_block_attention_is_bidirectional() -> None:
    """功能：后面的 token 应能影响第一个 token。参数：无。返回：None。"""
    torch.manual_seed(0)
    block = TransformerBlock(hidden_size=8, num_heads=2, mlp_ratio=2).eval()
    tokens = torch.randn(1, 5, 8)
    changed = tokens.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1]) * 5
    with torch.no_grad():
        assert not torch.allclose(block(tokens)[:, 0], block(changed)[:, 0])


def test_block_backward_is_finite() -> None:
    """功能：检查注意力与 MLP 参与反向传播。参数：无。返回：None。"""
    block = TransformerBlock(hidden_size=8, num_heads=2, mlp_ratio=2)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    prediction = block(tokens)
    assert prediction.shape == tokens.shape
    prediction.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert sum(parameter.numel() for parameter in block.parameters()) > 0
    for parameter in block.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_model_output_shape(batch_size: int) -> None:
    """功能：检查不同批大小及双通道。参数：batch_size 为 1 或 3。返回：None。"""
    model = TinyDiT(image_size=8, patch_size=2, in_channels=2, hidden_size=16, depth=2)
    x_t = torch.randn(batch_size, 2, 8, 8)
    actual = model(x_t, torch.arange(batch_size, dtype=torch.int64))
    assert actual.shape == x_t.shape and actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


def test_model_uses_time_condition() -> None:
    """功能：相同图片输入不同 t，应有不同预测。参数：无。返回：None。"""
    torch.manual_seed(0)
    model = TinyDiT(image_size=8, patch_size=2, hidden_size=16, depth=2).eval()
    x_t = torch.randn(1, 1, 8, 8).repeat(2, 1, 1, 1)
    with torch.no_grad():
        prediction = model(x_t, torch.tensor([0, 999]))
    assert not torch.allclose(prediction[0], prediction[1])


def test_model_backward_reaches_registered_parameters() -> None:
    """功能：所有注册的可训练参数都应参与梯度计算。参数：无。返回：None。"""
    model = TinyDiT(image_size=8, patch_size=2, hidden_size=16, depth=2)
    prediction = model(torch.randn(2, 1, 8, 8), torch.tensor([0, 9]))
    prediction.square().mean().backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_model_output_is_not_clipped() -> None:
    """功能：输出头设为常数 2，验证未加 sigmoid/tanh/clamp。参数：无。返回：None。"""
    model = TinyDiT(image_size=8, patch_size=2, hidden_size=16, depth=2)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(2)
    actual = model(torch.zeros(1, 1, 8, 8), torch.tensor([0]))
    torch.testing.assert_close(actual, torch.full((1, 1, 8, 8), 2.0))


@pytest.mark.parametrize(
    "kwargs", [{"patch_size": 3}, {"hidden_size": 10, "num_heads": 4}, {"depth": 0}]
)
def test_model_invalid_configuration(kwargs: dict) -> None:
    """功能：检查模型配置限制。参数：kwargs 为不合法的构造参数。返回：None。"""
    with pytest.raises(ValueError):
        TinyDiT(**kwargs)

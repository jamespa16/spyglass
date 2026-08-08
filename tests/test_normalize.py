import numpy as np

from spyglass.normalize import downsample_block_mean, to_uint8


def test_to_uint8_known_values():
    clip = 0.5
    arr = np.array([-clip, 0.0, clip], dtype=np.float32)
    result = to_uint8(arr, clip)
    assert result.tolist() == [0, 128, 255]


def test_to_uint8_clips_outliers():
    clip = 1.0
    arr = np.array([-10.0, 10.0], dtype=np.float32)
    result = to_uint8(arr, clip)
    assert result.tolist() == [0, 255]


def test_to_uint8_degenerate_clip_returns_flat_gray():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = to_uint8(arr, 0.0)
    assert np.all(result == 128)
    result = to_uint8(arr, float("nan"))
    assert np.all(result == 128)


def test_downsample_block_mean_no_op_when_small_enough():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = downsample_block_mean(arr, max_dim=8)
    assert result is arr


def test_downsample_block_mean_averages_blocks():
    arr = np.array(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 6.0, 6.0], [4.0, 4.0, 6.0, 6.0]],
        dtype=np.float32,
    )
    result = downsample_block_mean(arr, max_dim=2)
    assert result.shape == (2, 2)
    assert result.tolist() == [[0.0, 2.0], [4.0, 6.0]]


def test_downsample_block_mean_pads_uneven_shapes():
    arr = np.ones((3, 5), dtype=np.float32)
    result = downsample_block_mean(arr, max_dim=2)
    assert result.shape == (2, 2)
    assert np.allclose(result, 1.0)


def test_downsample_block_mean_handles_leading_tile_axis():
    # A 3D MoE-style stack (n_expert, rows, cols): each expert's slice
    # downsamples independently, same as the 2D case, and the expert axis
    # is left untouched.
    tile = np.array(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 6.0, 6.0], [4.0, 4.0, 6.0, 6.0]],
        dtype=np.float32,
    )
    arr = np.stack([tile, tile + 10.0, tile + 20.0])
    result = downsample_block_mean(arr, max_dim=2)

    assert result.shape == (3, 2, 2)
    assert result[0].tolist() == [[0.0, 2.0], [4.0, 6.0]]
    assert result[1].tolist() == [[10.0, 12.0], [14.0, 16.0]]
    assert result[2].tolist() == [[20.0, 22.0], [24.0, 26.0]]

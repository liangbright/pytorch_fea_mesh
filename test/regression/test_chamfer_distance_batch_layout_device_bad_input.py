"""chamfer_distance.py: batch normalization, coincident points, memory layout, CUDA device and bad input.

A test named test_unchanged_... checks behaviour the fix kept, so it also passes on the file before the fix.
Every other test should fail on that file.
"""
import pytest
import torch

pool = pytest.importorskip('torch_geometric.nn.pool')
# Only skip an explicitly missing optional backend; runtime errors should fail.
try:
    pool.knn(torch.zeros((1, 3)), torch.zeros((1, 3)), 1)
except ImportError as error:
    pytest.skip(str(error), allow_module_level=True)

from torch_fea_mesh.utility import chamfer_distance as m

pytestmark = pytest.mark.chamfer

F32, F64 = torch.float32, torch.float64
INF, NAN = float('inf'), float('nan')
SINGLE = [m.cal_distance, m.cal_chamfer_distance]
BATCH = [m.cal_distance_batch, m.cal_chamfer_distance_batch]


def true_distance(query, reference):
    """Squared distance from each query point to its nearest reference point, by comparing all pairs."""
    q, r = query.detach().cpu().double(), reference.detach().cpu().double()
    return ((q[:, None] - r[None]) ** 2).sum(-1).min(dim=1).values


def assert_true_single(P1, P2, **kwargs):
    d1 = m.cal_distance(P1, P2, 'none', True, **kwargs)
    c1, c2 = m.cal_chamfer_distance(P1, P2, 'none', True, **kwargs)
    t1, t2 = true_distance(P1, P2), true_distance(P2, P1)
    torch.testing.assert_close(d1.double().cpu(), t1, rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(c1.double().cpu(), t1, rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(c2.double().cpu(), t2, rtol=1e-4, atol=1e-12)


def assert_true_batch(P1, P2, **kwargs):
    d1 = m.cal_distance_batch(P1, P2, 'none', True, **kwargs)
    c1, c2 = m.cal_chamfer_distance_batch(P1, P2, 'none', True, **kwargs)
    t1 = torch.stack([true_distance(a, b) for a, b in zip(P1, P2)])
    t2 = torch.stack([true_distance(b, a) for a, b in zip(P1, P2)])
    torch.testing.assert_close(d1.double().cpu(), t1, rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(c1.double().cpu(), t1, rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(c2.double().cpu(), t2, rtol=1e-4, atol=1e-12)


def test_batch_examples_do_not_affect_each_other():
    # DIST-2, the reported float32 example
    near = torch.tensor([[0., 0, 0], [1, 0, 0]])
    far = torch.tensor([[1e8, 0, 0], [1e8 + 32, 0, 0]])
    assert_true_batch(torch.stack([near, far]), torch.stack([near.flip(0), far.flip(0)]))
    # the (B,N,D) branch of scale gives each example what it would get alone
    x, y = torch.rand(2, 5, 3, dtype=F64), torch.rand(2, 7, 3, dtype=F64)
    x[1] += 100
    y[1] *= 10
    sx, sy = m.scale(x, y)
    for k in range(2):
        ax, ay = m.scale(x[k], y[k])
        torch.testing.assert_close(sx[k], ax)
        torch.testing.assert_close(sy[k], ay)


def test_coincident_points_give_zero():
    # DIST-3. One point used to give NaN, and a power-of-2 count always raised.
    for n in (1, 4):
        p = torch.tensor([[.5, .5, .5]]).repeat(n, 1)
        for function, q in [(f, p) for f in SINGLE] + [(f, p[None]) for f in BATCH]:
            for reduction in ('mean', 'sum'):
                assert function(q, q.clone(), reduction, True) == 0
        assert m.cal_distance(p, p.clone(), 'none', True).shape == (n,)
        assert m.cal_distance_batch(p[None], p[None].clone(), 'none', True).shape == (1, n)
    # one coincident example next to an ordinary one
    other = torch.rand(4, 3)
    x = torch.stack([torch.zeros(4, 3), other])
    y = torch.stack([torch.zeros(5, 3), torch.rand(5, 3)])
    assert_true_batch(x, y)


def test_batch_functions_accept_any_memory_layout():
    # DIST-4. Each of these raised from view.
    torch.manual_seed(0)
    big = torch.rand(4, 6, 3)
    wide = torch.rand(4, 3, 6)
    for p in (wide.transpose(1, 2), big[:, :2], big[:, 1:], big[::2]):
        assert not p.is_contiguous()
        q = torch.rand(p.shape[0], 9, 3)
        assert_true_batch(p, q)
        assert_true_batch(q, p)


def test_memory_layout_does_not_change_the_neighbours():
    # knn ignores strides: these gave wrong neighbours with no error
    torch.manual_seed(1)
    permuted, other = torch.rand(3, 300).T, torch.rand(200, 3)
    base = torch.rand(600, 4)
    for flag in (True, False):
        assert_true_single(permuted, other, scale_input=flag)
        assert_true_single(other, permuted, scale_input=flag)
        assert_true_batch(torch.rand(1, 3, 300).transpose(1, 2), other[None], scale_input=flag)
    # slices reach knn as they are only with scale_input=False
    for p in (base[::2, :3], base[:, :3]):
        assert_true_single(p, other, scale_input=False)
        assert_true_single(other, p, scale_input=False)
    assert_true_batch(other[None], torch.rand(2, 300, 3)[:, ::2][:1], scale_input=False)


def test_cuda_input_on_any_device():
    # knn runs on the current CUDA device, so input on another device used to fail with an illegal memory access
    torch.manual_seed(3)
    x, y = torch.rand(2, 50, 3, dtype=F64), torch.rand(2, 70, 3, dtype=F64)
    current = torch.cuda.current_device()
    for index in range(torch.cuda.device_count()):
        device = torch.device('cuda', index)
        for flag in (True, False):
            assert_true_single(x[0].to(device), y[0].to(device), scale_input=flag)
            assert_true_batch(x.to(device), y.to(device), scale_input=flag)
        leaf = x[0].to(device).requires_grad_()
        value = m.cal_chamfer_distance(leaf, y[0].to(device), 'mean', True)
        gradient, = torch.autograd.grad(value, leaf)
        assert value.device == device and gradient.device == device and gradient.abs().sum() > 0
        assert torch.cuda.current_device() == current
    with pytest.raises(ValueError, match='same device'):
        m.cal_distance(x[0], y[0].cuda(), 'mean', True)


def test_bad_input_raises_a_clear_error():
    some, empty = torch.rand(3, 3), torch.zeros(0, 3)
    for flag in (True, False):
        for function in SINGLE:
            # an empty cloud used to crash python with scale_input=False
            for a, b in ((empty, some), (some, empty), (empty, empty)):
                with pytest.raises(ValueError, match='empty'):
                    function(a, b, 'none', True, flag)
            # 3D input used to return a wrong result without an error
            with pytest.raises(ValueError, match='dimensions'):
                function(torch.rand(2, 5, 3), torch.rand(2, 5, 3), 'none', True, flag)
            with pytest.raises(ValueError, match='shape'):
                function(some, torch.rand(3, 2), 'none', True, flag)
            # a NaN in P2 used to be skipped without an error when scale_input=False
            for value in (NAN, INF):
                bad = some.clone()
                bad[0, 0] = value
                for a, b in ((bad, some), (some, bad)):
                    with pytest.raises(ValueError, match='NaN or inf'):
                        function(a, b, 'none', True, flag)
        for function in BATCH:
            with pytest.raises(ValueError, match='empty'):
                function(empty[None], some[None], 'none', True, flag)
            # 2D input used to be read as batches of 1D points
            with pytest.raises(ValueError, match='dimensions'):
                function(some, some.clone(), 'none', True, flag)
            with pytest.raises(ValueError, match='shape'):
                function(torch.rand(2, 3, 3), torch.rand(1, 3, 3), 'none', True, flag)
    with pytest.raises(ValueError, match='dimensions'):
        m.scale(torch.rand(5), torch.rand(5))


def test_no_neighbour_from_knn_raises():
    # the squared distance overflows, so knn finds nothing: one point used to give NaN through broadcasting
    p = torch.tensor([[1e200, 0, 0]], dtype=F64)
    for function, a, b in ((m.cal_distance, p, -p), (m.cal_distance_batch, p[None], -p[None])):
        with pytest.raises(RuntimeError, match='neighbour'):
            function(a, b, 'mean', True, False)
    assert m.cal_distance(p, -p, 'mean', True) == INF
    # float32 reaches this at a lower size
    q = torch.tensor([[1e30, 0, 0]], dtype=F32)
    with pytest.raises(RuntimeError, match='neighbour'):
        m.cal_distance(q, -q, 'mean', True, False)


def test_unchanged_ordinary_input():
    torch.manual_seed(4)
    for dtype in (F32, F64):
        for dim in (2, 3):
            x, y = torch.rand(3, 50, dim, dtype=dtype), torch.rand(3, 70, dim, dtype=dtype)
            x0, y0 = x.clone(), y.clone()
            assert_true_batch(x, y)
            assert_true_single(x[0], y[0])
            d1, d2 = m.cal_chamfer_distance_batch(x, y, 'none', True)
            assert m.cal_chamfer_distance_batch(x, y, 'mean', True) == 0.5 * (d1.mean() + d2.mean())
            assert m.cal_chamfer_distance_batch(x, y, 'sum', True) == d1.sum() + d2.sum()
            torch.testing.assert_close(m.cal_distance_batch(x, y, 'none', False), m.cal_distance_batch(x, y, 'none', True).sqrt())
            assert (x == x0).all() and (y == y0).all()
    leaf = torch.tensor([[0., 0, 0], [2, 0, 0]], requires_grad=True)
    target = torch.tensor([[0., 1, 0], [2, 1, 0]])
    gradient, = torch.autograd.grad(m.cal_chamfer_distance(leaf, target, 'mean', True), leaf)
    torch.testing.assert_close(gradient, torch.tensor([[0., -1, 0], [0, -1, 0]]))
    with pytest.raises(ValueError, match='unknown reduction'):
        m.cal_distance(x[0], y[0], 'median', True)

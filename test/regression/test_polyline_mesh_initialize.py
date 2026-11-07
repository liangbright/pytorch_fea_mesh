"""PolylineMesh.initialize and from_curve: the forms of input they take, and what they do with anything else.

Keep this file small: add a test only for a behaviour nothing here catches.
"""
import warnings

import numpy as np
import pytest
import torch

from torch_fea_mesh.mesh.polyline_mesh import PolylineMesh

pytestmark = [pytest.mark.geometry]

F32, F64 = torch.float32, torch.float64
P = torch.tensor([[0.1, 0., 0.], [1.1, 0., 0.], [2.1, 1., 0.]], dtype=F64)
Q = P[:2]+10


def build(curve):
    """Build by initialize and by from_curve, which must agree."""
    mesh = PolylineMesh()
    mesh.initialize(curve=curve)
    other = PolylineMesh.from_curve(curve)
    assert other.node.dtype == mesh.node.dtype
    assert torch.equal(other.node, mesh.node)
    assert other.element == mesh.element
    return mesh


def test_every_form_of_input_gives_the_same_mesh():
    one_curve = [P.tolist(), tuple(P.tolist()), P, P.numpy(), list(P), tuple(P), list(P.numpy())]
    two_curves = [[P, Q], [P.numpy(), Q.numpy()], [P.tolist(), Q.tolist()], [list(P), list(Q)],
                  [list(P.numpy()), list(Q.numpy())], [[], P.tolist(), Q.tolist()], (P, Q)]
    same_length = [torch.stack([P, P+10]), np.stack([P.numpy(), P.numpy()+10])]
    for curves, node, element in ((one_curve, P, [[0, 1], [1, 2]]),
                                  (two_curves, torch.cat([P, Q]), [[0, 1], [1, 2], [3, 4]]),
                                  (same_length, torch.cat([P, P+10]), [[0, 1], [1, 2], [3, 4], [4, 5]])):
        for curve in curves:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                mesh = build(curve)
            assert mesh.node.dtype == F64
            assert torch.equal(mesh.node, node)
            assert mesh.element == element


def test_node_has_the_widest_dtype_of_the_curves():
    P32, Q32 = P.to(F32), Q.to(F32)
    for curve, dtype in (([P, Q32], F64), ([P32, Q], F64), ([P32, Q32], F32), (P32, F32), (P32.numpy(), F32),
                         ([P32, Q.tolist()], F64), ([P.tolist(), Q32], F64), ([P32.numpy(), Q], F64), ([P.numpy(), Q32], F64),
                         ([[], P32], F32), ([P32, []], F32), ([torch.empty(0, 3, dtype=F64), P32], F32)):
        assert build(curve).node.dtype == dtype
    # The float64 curve is not rounded, whichever curve comes last.
    assert torch.equal(build([P, Q32]).node[:3], P)


def test_bad_input_raises_value_error_and_keeps_the_mesh():
    mesh = PolylineMesh.from_curve(P)
    for curve, message in (({'a': 1}, 'unsupported type'), ({}, 'unsupported type'), ('abc', 'unsupported type'), ('', 'unsupported type'),
                           (None, 'unsupported type'), (5, 'unsupported type'), ([{'a': 1}], 'unsupported type'),
                           (P[0], 'must be 2D or 3D'), (P.numpy()[0], 'must be 2D or 3D'), (P.reshape(1, 1, 3, 3), 'must be 2D or 3D'),
                           ([[], P[0]], 'made of points'), ([[], [1., 2., 3.]], 'made of points')):
        with pytest.raises(ValueError, match=message):
            PolylineMesh.from_curve(curve)
        with pytest.raises(ValueError, match=message):
            mesh.initialize(curve)
        assert torch.equal(mesh.node, P)


def test_ctrl_c_is_not_swallowed():
    class Interrupting:
        def __getitem__(self, k):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        PolylineMesh.from_curve([Interrupting()])


def test_nothing_gives_an_empty_mesh_and_one_point_gives_a_node():
    for curve in ([], (), [[]], np.array([]), torch.empty(0), torch.empty(0, 3), torch.empty(1, 0, 3)):
        assert PolylineMesh.from_curve(curve).node is None
        mesh = PolylineMesh.from_curve(P)
        mesh.initialize(curve)
        assert mesh.node is None
    for curve in (P[:1].tolist(), P[:1], [P[0]]):
        mesh = build(curve)
        assert torch.equal(mesh.node, P[:1])
        assert mesh.element == []

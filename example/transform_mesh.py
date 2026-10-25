"""Rotate a point through ninety degrees about the z axis."""
import torch
from torch_fea_mesh.transform.rotation_transform import RotationTransform3D


def main():
    point = torch.tensor([[1.,0.,0.]], dtype=torch.float64)
    angle = torch.tensor([0.,0.,torch.pi/2], dtype=torch.float64)
    print(RotationTransform3D.rotate(point, angle))


if __name__ == '__main__':
    main()

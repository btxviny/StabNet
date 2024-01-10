import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dense_warp(image, flow):
    """
    Densely warps an image using optical flow.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
        flow (torch.Tensor): Optical flow tensor of shape (batch_size, 2, height, width).

    Returns:
        torch.Tensor: Warped image tensor of shape (batch_size, channels, height, width).
    """
    batch_size, channels, height, width = image.size()

    # Generate a grid of pixel coordinates based on the optical flow
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width),indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).to(image.device)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    new_grid = grid + flow.permute(0, 2, 3, 1)

    # Normalize the grid coordinates between -1 and 1
    new_grid /= torch.tensor([width - 1, height - 1], dtype=torch.float32, device=image.device)
    new_grid = new_grid * 2 - 1
    # Perform the dense warp using grid_sample
    warped_image = F.grid_sample(image, new_grid, align_corners=False)

    return warped_image


def find_homography_numpy(src_points, dst_points):
    A = []
    B = []
    for src, dst in zip(src_points, dst_points):
        x, y = src
        x_prime, y_prime = dst
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        B.extend([-x_prime, -y_prime])
    A = np.array(A)
    B = np.array(B)
    ATA = np.dot(A.T, A)
    eigenvalues, eigenvectors = np.linalg.eigh(ATA)
    min_eigenvalue_index = np.argmin(eigenvalues)
    homography_vector = eigenvectors[:, min_eigenvalue_index]
    homography_vector /= homography_vector[-1] 
    homography_matrix = np.reshape(homography_vector,(3, 3))
    
    return homography_matrix

import torch

def warp(img, mat):
    device = img.device
    mat = torch.cat([mat,torch.ones((mat.size(0),1),device = device)], axis = -1).view(-1,3,3)
    batch_size, channels, height, width = img.shape
    cy, cx = height // 2, width // 2

    # Compute the translation matrix to shift the center to the origin
    translation_matrix1 = torch.tensor([[1, 0, -cx],
                                        [0, 1, -cy],
                                        [0, 0, 1]], dtype=torch.float32, device=device)
    translation_matrix1 = translation_matrix1.repeat(batch_size, 1, 1)

    # Compute the translation matrix to shift the origin back to the center
    translation_matrix2 = torch.tensor([[1, 0, cx],
                                        [0, 1, cy],
                                        [0, 0, 1]], dtype=torch.float32, device=device)
    translation_matrix2 = translation_matrix2.repeat(batch_size, 1, 1)
    transformation_matrix = torch.matmul(translation_matrix2, torch.matmul(mat, translation_matrix1))

    # Compute the grid coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))
    coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1).float()
    coords = coords.view(1, -1, 3).repeat(batch_size, 1, 1)

    # Apply the transformation matrix to the grid coordinates
    transformed_coords = torch.matmul(coords, transformation_matrix.transpose(1, 2))

    # Normalize the transformed coordinates
    x_transformed = transformed_coords[:, :, 0] / transformed_coords[:, :, 2]
    y_transformed = transformed_coords[:, :, 1] / transformed_coords[:, :, 2]

    # Reshape the transformed coordinates to match the image size
    x_transformed = x_transformed.view(batch_size, height, width)
    y_transformed = y_transformed.view(batch_size, height, width)

    # Normalize the grid coordinates to the range [-1, 1]
    x_normalized = (x_transformed / (width - 1)) * 2 - 1
    y_normalized = (y_transformed / (height - 1)) * 2 - 1

    # Perform bilinear interpolation using grid_sample
    grid = torch.stack([x_normalized, y_normalized], dim=-1)
    warped_image = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=False ,padding_mode='zeros')
    
    return warped_image


def find_homography(src_points, dst_points):
    device = src_points.device
    A = []
    B = []
    # Convert input lists to PyTorch tensors
    src_points = torch.tensor(src_points, dtype=torch.float32)
    dst_points = torch.tensor(dst_points, dtype=torch.float32)
    for src, dst in zip(src_points, dst_points):
        x, y = src
        x_prime, y_prime = dst
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        B.extend([-x_prime, -y_prime])
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    # Calculate ATA matrix
    ATA = torch.matmul(A.T, A)
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(ATA)
    # Find the index of the smallest eigenvalue
    min_eigenvalue_index = torch.argmin(eigenvalues)
    # Extract the corresponding eigenvector
    homography_vector = eigenvectors[:, min_eigenvalue_index]
    # Normalize homography vector
    homography_vector = homography_vector /  homography_vector[-1]
    # Reshape to obtain the homography matrix
    homography_matrix = homography_vector.view(3, 3)
    return homography_matrix.to(device)


def findHomography(grids, new_grids_loc):
    """
    @param: grids the location of origin grid vertices [2, H, W]
    @param: new_grids_loc the location of desired grid vertices [2, H, W]

    @return: homo_t homograph projection matrix for each grid [3, 3, H-1, W-1]
    """

    _, H, W = grids.shape

    new_grids = new_grids_loc.unsqueeze(0)

    Homo = torch.zeros(1, 3, 3, H-1, W-1).to(grids.device)

    grids = grids.unsqueeze(0)

    try:
        # for common cases if all the homograph can be calculated
        one = torch.ones_like(grids[:, 0:1, :-1, :-1], device=grids.device)
        zero = torch.zeros_like(grids[:, 1:2, :-1, :-1], device=grids.device)

        A = torch.cat([
            torch.stack([grids[:, 0:1, :-1, :-1], grids[:, 1:2, :-1, :-1], one, zero, zero, zero,
                         -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 0:1, :-1, :-1], -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 0:1, :-1, :-1]], 2),  # 1, 1, 8, h-1, w-1
            torch.stack([grids[:, 0:1, 1:, :-1], grids[:, 1:2, 1:, :-1], one, zero, zero, zero,
                         -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 0:1, 1:, :-1], -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 0:1, 1:, :-1]], 2),
            torch.stack([grids[:, 0:1, :-1, 1:], grids[:, 1:2, :-1, 1:], one, zero, zero, zero,
                         -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 0:1, :-1, 1:], -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 0:1, :-1, 1:]], 2),
            torch.stack([grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:, 1:], one, zero, zero, zero,
                         -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 0:1, 1:, 1:], -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 0:1, 1:, 1:]], 2),
            torch.stack([zero, zero, zero, grids[:, 0:1, :-1, :-1], grids[:, 1:2, :-1, :-1], one,
                         -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 1:2, :-1, :-1], -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 1:2, :-1, :-1]], 2),
            torch.stack([zero, zero, zero, grids[:, 0:1, 1:, :-1], grids[:, 1:2, 1:, :-1], one,
                         -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 1:2, 1:, :-1], -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 1:2, 1:, :-1]], 2),
            torch.stack([zero, zero, zero, grids[:, 0:1, :-1, 1:], grids[:, 1:2, :-1, 1:], one,
                         -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 1:2, :-1, 1:], -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 1:2, :-1, 1:]], 2),
            torch.stack([zero, zero, zero, grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:, 1:], one,
                         -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 1:2, 1:, 1:], -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 1:2, 1:, 1:]], 2),
        ], 1).view(8, 8, -1).permute(2, 0, 1)  # 1, 8, 8, h-1, w-1
        B_ = torch.stack([
            new_grids[:, 0, :-1, :-1],
            new_grids[:, 0, 1:, :-1],
            new_grids[:, 0, :-1, 1:],
            new_grids[:, 0, 1:, 1:],
            new_grids[:, 1, :-1, :-1],
            new_grids[:, 1, 1:, :-1],
            new_grids[:, 1, :-1, 1:],
            new_grids[:, 1, 1:, 1:],
        ], 1).view(8, -1).permute(1, 0)  # B, 8, h-1, w-1 ==> A @ H = B ==> H = A^-1 @ B
        A_inverse = torch.inverse(A)
        # B, 8, 8 @ B, 8, 1 --> B, 8, 1
        H_recovered = torch.bmm(A_inverse, B_.unsqueeze(2))

        H_ = torch.cat([H_recovered, torch.ones_like(
            H_recovered[:, 0:1, :], device=H_recovered.device)], 1).view(H_recovered.shape[0], 3, 3)

        H_ = H_.permute(1, 2, 0)
        H_ = H_.view(Homo.shape)
        Homo = H_
    except:
        # if some of the homography can not be calculated
        one = torch.ones_like(grids[:, 0:1, 0, 0], device=grids.device)
        zero = torch.zeros_like(grids[:, 1:2, 0, 0], device=grids.device)
        H_ = torch.eye(3, device=grids.device)
        for i in range(H - 1):
            for j in range(W - 1):
                A = torch.cat([
                    torch.stack([grids[:, 0:1, i, j], grids[:, 1:2, i, j], one, zero, zero, zero,
                                 -1 * grids[:, 0:1, i, j] * new_grids[:, 0:1, i, j], -1 * grids[:, 1:2, i, j] * new_grids[:, 0:1, i, j]], 2),
                    torch.stack([grids[:, 0:1, i+1, j], grids[:, 1:2, i+1, j], one, zero, zero, zero,
                                 -1 * grids[:, 0:1, i+1, j] * new_grids[:, 0:1, i+1, j], -1 * grids[:, 1:2, i+1, j] * new_grids[:, 0:1, i+1, j]], 2),
                    torch.stack([grids[:, 0:1, i, j+1], grids[:, 1:2, i, j+1], one, zero, zero, zero,
                                 -1 * grids[:, 0:1, i, j+1] * new_grids[:, 0:1, i, j+1], -1 * grids[:, 1:2, i, j+1] * new_grids[:, 0:1, i, j+1]], 2),
                    torch.stack([grids[:, 0:1, i+1, j+1], grids[:, 1:2, i+1, j+1], one, zero, zero, zero,
                                 -1 * grids[:, 0:1, i+1, j+1] * new_grids[:, 0:1, i+1, j+1], -1 * grids[:, 1:2, i+1, j+1] * new_grids[:, 0:1, i+1, j+1]], 2),
                    torch.stack([zero, zero, zero, grids[:, 0:1, i, j], grids[:, 1:2, i, j], one,
                                 -1 * grids[:, 0:1, i, j] * new_grids[:, 1:2, i, j], -1 * grids[:, 1:2, i, j] * new_grids[:, 1:2, i, j]], 2),
                    torch.stack([zero, zero, zero, grids[:, 0:1, i+1, j], grids[:, 1:2, i+1, j], one,
                                 -1 * grids[:, 0:1, i+1, j] * new_grids[:, 1:2, i+1, j], -1 * grids[:, 1:2, i+1, j] * new_grids[:, 1:2, i+1, j]], 2),
                    torch.stack([zero, zero, zero, grids[:, 0:1, i, j+1], grids[:, 1:2, i, j+1], one,
                                 -1 * grids[:, 0:1, i, j+1] * new_grids[:, 1:2, i, j+1], -1 * grids[:, 1:2, i, j+1] * new_grids[:, 1:2, i, j+1]], 2),
                    torch.stack([zero, zero, zero, grids[:, 0:1, i+1, j+1], grids[:, 1:2, i+1, j+1], one,
                                 -1 * grids[:, 0:1, i+1, j+1] * new_grids[:, 1:2, i+1, j+1], -1 * grids[:, 1:2, i+1, j+1] * new_grids[:, 1:2, i+1, j+1]], 2),
                ], 1)  # B, 8, 8
                B_ = torch.stack([
                    new_grids[:, 0, i, j],
                    new_grids[:, 0, i+1, j],
                    new_grids[:, 0, i, j+1],
                    new_grids[:, 0, i+1, j+1],
                    new_grids[:, 1, i, j],
                    new_grids[:, 1, i+1, j],
                    new_grids[:, 1, i, j+1],
                    new_grids[:, 1, i+1, j+1],
                ], 1)  # B, 8 ==> A @ H = B ==> H = A^-1 @ B
                try:
                    A_inverse = torch.inverse(A)

                    # B, 8, 8 @ B, 8, 1 --> B, 8, 1
                    H_recovered = torch.bmm(A_inverse, B_.unsqueeze(2))

                    H_ = torch.cat([H_recovered, torch.ones_like(H_recovered[:, 0:1, :]).to(
                        H_recovered.device)], 1).view(H_recovered.shape[0], 3, 3)
                except:
                    pass
                Homo[:, :, :, i, j] = H_

    homo_t = Homo.view(3, 3, H-1, W-1)

    return homo_t
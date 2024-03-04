
import numpy as np
import torch
from tqdm import tqdm

def compute_jacob_ftmle(model, input_data, iteration, device):
    """
    Calculate ftmle based on singular value of jacob.
    Used by small network (e.g. FFESN) for accurate ftmle evaluation.

    - model: analyzed model
    - input_data: layer input
    - iteration: time of iteration
    """
    model.eval()
    model = model.to(device)
    input_data = input_data.to(device)
    input_data.requires_grad=True
    y_out = model(input_data)
        
    sample_dim = input_data.shape[0]
    input_dim = input_data.nelement() // input_data.shape[0]
    output_dim = y_out.nelement() // y_out.shape[0]
    
    jacobs = torch.zeros(sample_dim, output_dim, input_dim, device=device)
    for i in range(0, output_dim):
        vectors = torch.zeros(y_out.shape[0], output_dim).to(device)
        vectors[:, i] = 1.0
        vectors = vectors.view_as(y_out)

        if input_data.grad is not None:
            input_data.grad.data.zero_()
        
        grad = torch.autograd.grad(y_out, input_data, grad_outputs=vectors, 
                                   retain_graph=True)[0]
        jacobs[:, i] = grad.view(grad.shape[0], input_dim).detach()

    del input_data

    exp_temp = torch.zeros(sample_dim, device=device)
    ftmle_temp = torch.zeros(sample_dim, device=device)

    for i in range(sample_dim):
        _, s, _ = torch.linalg.svd(jacobs[i], full_matrices=False)  # Truncated SVD
        exp_temp[i] = s.max()
        ftmle_temp[i] = torch.log(exp_temp[i]) / iteration

    del jacobs
    
    return exp_temp.cpu().numpy(), ftmle_temp.cpu().numpy()

def compute_jacob_ftmle_low_mem(model, input_data, iteration, device, chunk_size=128, random_svd=False, k=5):
    """
    Calculate ftmle based on singular value of jacob, using chunked Jacobian calculation.
    Used by large network (e.g. CNN) to approach ftmle evaluation meanwhile saving memory and computation time.
    
    Parameters:
    - model: analyzed model
    - input_data: layer input
    - iteration: time of iteration
    - k: the numbe of singular value while computing random SVD.
    """

    model.eval()
    model = model.to(device)
    input_data = input_data.to(device)
    input_data.requires_grad=True
    y_out = model(input_data)

    sample_dim = input_data.shape[0]
    input_dim = input_data.nelement() // input_data.shape[0]
    output_dim = y_out.nelement() // y_out.shape[0]

    exp_temp = torch.zeros(sample_dim, device=device)

    for start in tqdm(range(0, output_dim, chunk_size)):
        end = min(start + chunk_size, output_dim)
        
        # Temporary tensor to store the Jacobian for the current chunk
        partial_jacob = torch.zeros(sample_dim, end-start, input_dim, device=device)

        # Calculate the Jacobian for the current chunk
        for idx, i in enumerate(range(start, end)):
            vectors = torch.zeros(sample_dim, output_dim).to(device)
            vectors[:, i] = 1.0
            vectors = vectors.view_as(y_out)
        
            if input_data.grad is not None:
                input_data.grad.data.zero_()

            grads = torch.autograd.grad(y_out, input_data, 
                                        grad_outputs=vectors, 
                                        retain_graph=True)[0]
            partial_jacob[:, idx] = grads.view(sample_dim, input_dim).detach()
            del grads
        
        for j in range(sample_dim):
            if random_svd:
                s = randomized_svd(partial_jacob[j], k=k)
            else:
                _, s, _ = torch.linalg.svd(partial_jacob[j], full_matrices=False)
            max_s = s.max()
            exp_temp[j] = torch.max(exp_temp[j], max_s)
            
        del partial_jacob

    ftmle_temp = torch.log(exp_temp) / iteration
    return exp_temp.cpu().numpy(), ftmle_temp.cpu().numpy()


def randomized_svd(A, k=10):
    """
    Compute the approximate SVD of matrix A using randomization.
    
    Parameters:
    - A: The matrix for which the SVD will be computed.
    - k: Number of singular values and vectors to compute.

    Returns:
    - U, s, V: Approximate SVD components.
    """
    # Sample a random matrix O
    O = torch.randn(A.size(1), k).to(A.device)
    
    # Form a matrix Y = A * O
    Y = torch.mm(A, O)
    
    # Orthonormalize Y using QR decomposition
    Q, _ = torch.linalg.qr(Y, 'reduced')
    
    # Form the matrix B = Q^T * A
    B = torch.mm(Q.t(), A)
    
    # Compute the SVD of B
    _, s, _ = torch.linalg.svd(B, full_matrices=False)
    
    # Compute the approximate U = Q * U_B
    # U = torch.mm(Q, U_B)

    return s
import torch

def eigh(a, b, subset_by_index=None):
    """
    Solve standard or generalized eigenvalue problem for a complex Hermitian
    or real symmetric matrix a, with b being the identity matrix for the standard
    problem or a positive definite matrix for the generalized problem.

    Parameters:
    - a: Tensor, the complex Hermitian or real symmetric matrix.
    - b: Tensor or None, the positive definite matrix for generalized problems,
         or None for standard problems.

    Returns:
    - w: Eigenvalues tensor.
    - v: Eigenvectors tensor.
    """
    

    # Generalized eigenvalue problem
    # Perform Cholesky decomposition of b
    L = torch.linalg.cholesky(b)
    L_inv = torch.linalg.inv(L)
    L_inv_conj_trans = L_inv.conj().transpose(-2, -1)
    
    # Transform a into a standard problem
    transformed_a = torch.bmm(torch.bmm(L_inv, a), L_inv_conj_trans)
    
    # Solve the standard problem
    w, transformed_v = torch.linalg.eigh(transformed_a)
    
    # Transform eigenvectors back to the original problem
    v = torch.bmm(L_inv_conj_trans, transformed_v)
    if subset_by_index is None:
      return w, v
    else:
      return w[..., subset_by_index], v[..., subset_by_index]

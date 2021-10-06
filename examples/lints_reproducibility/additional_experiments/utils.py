import numpy as np


def contains_repeated_eigenvalues(model):
    has_repeated = []
    for k in model.arm_to_model:
        num_vals = model.arm_to_model[k].A_inv.shape[0]
        num_unique = np.unique(np.floor(np.linalg.eig(model.arm_to_model[k].A_inv)[0] * 1e7) / 1e7).shape[0]
        has_repeated.append(num_vals != num_unique)
    return np.any(has_repeated)


def all_positive_definite(model):
    eigenval_all_positive = np.all([(np.linalg.eig(model.arm_to_model[k].A_inv)[0] > 0).all() for k in model.arm_to_model])
    is_symmetric = np.all([np.allclose(model.arm_to_model[k].A_inv, model.arm_to_model[k].A_inv.T) for k in model.arm_to_model])
    return eigenval_all_positive and is_symmetric

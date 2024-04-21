import argparse
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from data_utils import data_to_matrices, Z_prime, nan_to_user_movie_mean
import warnings
import random
warnings.filterwarnings('ignore')


def RMSE(Z_prim, T):
    mask = ~np.isnan(Z_prim) & ~np.isnan(T)
    rmse = np.sqrt(np.nanmean((Z_prim[mask] - T[mask]) ** 2))
    return rmse


def RCMD(train, test, alg, result):
    Z, T = data_to_matrices(train, test)

    if alg == "NMF":
        Z_replaced = nan_to_user_movie_mean(Z)
        model = NMF(n_components=14, init='random', random_state=0)
        W = model.fit_transform(Z_replaced)
        H = model.components_
        Z_approx = np.dot(W, H)

        Z_prim = Z_prime(Z_approx, T)
        rmse = RMSE(Z_prim, T)
        with open(result, 'w') as f:
            f.write(f"{rmse}")

    elif alg == "SVD1":
        svd = TruncatedSVD(n_components=15, random_state=0)
        Z_replaced = nan_to_user_movie_mean(Z)
        svd.fit(Z_replaced)
        sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z_replaced) / svd.singular_values_
        H = np.dot(sigma2, VT)
        Z_approx = np.dot(W, H)

        Z_prim = Z_prime(Z_approx, T)
        rmse = RMSE(Z_prim, T)
        with open(result, 'w') as f:
            f.write(f"{rmse}")

    elif alg == "SVD2":
        n_components, max_iter = 15, 100
        tol = 1e-4
        Z_replaced = nan_to_user_movie_mean(Z)
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        rmse_list = []

        for i in range(max_iter):
            svd.fit(Z_replaced)
            sigma2 = np.diag(svd.singular_values_)
            VT = svd.components_
            W = svd.transform(Z_replaced) / svd.singular_values_
            H = np.dot(sigma2, VT)
            Z_approx = np.dot(W, H)
            Z_replaced[np.isnan(Z)] = Z_approx[np.isnan(Z)]

            Z_prim = Z_prime(Z_approx, T)
            current_rmse = RMSE(Z_prim, T)
            rmse_list.append(current_rmse)

            if i > 0:
                rmse_diff = rmse_list[-1] - rmse_list[-2]
                if abs(rmse_diff) < tol:
                    break
                elif rmse_diff > 0:
                    break

        best_rmse = min(rmse_list)
        with open(result, 'w') as f:
            f.write(f"{best_rmse}")

    elif alg == "SGD":
        n, d = Z.shape
        r, alpha = 13, 0.008
        num_iterations = 300000
        W = np.random.rand(n, r)
        H = np.random.rand(r, d)
        nn_set = [(i, j) for i in range(n) for j in range(d) if not np.isnan(Z[i, j])]

        for iteration in range(num_iterations):
            i, j = random.choice(nn_set)
            e = Z[i, j] - np.dot(W[i, :], H[:, j])
            dw_i = -2 * e * H[:, j]
            dh_j = -2 * e * W[i, :]
            W[i, :] -= alpha * dw_i
            H[:, j] -= alpha * dh_j
        Z_approx = np.dot(W, H)

        Z_prim = Z_prime(Z_approx, T)
        rmse = RMSE(Z_prim, T)
        with open(result, 'w') as f:
            f.write(f"{rmse}")

    print(f"Training with {train}, testing with {test}, using {alg} algorithm and saving results to {result}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recommender System Parameters')

    parser.add_argument('train', type=str, help='Path to the training file')
    parser.add_argument('test', type=str, help='Path to the test file')
    parser.add_argument('alg', type=str, choices=['NMF', 'SVD1', 'SVD2', 'SGD', 'SVD3'], help='Algorithm to use')
    parser.add_argument('result', type=str, help='Path to save the result file')

    args = parser.parse_args()
    RCMD(args.train, args.test, args.alg, args.result)

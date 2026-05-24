



import cvxpy as cp, numpy as np
import matplotlib.pyplot as plt
import os


def calculate_portfolio_return(returns, weights):
    return np.dot(returns, weights)

def calculate_optimal_portfolio(mu, sigma, gamma, negativity=False):
    G = np.linalg.cholesky(sigma)

    ## using cvxpy to implement this
    X = cp.Variable(len(mu), nonneg=negativity)
    constraints = [cp.sum(X)==1, cp.norm(G.T @ X) <= np.sqrt(gamma)]
    objective = cp.Maximize(mu.T @ X)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    returns = prob.value
    portfolio = X.value
    return returns, portfolio


def plot_portfolio(weights, labels=None):
    if labels is None:
        labels = [f"Stock {i+1}" for i in range(8)]
    os.makedirs(r"c:\Users\micha\.vscode\Python Projects\portfolio-optimizer\plots", exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar(labels, weights)
    ax.set_xlabel("Stock")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Composition")
    fig.savefig(r"c:\Users\micha\.vscode\Python Projects\portfolio-optimizer\plots\portfolio.png")
    plt.show()


def print_summary(returns, weights, labels=None):
    if labels is None:
        labels = [f"Stock {i+1}" for i in range(len(weights))]

    print("=== Portfolio Summary ===")
    print(f"Optimal Return: {returns:.4f}")
    print()

    col_w = max(len(lbl) for lbl in labels)
    col_w = max(col_w, len("Stock"))
    header = f"{'Stock':<{col_w}}  {'Weight':>8}"
    sep    = f"{'-'*col_w}  {'-'*8}"
    print(header)
    print(sep)
    for lbl, w in zip(labels, weights):
        print(f"{lbl:<{col_w}}  {w:>8.4f}")

    variance = float(weights @ weights)
    print()
    print(f"Portfolio variance: {variance:.4f}")


if __name__ == "__main__":
    print('hi')
    m = np.array(
    [0.0720, 0.1552, 0.1754, 0.0898, 0.4290, 0.3929, 0.3217, 0.1838])
    S = np.array([
        [0.0946, 0.0374, 0.0349, 0.0348, 0.0542, 0.0368, 0.0321, 0.0327],
        [0.0374, 0.0775, 0.0387, 0.0367, 0.0382, 0.0363, 0.0356, 0.0342],
        [0.0349, 0.0387, 0.0624, 0.0336, 0.0395, 0.0369, 0.0338, 0.0243],
        [0.0348, 0.0367, 0.0336, 0.0682, 0.0402, 0.0335, 0.0436, 0.0371],
        [0.0542, 0.0382, 0.0395, 0.0402, 0.1724, 0.0789, 0.0700, 0.0501],
        [0.0368, 0.0363, 0.0369, 0.0335, 0.0789, 0.0909, 0.0536, 0.0449],
        [0.0321, 0.0356, 0.0338, 0.0436, 0.0700, 0.0536, 0.0965, 0.0442],
        [0.0327, 0.0342, 0.0243, 0.0371, 0.0501, 0.0449, 0.0442, 0.0816]
    ])
    ret, portfolio = calculate_optimal_portfolio(m, S, 0.05)
    print_summary(ret, portfolio)
    plot_portfolio(portfolio)

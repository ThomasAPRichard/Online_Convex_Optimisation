# ============================================
# Imports
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from tqdm import tqdm

# ============================================
# Part 1 — Rock Paper Scissors
# ============================================

# Question 1

p0 = np.array([1/3, 1/3, 1/3]) # distribution fixe de l'adversaire

# Ici M = N = 3, donc on a a une matrice de pertes 3x3 avec en ligne et en colonne dans l'ordre : Rock, Paper, Scissors
loss_matrix = np.array([[0, 1, -1],
                        [-1, 0, 1],
                        [1, -1, 0]])
actions = ['Rock', 'Paper', 'Scissors']


# ============================================
# Question 2

def rand_weight (p):
    """
    p :vecteur d3 de proba
    i :l'index i choisi selon la loi p
    """
    uni = np.random.uniform(0, 1)
    somme = 0.0              # Mon idée : on place p1, p2, p3 sur [0,1] et on tire un uniforme. 
    for i in range(3):
        somme += p[i]
        if uni < somme :    # Si le tirage est dans l'intervalle de pi, on renvoie i
            return i 
        
# Test empirique de rand_weight
p0 = np.array([1/3, 1/6, 1/2])
counts = np.zeros(3)
n_trials = 100000
for _ in range(n_trials):
    i = rand_weight(p0)
    counts[i] += 1
print("Empirical probabilities:", counts / n_trials)


#Je rajoute un paramètre eta non demandé dans l'énoncé pour correspondre à la définition de p_t+1
def EWA_uptade(p, loss_vector, eta = 0.5):  
    """
    p : vecteur de proba d3
    loss_vector : vecteur de pertes d3
    
    return : nouveau vecteur de proba
    """
    new_p = np.zeros(3)
    Z = 0.0
    for i in range(3):
        new_p[i] = p[i] * np.exp(-eta * loss_vector[i])
        Z += new_p[i]
    new_p /= Z
    return new_p


# ============================================
# Question 3


def simulation_EWA (T, p0, q0, loss_matrix, eta=0.5):
    """
    T : nombre de tours
    p0 : vecteur de proba initial
    loss_matrix : matrice des pertes
    
    return : liste des vecteurs de proba à chaque tour
    """
    proba_joueur = p0
    proba_adversaire = q0
    evolution_probas = [proba_joueur.copy()]
    losses = np.zeros(T)
    for i in range (T):
        action_joueur = rand_weight(proba_joueur)
        action_adversaire = rand_weight(proba_adversaire)

        loss_ijt = loss_matrix[action_joueur, action_adversaire]   
        losses[i] = loss_ijt

        loss_vector = loss_matrix[:, action_adversaire]
        new_proba = EWA_uptade(proba_joueur, loss_vector, eta)
        proba_joueur = new_proba
        evolution_probas.append(proba_joueur.copy())

    return evolution_probas, losses

# Test de la simulation EWA
T = 100
proba_joueur = np.array([1/3, 1/3, 1/3])
proba_adversaire = np.array([1/6, 1/3, 1/2])
evolution_probas, losses = simulation_EWA(T, proba_joueur, proba_adversaire, loss_matrix, eta=1)

# Affichage de l'évolution des probabilités
plt.plot([p[0] for p in evolution_probas], label='Rock')
plt.plot([p[1] for p in evolution_probas], label='Paper')
plt.plot([p[2] for p in evolution_probas], label='Scissors')
plt.xlabel('Round')
plt.ylabel('Probability')
plt.title('Evolution of Player Probabilities using EWA')
plt.legend()
plt.show()


n_experiments = 200
eta = 1
L = loss_matrix
q = np.array([1/6, 1/3, 1/2])

all_losses = np.zeros((n_experiments, T))  # shape: (n_experiments, T)

for exp in range(n_experiments):
    truc, all_losses[exp, :] = simulation_EWA(T, proba_joueur, q, L, eta)

# pertes cumulées par expérience
cum_losses = all_losses.cumsum(axis=1)                      # somme sur le temps
t_grid = np.arange(1, T + 1)
avg_loss_per_exp = cum_losses / t_grid                     # \bar ℓ_t pour chaque exp
avg_loss_over_exps = avg_loss_per_exp.mean(axis=0)         # moyenne sur les 200 exp

# plot de \bar ℓ_t en fonction de t 
plt.figure()
plt.plot(t_grid, avg_loss_over_exps)
plt.xlabel("t")
plt.ylabel("average loss L_t")
plt.title("Average cumulative loss over n = 200 simulations (EWA, $\eta=1$)")
plt.grid(True)
plt.show()



# d.
liste_etas = [0.01, 0.03, 0.1, 0.3, 1]

avg_losses_eta = []
n_rep = 200

for eta in liste_etas:
    mean_losses = []
    for _ in range(n_rep):
        truc ,losses = simulation_EWA(T, proba_joueur, q, L, eta)
        mean_losses.append(losses.mean())   # \bar ℓ_T pour cette run
    avg_losses_eta.append(np.mean(mean_losses))  # moyenne sur les répétitions

plt.figure()
plt.plot(liste_etas, avg_losses_eta, marker="o")
plt.xlabel("learning rate η")
plt.ylabel("average loss $\\bar{\\ell}_T$")
plt.title("Average loss vs learning rate η (T = {})".format(T))
plt.grid(True)
plt.show()

# ============================================
# Question 4

def proj_simplex(x):
    """
    x : vecteur réel 
    return : vecteur q dans le simplexe (positif, somme = 1)
    """
    N = x.size

    # 1) Trier x en ordre décroissant
    u = np.sort(x)[::-1]
    # 2) Sommes cumulées
    cssv = np.cumsum(u)
    # 3) Trouver rho = max { j : u_j + (1 - sum_{k<=j} u_k) / j > 0 }
    rho = np.where(u + (1 - cssv) / np.arange(1, N + 1) > 0)[0][-1]
    # 4) Calculer le seuil theta
    theta = (cssv[rho] - 1) / (rho + 1)
    # 5) Projeter
    q = np.maximum(x - theta, 0.0)
    return q

def OGD_update(q_t, loss_vector, t):
    """
    q_t : vecteur de proba courant de l'adversaire
    loss_vector : vecteur des pertes L_{I_t, j}
    t : numéro du tour (t >= 1), sert à calculer eta_t = 1/sqrt(t)

    return : nouveau vecteur de proba q_{t+1} dans le simplexe
    """
    eta_t = 1 / np.sqrt(t)
    grad = loss_vector
    q_tilde = q_t - eta_t * grad
    q_next = proj_simplex(q_tilde)
    return q_next

# ============================================
# Quesrion 5
T = 10000

# Probas initiales
p0 = np.array([1/3, 1/3, 1/3])
q0 = np.array([1/3, 1/3, 1/3])

eta_player = 0.3

proba_joueur = p0.copy()
proba_adversaire = q0.copy()

# pour la moyenne \bar p_t
p_bar = np.zeros(3)
uniform = np.array([1/3, 1/3, 1/3])
norm_diffs = np.zeros(T)

for t in range(1, T + 1):
    # 1) moyenne \bar p_t (en incluant p_t courant)
    p_bar = ((t - 1) * p_bar + proba_joueur) / t
    norm_diffs[t - 1] = np.linalg.norm(p_bar - uniform)

    # 2) tirage des actions selon p_t et q_t
    action_joueur = rand_weight(proba_joueur)
    action_adversaire = rand_weight(proba_adversaire)

    # 3) pertes instantanées pour construire les vecteurs de pertes
    #    pour le joueur (EWA) : L_{i, J_t} -> colonne de loss_matrix
    loss_vector_player = loss_matrix[:, action_adversaire]

    #    pour l'adversaire (OGD) : L_{I_t, j} -> ligne de loss_matrix
    loss_vector_adversaire = loss_matrix[action_joueur, :]

    # 4) mise à jour EWA (joueur)
    proba_joueur = EWA_uptade(proba_joueur, loss_vector_player, eta=eta_player)

    # 5) mise à jour OGD (adversaire)
    proba_adversaire = OGD_update(proba_adversaire, loss_vector_adversaire, t)


# le plot
t_grid = np.arange(1, T + 1)

plt.figure()
plt.loglog(t_grid, norm_diffs)
plt.xlabel("t")
plt.ylabel(r"$\|\bar p_t - (1/3,1/3,1/3)\|_2$")
plt.title("EWA (player) vs OGD (adversary) on Rock-Paper-Scissors")
plt.grid(True, which="both")
plt.show()


# ============================================
# Question 6
def Hedge_update(p_t, q_t, eta):
    """
    Mise à jour Hedge pour le joueur.

    p_t : vecteur de proba courant du joueur 
    q_t : stratégie de l'adversaire 

    return : p_{t+1} dans le simplexe
    """

    loss_vector_player = q_t @ loss_matrix

    # mise à jour exponentielle (comme EWA, mais avec la perte moyenne)
    weights = p_t * np.exp(-eta * loss_vector_player)
    p_next = weights / weights.sum()
    return p_next


# Run et plot
T = 10000

# proba initiales
p0 = np.array([1/3, 1/3, 1/3])
q0 = np.array([1/3, 1/3, 1/3])

proba_joueur = p0.copy()
proba_adversaire = q0.copy()

eta_player = 0.3 

p_bar = np.zeros(3)
uniform = np.array([1/3, 1/3, 1/3])

norm_diffs_hedge = np.zeros(T)

for t in range(1, T + 1):
    # moyenne p_t BAR
    p_bar = ((t - 1) * p_bar + proba_joueur) / t
    norm_diffs_hedge[t - 1] = np.linalg.norm(p_bar - uniform)

    action_joueur = rand_weight(proba_joueur)
    action_adversaire = rand_weight(proba_adversaire)

    loss_vector_adversaire = loss_matrix[action_joueur, :]

    # --- mise à jour Hedge (joueur) ---
    proba_joueur = Hedge_update(proba_joueur, proba_adversaire, eta=eta_player)

    # --- mise à jour OGD (adversaire) ---
    proba_adversaire = OGD_update(proba_adversaire, loss_vector_adversaire, t)

# --- Plot en log-log ---
t_grid = np.arange(1, T + 1)

plt.figure()
plt.loglog(t_grid, norm_diffs_hedge, label="Hedge vs OGD")
plt.loglog(t_grid, norm_diffs, label="EWA vs OGD")
plt.xlabel("t")
plt.ylabel(r"$\|\bar p_t - (1/3,1/3,1/3)\|_2$")
plt.title("Hedge (player) vs OGD (adversary) - Rock Paper Scissors")
plt.grid(True, which="both")
plt.legend()
plt.show()




# ============================================
# Exercice 2 — MINST 
# ============================================


## C'est CHATGPT qui à généré cette partie
import numpy as np
from pathlib import Path
import struct

DATA_DIR = Path("/Users/thomasrichard/Desktop/Etudes/Master2/Online_Convex_Optimisation/archive")

def read_mnist_images(path: Path) -> np.ndarray:
    """Lit un fichier images MNIST idx3-ubyte (non .gz) -> (n, 784) uint8."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} (expected 2051) in {path.name}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)

def read_mnist_labels(path: Path) -> np.ndarray:
    """Lit un fichier labels MNIST idx1-ubyte (non .gz) -> (n,) uint8."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} (expected 2049) in {path.name}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def pick_file(kind: str, split: str) -> Path:
    """
    kind: "images" ou "labels"
    split: "train" ou "t10k"
    Choisit automatiquement le bon fichier parmi ceux présents.
    """
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if kind == "images":
        cand = [p for p in files if split in p.name and "images" in p.name and "idx3" in p.name and "ubyte" in p.name]
        # on préfère le nom standard avec '-' si présent
        cand.sort(key=lambda p: (0 if "-idx3-" in p.name else 1, len(p.name)))
    else:
        cand = [p for p in files if split in p.name and "labels" in p.name and "idx1" in p.name and "ubyte" in p.name]
        cand.sort(key=lambda p: (0 if "-idx1-" in p.name else 1, len(p.name)))
    if not cand:
        raise FileNotFoundError(f"Impossible de trouver {split} {kind} dans {DATA_DIR}")
    return cand[0]

train_images_path = pick_file("images", "train")
train_labels_path = pick_file("labels", "train")
test_images_path  = pick_file("images", "t10k")
test_labels_path  = pick_file("labels", "t10k")

print("Fichiers utilisés :")
print(" train images:", train_images_path.name)
print(" train labels:", train_labels_path.name)
print(" test images :", test_images_path.name)
print(" test labels :", test_labels_path.name)

X_train = read_mnist_images(train_images_path)
y_train = read_mnist_labels(train_labels_path)
X_test  = read_mnist_images(test_images_path)
y_test  = read_mnist_labels(test_labels_path)

print("Shapes:")
print(" X_train:", X_train.shape, X_train.dtype)
print(" y_train:", y_train.shape, y_train.dtype)
print(" X_test :", X_test.shape,  X_test.dtype)
print(" y_test :", y_test.shape,  y_test.dtype)

# Optionnel: normaliser + intercept + binaire 0 vs not-0
X_train_f = X_train.astype(np.float32) / 255.0
X_test_f  = X_test.astype(np.float32) / 255.0

X_train_f = np.hstack([np.ones((X_train_f.shape[0], 1), dtype=np.float32), X_train_f])  # (n, 785)
X_test_f  = np.hstack([np.ones((X_test_f.shape[0], 1), dtype=np.float32), X_test_f])

y_train_bin = np.where(y_train == 0, 1, -1).astype(np.int8)
y_test_bin  = np.where(y_test == 0, 1, -1).astype(np.int8)

print("Binaire:", np.unique(y_train_bin))
print("X_train_f:", X_train_f.shape)

# ============================================
# Question 2
def loss_logistic (y,z):
    return np.log(1 + np.exp(-y * z))

def gradient_logistic(theta, x, y):
    return (-y / (1 + np.exp(y * np.dot(theta, x)))) * x


def hessian_logistic(theta, x, y):
    return (np.exp(y * float(np.dot(theta, x))) / (1.0 + np.exp(y * float(np.dot(theta, x))))**2) * np.outer(x, x)

# ============================================
# Question 3

def proj_l2_ball(theta, D):
    nrm = np.linalg.norm(theta)
    if nrm <= D:
        return theta
    return (D / nrm) * theta

def run_OGD(X, y, D, eta0=None, T=None, theta0=None):
    n, d = X.shape
    if T is None:
        T = n
    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()


    B = np.sqrt(d)

    if eta0 is None:
        eta0 = 0.1

    thetas = np.zeros((T, d))
    losses = np.zeros(T)

    for t in range(1, T + 1):
        x_t = X[t - 1]
        y_t = y[t - 1]

        losses[t - 1] = loss_logistic(np.dot(theta, x_t), y_t)
        thetas[t - 1] = theta

        g = gradient_logistic(theta, x_t, y_t)

        eta_t = eta0 / np.sqrt(t)
        theta = theta - eta_t * g
        theta = proj_l2_ball(theta, D)

    return thetas, losses



def run_AdaGrad(X, y, D, eta0=None, eps=1e-8, T=None, theta0=None):
    n, d = X.shape
    if T is None:
        T = n
    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    B = np.sqrt(d)
    if eta0 is None:
        eta0 = 0.1

    G = np.zeros(d)  # accumulateur des carrés de gradients

    thetas = np.zeros((T, d))
    losses = np.zeros(T)

    for t in range(1, T + 1):
        x_t = X[t - 1]
        y_t = y[t - 1]

        losses[t - 1] = loss_logistic(np.dot(theta, x_t), y_t)
        thetas[t - 1] = theta

        g = gradient_logistic(theta, x_t, y_t)
        G += g * g

        theta = theta - eta0 * g / (np.sqrt(G) + eps)
        theta = proj_l2_ball(theta, D)

    return thetas, losses


D = 10 #au pif
T = 20000  
# Pour eta je n'ai aucune idée de ce que je dois prendre

thetas_ogd, losses_ogd = run_OGD(X_train_f, y_train_bin, D=D, T=T)
thetas_adg, losses_adg = run_AdaGrad(X_train_f, y_train_bin, D=D, T=T)

print("OGD final loss mean:", losses_ogd.mean())
print("AdaGrad final loss mean:", losses_adg.mean())


# ============================================
# Question 4

def batch_grad_logistic(theta, X, y):
    s = y * (X @ theta)
    coef = 1.0 / (1.0 + np.exp(s))
    return X.T @ (-y * coef)

def fit_theta_star_PGD(X, y, D, n_epochs=5, lr=0.1, batch_size=2048, seed=0):

    rng = np.random.default_rng(seed)
    n, d = X.shape
    theta = np.zeros(d)

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            g = batch_grad_logistic(theta, X[idx], y[idx])
            theta = theta - lr * g / max(1, len(idx))   # moyenne sur le batch
            theta = proj_l2_ball(theta, D)

    return theta

D = 10
T = 20000  # prends le même T que pour OGD/AdaGrad pour comparer proprement

X = X_train_f[:T]
y = y_train_bin[:T]

theta_star = fit_theta_star_PGD(X, y, D=D, n_epochs=5, lr=0.5, batch_size=2048)
print("||theta_star||:", np.linalg.norm(theta_star))



def losses_for_fixed_theta(theta, X, y):
    margins = y * (X @ theta)
    return np.logaddexp(0.0, -margins)


X = X_train_f[:T]
y = y_train_bin[:T]

loss_star = losses_for_fixed_theta(theta_star, X, y)   

S_ogd = np.cumsum(losses_ogd)
S_adg = np.cumsum(losses_adg)
S_star = np.cumsum(loss_star)

regret_ogd = S_ogd - S_star
regret_adg = S_adg - S_star

t_grid = np.arange(1, T + 1)

plt.figure()
plt.plot(t_grid, regret_ogd, label="Regret OGD")
plt.plot(t_grid, regret_adg, label="Regret AdaGrad")
plt.xlabel("t")
plt.ylabel("Regret_t")
plt.title("Regret vs best fixed theta* in K")
plt.grid(True)
plt.legend()
plt.show()

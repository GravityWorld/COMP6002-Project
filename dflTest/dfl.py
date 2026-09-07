"""
Decentralized Federated Learning (DFL) on MNIST.

No central server: each client only exchanges model weights with its
graph neighbors (gossip averaging), instead of all clients syncing
through a coordinator like in standard FedAvg.

Each round:
  1. Every client trains locally for a few epochs on its own MNIST shard.
  2. Every client updates its weights to a weighted average of its own
     weights and its neighbors' weights (using a Metropolis-Hastings
     mixing matrix, which is doubly stochastic -> guarantees convergence
     to consensus under mild conditions).

Try changing TOPOLOGY between 'ring', 'random', and 'fully_connected'
to see how connectivity affects convergence speed, and IID=False to
see how label-skewed data hurts decentralized training.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ----------------------------- Config -----------------------------
NUM_CLIENTS = 3
ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.05
TOPOLOGY = "ring"          # 'ring', 'random', 'fully_connected'
RANDOM_TOPO_P = 0.3        # edge probability if TOPOLOGY == 'random'
IID = False                # False -> each client mostly sees 2 digit classes
SEED = 0
EVAL_EVERY = 2              # how often (in rounds) to compute test accuracy
LIVE_PLOT = True            # update the plot window as training runs

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Data -----------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


def partition_iid(dataset, n):
    idx = np.random.permutation(len(dataset))
    return np.array_split(idx, n)


def partition_noniid(dataset, n, shards_per_client=2):
    labels = np.array(dataset.targets)
    idx_sorted = np.argsort(labels)
    num_shards = n * shards_per_client
    shard_size = len(dataset) // num_shards
    shards = [idx_sorted[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)
    return [
        np.concatenate(shards[i * shards_per_client:(i + 1) * shards_per_client])
        for i in range(n)
    ]


client_indices = (
    partition_iid(train_set, NUM_CLIENTS)
    if IID
    else partition_noniid(train_set, NUM_CLIENTS)
)
client_loaders = [
    DataLoader(Subset(train_set, idx), batch_size=BATCH_SIZE, shuffle=True)
    for idx in client_indices
]
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# Which digit classes does each client actually see, and how much of each?
NUM_CLASSES = 10
train_labels = np.array(train_set.targets)
client_class_counts = []  # list of length-10 arrays, samples per class per client
client_own_classes = []   # classes that make up a meaningful share of each client's data
for i, idx in enumerate(client_indices):
    counts = np.bincount(train_labels[idx], minlength=NUM_CLASSES)
    client_class_counts.append(counts)
    own = [c for c in range(NUM_CLASSES) if counts[c] > 0.05 * counts.sum()]
    client_own_classes.append(own)
    dist_str = ", ".join(f"{c}:{counts[c]}" for c in range(NUM_CLASSES) if counts[c] > 0)
    print(f"Client {i} sees classes -> {dist_str}")


# ----------------------------- Model -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ------------------------ Topology / mixing matrix ------------------------
def build_adjacency(n, topology):
    A = np.zeros((n, n), dtype=int)
    if topology == "ring":
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[i, (i - 1) % n] = 1
    elif topology == "fully_connected":
        A = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
    elif topology == "random":
        A = (np.random.rand(n, n) < RANDOM_TOPO_P).astype(int)
        A = np.triu(A, 1)
        A = A + A.T
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return A


def metropolis_weights(A):
    """Doubly-stochastic mixing matrix -> gossip averaging provably
    converges to the consensus (average) value."""
    n = A.shape[0]
    degree = A.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (max(degree[i], degree[j]) + 1)
        W[i, i] = 1 - W[i].sum()
    return W


A = build_adjacency(NUM_CLIENTS, TOPOLOGY)
W = metropolis_weights(A)
print(f"Topology: {TOPOLOGY}, avg degree: {A.sum(axis=1).mean():.1f}")


def plot_topology(A, save_path="topology.png"):
    """Draw the communication graph (nodes on a circle, edges between neighbors)."""
    n = A.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="gray", zorder=1)
    ax.scatter(xs, ys, s=600, color="#4C72B0", zorder=2)
    for i in range(n):
        ax.text(xs[i], ys[i], str(i), color="white", ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=3)
    ax.set_title(f"Topology: {TOPOLOGY} ({n} clients)")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved topology diagram to {save_path}")
    plt.close(fig)


plot_topology(A)

# ------------------------ Client models ------------------------
client_models = [CNN().to(device) for _ in range(NUM_CLIENTS)]


def local_train(model, loader, epochs):
    model.train()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view(p.shape))
        idx += n


def evaluate_per_class(model, loader, num_classes=NUM_CLASSES):
    """Returns (per_class_accuracy, overall_accuracy)."""
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for c in range(num_classes):
                mask = y == c
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()
    per_class = correct / np.maximum(total, 1)
    overall = correct.sum() / total.sum()
    return per_class, overall


def avg_pairwise_disagreement(flat_params):
    """Mean L2 distance between every pair of clients' flattened weights.
    This is the direct DFL consensus metric -> should trend toward 0
    (or plateau above 0 under non-IID data / sparse graphs)."""
    n = len(flat_params)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(torch.norm(flat_params[i] - flat_params[j]).item())
    return float(np.mean(dists))


# ------------------------ Live plot setup ------------------------
history = {
    "round": [],
    "disagreement": [],
    "acc_round": [],
    "overall_acc": [[] for _ in range(NUM_CLIENTS)],
    "per_class_acc": [[] for _ in range(NUM_CLIENTS)],  # per_class_acc[client] = list of length-10 arrays over time
}

CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

if LIVE_PLOT:
    plt.ion()
    fig = plt.figure(figsize=(4.2 * NUM_CLIENTS, 8))
    gs = fig.add_gridspec(2, NUM_CLIENTS, height_ratios=[2, 1])
    client_axes = [fig.add_subplot(gs[0, i]) for i in range(NUM_CLIENTS)]
    summary_ax = fig.add_subplot(gs[1, :])
    summary_ax2 = summary_ax.twinx()
    fig.suptitle(f"DFL on MNIST — {NUM_CLIENTS} clients, {TOPOLOGY} topology, IID={IID}")

# ------------------------ Decentralized training loop ------------------------
for rnd in range(1, ROUNDS + 1):
    # 1) local training on each client's own shard
    for i in range(NUM_CLIENTS):
        local_train(client_models[i], client_loaders[i], LOCAL_EPOCHS)

    # 2) gossip averaging: each client mixes with its graph neighbors only
    flat_params = [get_flat_params(m) for m in client_models]
    disagreement = avg_pairwise_disagreement(flat_params)
    history["round"].append(rnd)
    history["disagreement"].append(disagreement)

    new_flat = []
    for i in range(NUM_CLIENTS):
        agg = torch.zeros_like(flat_params[0])
        for j in range(NUM_CLIENTS):
            if W[i, j] > 0:
                agg += W[i, j] * flat_params[j]
        new_flat.append(agg)
    for i in range(NUM_CLIENTS):
        set_flat_params(client_models[i], new_flat[i])

    if rnd % EVAL_EVERY == 0 or rnd == 1:
        history["acc_round"].append(rnd)
        overall_accs = []
        for i in range(NUM_CLIENTS):
            per_class, overall = evaluate_per_class(client_models[i], test_loader)
            history["per_class_acc"][i].append(per_class)
            history["overall_acc"][i].append(overall)
            overall_accs.append(overall)
        print(
            f"Round {rnd:3d} | avg acc: {np.mean(overall_accs):.4f} "
            f"| spread: {np.max(overall_accs) - np.min(overall_accs):.4f} "
            f"| param disagreement: {disagreement:.3f}"
        )

    if LIVE_PLOT:
        for i, ax in enumerate(client_axes):
            ax.clear()
            own = client_own_classes[i]
            per_class_history = history["per_class_acc"][i]  # list of length-10 arrays
            for c in range(NUM_CLASSES):
                y = [pc[c] for pc in per_class_history]
                if c in own:
                    ax.plot(history["acc_round"], y, color=CLASS_COLORS[c],
                            linewidth=2.2, label=f"digit {c}")
                else:
                    ax.plot(history["acc_round"], y, color=CLASS_COLORS[c],
                            linewidth=0.8, linestyle=":", alpha=0.5)
            ax.set_title(f"Client {i} — trained on {own}")
            ax.set_ylim(0, 1)
            ax.set_xlabel("round")
            if i == 0:
                ax.set_ylabel("per-class accuracy")
            ax.legend(fontsize=6, ncol=2, loc="lower right")

        summary_ax.clear()
        summary_ax2.clear()
        for i in range(NUM_CLIENTS):
            summary_ax.plot(history["acc_round"], history["overall_acc"][i],
                             marker="o", markersize=3, label=f"client {i} overall")
        summary_ax2.plot(history["round"], history["disagreement"], color="black",
                          linestyle="--", linewidth=1.2, label="param disagreement")
        summary_ax.set_ylim(0, 1)
        summary_ax.set_xlabel("round")
        summary_ax.set_ylabel("overall accuracy")
        summary_ax2.set_ylabel("param disagreement (L2)")
        lines1, labels1 = summary_ax.get_legend_handles_labels()
        lines2, labels2 = summary_ax2.get_legend_handles_labels()
        summary_ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")

        fig.canvas.draw()
        plt.pause(0.01)

print("\nDone. Solid lines = digits the client trained on directly; dotted lines =")
print("digits it never saw locally, only learned about through gossip from neighbors.")

if LIVE_PLOT:
    plt.ioff()
    fig.savefig("dfl_metrics.png", dpi=120, bbox_inches="tight")
    print("Saved final metrics plot to dfl_metrics.png")
    plt.show()
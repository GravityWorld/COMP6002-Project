"""
One-directional teacher -> student weight transfer on MNIST.

Two models, NOT symmetric gossip:
  - Model 0 ("teacher") only ever trains on its own MNIST shard.
    It never incorporates anything from the student. Its weights
    are completely unaffected by the student's existence.
  - Model 1 ("student") trains on its own (smaller) shard, then
    each round blends its weights toward the teacher's:
        student <- (1 - ALPHA) * student + ALPHA * teacher
    This is a directed pull, not mutual averaging.

Expected outcome: the student starts out only knowing its own classes,
and by the end should have picked up accuracy on the teacher's classes
too - purely by repeatedly absorbing a fraction of the teacher's weights,
with no gradient steps of its own on those classes.
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
STUDENT_CLASSES = [2, 3, 4]              # what the student starts out knowing
TEACHER_CLASSES = [0, 1, 5, 6, 7, 8, 9]   # what the teacher knows (never changes)
ALPHA = 0.3          # how strongly the student pulls toward the teacher each round
                      # (0 = no teaching at all, 1 = student becomes a teacher copy immediately)
ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.05
EVAL_EVERY = 2
SEED = 0
NUM_CLASSES = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Data -----------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_labels = np.array(train_set.targets)


def indices_for_classes(classes):
    return np.where(np.isin(train_labels, classes))[0]


teacher_idx = indices_for_classes(TEACHER_CLASSES)
student_idx = indices_for_classes(STUDENT_CLASSES)
teacher_loader = DataLoader(Subset(train_set, teacher_idx), batch_size=BATCH_SIZE, shuffle=True)
student_loader = DataLoader(Subset(train_set, student_idx), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

print(f"Teacher trains only on classes {TEACHER_CLASSES} ({len(teacher_idx)} samples)")
print(f"Student trains only on classes {STUDENT_CLASSES} ({len(student_idx)} samples)")
print(f"Student pulls {ALPHA:.0%} of the way toward the teacher's weights each round.\n")


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


teacher = CNN().to(device)
student = copy.deepcopy(teacher)  # SAME initial weights as teacher, not a fresh random init -
                                    # averaging two independently-initialized nets is unstable
                                    # (see "linear mode connectivity"); same start avoids that.


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


# ------------------------ Live plot setup ------------------------
history = {
    "acc_round": [],
    "per_class_acc": {"teacher": [], "student": []},
    "overall_acc": {"teacher": [], "student": []},
}
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

plt.ion()
fig, (ax_teacher, ax_student) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
fig.suptitle(f"Teacher -> Student weight transfer (ALPHA={ALPHA})")

# ------------------------ Training loop ------------------------
for rnd in range(1, ROUNDS + 1):
    # teacher trains completely independently - never sees the student
    local_train(teacher, teacher_loader, LOCAL_EPOCHS)
    # student trains on its own data
    local_train(student, student_loader, LOCAL_EPOCHS)

    # one-directional pull: student blends toward teacher, teacher is untouched
    teacher_flat = get_flat_params(teacher)
    student_flat = get_flat_params(student)
    new_student_flat = (1 - ALPHA) * student_flat + ALPHA * teacher_flat
    set_flat_params(student, new_student_flat)

    if rnd % EVAL_EVERY == 0 or rnd == 1:
        history["acc_round"].append(rnd)
        t_pc, t_overall = evaluate_per_class(teacher, test_loader)
        s_pc, s_overall = evaluate_per_class(student, test_loader)
        history["per_class_acc"]["teacher"].append(t_pc)
        history["per_class_acc"]["student"].append(s_pc)
        history["overall_acc"]["teacher"].append(t_overall)
        history["overall_acc"]["student"].append(s_overall)
        print(f"Round {rnd:3d} | teacher overall: {t_overall:.4f} | student overall: {s_overall:.4f}")

    for ax, name, own in [
        (ax_teacher, "teacher", TEACHER_CLASSES),
        (ax_student, "student", STUDENT_CLASSES),
    ]:
        ax.clear()
        for c in range(NUM_CLASSES):
            y = [pc[c] for pc in history["per_class_acc"][name]]
            if c in own:
                ax.plot(history["acc_round"], y, color=CLASS_COLORS[c], linewidth=2.2, label=f"digit {c}")
            else:
                ax.plot(history["acc_round"], y, color=CLASS_COLORS[c], linewidth=0.9,
                        linestyle=":", alpha=0.6, label=f"digit {c} (not trained)")
        ax.set_title(f"{name.capitalize()} — trained on {own}")
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        if name == "teacher":
            ax.set_ylabel("per-class accuracy")
        ax.legend(fontsize=6, ncol=2, loc="lower right")

    fig.canvas.draw()
    plt.pause(0.01)

print("\nDone. Teacher's curves should be flat/unaffected throughout (it never")
print("hears from the student). Student's dotted lines (teacher-only classes)")
print("should climb over time - that's it absorbing knowledge purely from")
print("blending weights with the teacher, without ever training on those digits.")

plt.ioff()
fig.savefig("teacher_student_metrics.png", dpi=120, bbox_inches="tight")
print("Saved final plot to teacher_student_metrics.png")
plt.show()
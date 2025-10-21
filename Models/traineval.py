import torch
import numpy as np
from torch import nn
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@torch.no_grad()
def test_model(test_dataloader: DataLoader, model: nn.Module, loss_fn=nn.CrossEntropyLoss(), DEVICE="cpu"):
    # confusion matrix
    confmat = ConfusionMatrix(num_classes=3, task="multiclass").to(DEVICE)

    model.eval()
    model.to(DEVICE)
    for data_batch in tqdm(test_dataloader):
        imgs, labels = data_batch
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        confmat.update(preds, labels)

    confmat = confmat.compute().cpu().numpy()
    accuracy = np.trace(confmat) / np.sum(confmat)

    precision = confmat[1, 1] / np.sum(confmat[:, 1])
    recall = confmat[1, 1] / np.sum(confmat[1, :])
    f1_score = 2 * (precision * recall) / (precision + recall)
    return loss, confmat, accuracy, precision, recall, f1_score

def train_model_epoch(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, criterion, optimizer, writer: SummaryWriter, DEVICE="cpu", epoch=0):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    mean_train_loss = 0.0
    for idx, (images, labels) in loop:
        model.train()
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred_logits = model(images)
        loss = criterion(pred_logits, labels)
        mean_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    mean_train_loss /= len(train_loader)
        
    loss, confmat, accuracy, precision, recall, f1_score = test_model(test_loader, model, DEVICE=DEVICE)
    writer.add_scalar('Train/Loss', mean_train_loss, epoch)
    writer.add_scalar('Test/Loss', loss.item(), epoch * len(train_loader) + idx)
    writer.add_scalar('Test/Accuracy', accuracy, epoch * len(train_loader) + idx)
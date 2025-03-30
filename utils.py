from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_mnist_dataset(BATCH_SIZE):
    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    return train_dataset, train_data_loader


def get_device():
    if torch.cuda.is_available(): DEVICE = 'cuda'
    if torch.backends.mps.is_available(): DEVICE = 'mps'
    else: DEVICE = 'cpu'
    return DEVICE


def train(EPOCHS, train_data_loader, DEVICE, model, loss_function, optimizer, N_TRAIN_SAMPLES):
    losses, accs = [], []
    for epoch in range(EPOCHS):
        epoch_loss, n_corrects = 0., 0
        for imgs, labels in tqdm(train_data_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            preds = model.forward(imgs)
            loss = loss_function(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            N_SAMPLES = imgs.shape[0]
            epoch_loss += loss.item() * N_SAMPLES
            _, pred_classes = torch.max(preds, axis=1)
            n_corrects += torch.sum(pred_classes == labels).item()

        epoch_loss /= N_TRAIN_SAMPLES
        epoch_acc = n_corrects / N_TRAIN_SAMPLES
        losses.append(epoch_loss)
        accs.append(epoch_acc)

        print(f'Epoch: {epoch + 1}')
        print(f'Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

    return losses, accs


def vis_losses_accs(losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(losses)
    axes[0].set_ylabel('CrossEntropyLoss', fontsize=20)
    axes[0].tick_params(labelsize=15)

    axes[1].plot(accs)
    axes[1].set_xlabel('Epoch', fontsize=20)
    axes[1].set_ylabel('Accracy', fontsize=20)
    axes[1].tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, DEVICE, num_images):
    # 테스트 데이터셋 로드
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 그리드 크기 계산
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    with torch.no_grad():
        for idx in range(num_images):
            # 이미지와 실제 레이블 가져오기
            img, true_label = test_dataset[idx]
            
            # 예측
            img = img.unsqueeze(0).to(DEVICE)  # 배치 차원 추가
            pred = model(img)
            predicted_label = torch.max(pred, 1)[1].item()
            
            # 서브플롯 인덱스 계산
            i = idx // grid_size
            j = idx % grid_size
            
            # 이미지 표시
            axes[i, j].imshow(img.cpu().squeeze(), cmap='gray')
            
            # 제목 설정 (예측값과 실제값)
            title_color = 'green' if predicted_label == true_label else 'red'
            axes[i, j].set_title(f'P:{predicted_label} / T:{true_label}', 
                               color=title_color)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, DEVICE):
    # 테스트 데이터셋 로드
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))  # 전체 데이터를 한 번에 처리
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    with torch.no_grad():
        # 전체 테스트 데이터에 대한 예측 수행
        images, labels = next(iter(test_loader))  # 전체 데이터 가져오기
        images = images.to(DEVICE)
        # images = images.reshape(images.shape[0], -1)  # Flatten
        
        # 예측
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # CPU로 이동
        predicted = predicted.cpu().numpy()
        labels = labels.numpy()
    
    # confusion matrix 계산 및 시각화
    cm = confusion_matrix(labels, predicted)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
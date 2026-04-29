import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class MultimodalExperimentDataset(Dataset):
    """
    Класс набора данных для мультимодальной классификации видеокадров.

    Обеспечивает загрузку визуальных (изображения) и текстовых (субтитры) данных,
    а также их предварительную обработку и токенизацию для подачи в нейронную сеть.
    """

    def __init__(self, data_dir, split_csv_path, split_type='train'):
        self.img_dir = os.path.join(data_dir, 'images')
        self.txt_dir = os.path.join(data_dir, 'texts')
        df = pd.read_csv(split_csv_path)
        self.df_split = df[df['split'] == split_type].copy()

        unique_cats = sorted(df['category'].unique())
        self.class_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        self.idx_to_class = {i: cat for cat, i in self.class_to_idx.items()}
        self.num_classes = len(unique_cats)

        self.samples = []
        valid_ids = set(self.df_split['video_id'].tolist())
        for file in os.listdir(self.img_dir):
            if file.endswith('.jpg'):
                video_id = file.split('_')[0]
                base_name = file.replace('.jpg', '')
                txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")
                if video_id in valid_ids and os.path.exists(txt_path):
                    cat = df[df['video_id'] == video_id]['category'].iloc[0]
                    self.samples.append({'base_name': base_name, 'label': self.class_to_idx[cat]})

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{sample['base_name']}.jpg")
        txt_path = os.path.join(self.txt_dir, f"{sample['base_name']}.txt")

        img = Image.open(img_path).convert('RGB')
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        enc = self.tokenizer(
            text,
            max_length=48,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'base_name': sample['base_name'],
            'img': self.transform(img),
            'ids': enc['input_ids'].squeeze(0),
            'mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


class MultimodalClassifier(nn.Module):
    """
    Архитектура нейронной сети для мультимодальной классификации (Late Fusion).

    Использует предварительно обученные модели ResNet18 (для визуальных признаков)
    и DistilBERT (для текстовых признаков). Базовые слои экстракторов признаков
    заморожены для предотвращения переобучения на малых выборках. Обучается
    исключительно финальный полносвязный классификатор.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, ids, mask):
        with torch.no_grad():
            img_feat = self.resnet(img)
            txt_feat = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

        combined_feat = torch.cat((img_feat, txt_feat), dim=1)
        return self.classifier(combined_feat)


def plot_learning_curves(history, total_epochs):
    """
    Визуализация кривых обучения: динамики функции потерь и точности.
    """
    epochs = range(1, total_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Динамика функции потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    if history.get('test_epochs'):
        plt.scatter(history['test_epochs'], history['test_acc'], color='red', label='Test Acc')
    plt.title('Динамика точности (Accuracy)')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader, test_loader, criterion, device, epochs, run_tests=True):
    """
    Выполнение цикла обучения и валидации модели.

    Args:
        model: Обучаемая архитектура нейронной сети.
        train_loader: Загрузчик обучающей выборки.
        val_loader: Загрузчик валидационной выборки.
        test_loader: Загрузчик тестовой выборки.
        criterion: Функция потерь.
        device: Вычислительное устройство (CPU/CUDA).
        epochs (int): Количество эпох обучения.
        run_tests (bool): Флаг проведения тестирования каждые 5 эпох.

    Returns:
        dict: Словарь, содержащий историю значений метрик по эпохам.
    """
    optimizer = optim.AdamW(model.classifier.parameters(), lr=3e-5)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_epochs': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        t_loss, t_corr, t_tot = 0, 0, 0
        for b in train_loader:
            img = b['img'].to(device)
            ids = b['ids'].to(device)
            mask = b['mask'].to(device)
            lbl = b['label'].to(device)

            optimizer.zero_grad()
            out = model(img, ids, mask)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            _, p = torch.max(out, 1)
            t_tot += lbl.size(0)
            t_corr += (p == lbl).sum().item()

        model.eval()
        v_loss, v_corr, v_tot = 0, 0, 0
        with torch.no_grad():
            for b in val_loader:
                img = b['img'].to(device)
                ids = b['ids'].to(device)
                mask = b['mask'].to(device)
                lbl = b['label'].to(device)

                out = model(img, ids, mask)
                loss = criterion(out, lbl)

                v_loss += loss.item()
                _, p = torch.max(out, 1)
                v_tot += lbl.size(0)
                v_corr += (p == lbl).sum().item()

        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))
        history['train_acc'].append(100 * t_corr / t_tot)
        history['val_acc'].append(100 * v_corr / v_tot)

        print(
            f"Эпоха {epoch + 1}/{epochs} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.1f}%")

        if run_tests and (epoch + 1) % 5 == 0:
            test_corr, test_tot = 0, 0
            with torch.no_grad():
                for b in test_loader:
                    img = b['img'].to(device)
                    ids = b['ids'].to(device)
                    mask = b['mask'].to(device)
                    lbl = b['label'].to(device)

                    out = model(img, ids, mask)
                    _, p = torch.max(out, 1)
                    test_tot += lbl.size(0)
                    test_corr += (p == lbl).sum().item()
            acc = 100 * test_corr / test_tot
            history['test_epochs'].append(epoch + 1)
            history['test_acc'].append(acc)
            print(f" Точность на тестовой выборке (Эпоха {epoch + 1}): {acc:.1f}%")

    return history


def main():
    BASE_DIR = os.getcwd()
    DATASET_PATH = '/content/drive/MyDrive/Dataset_Local'
    SPLITS_CSV = os.path.join(BASE_DIR, 'dataset_splits.csv')

    if not os.path.exists(SPLITS_CSV):
        print("Ошибка: Локальный датасет не найден. Требуется предварительный запуск модуля загрузки данных.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое вычислительное устройство: {device}")

    train_ds = MultimodalExperimentDataset(DATASET_PATH, SPLITS_CSV, 'train')
    val_ds = MultimodalExperimentDataset(DATASET_PATH, SPLITS_CSV, 'val')
    test_ds = MultimodalExperimentDataset(DATASET_PATH, SPLITS_CSV, 'test')

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    num_classes = train_ds.num_classes
    criterion = nn.CrossEntropyLoss()

    # Итерация 1: Поиск оптимальной точки останова (Early Stopping)
    print("\nЗапуск первичного цикла обучения (Анализ сходимости):")
    model_v1 = MultimodalClassifier(num_classes).to(device)
    hist_v1 = train_model(model_v1, train_loader, val_loader, test_loader, criterion, device, epochs=20, run_tests=True)
    plot_learning_curves(hist_v1, 20)

    optimal_epoch = np.argmin(hist_v1['val_loss']) + 1
    print(f"\nАнализ функции потерь: Оптимальное количество эпох до переобучения — {optimal_epoch}.")

    # Итерация 2: Финальное обучение
    print(f"\nЗапуск финального цикла обучения ({optimal_epoch} эпох):")
    model_final = MultimodalClassifier(num_classes).to(device)
    hist_final = train_model(model_final, train_loader, val_loader, test_loader, criterion, device,
                             epochs=optimal_epoch, run_tests=False)
    plot_learning_curves(hist_final, optimal_epoch)

    # Оценка качества модели
    print("\nОценка качества классификации на тестовой выборке:")
    model_final.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for b in test_loader:
            img = b['img'].to(device)
            ids = b['ids'].to(device)
            mask = b['mask'].to(device)
            lbl = b['label'].to(device)

            out = model_final(img, ids, mask)
            _, preds = torch.max(out, 1)

            y_true.extend(lbl.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Расчет метрик
    target_names = [test_ds.idx_to_class[i] for i in range(num_classes)]
    print("\nОтчет классификации (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Построение матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Визуализация матрицы ошибок сохранена в файл 'confusion_matrix.png'.")

if __name__ == '__main__':
    main()
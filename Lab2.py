import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


class MultimodalExperimentDataset(Dataset):
    """
    Класс набора данных для загрузки синхронизированных пар (изображение, текст).
    Обеспечивает изоляцию данных: принадлежность кадра к выборке определяется его видео-родителем.
    """

    def __init__(self, data_dir: str, split_csv_path: str, split_type: str = 'train'):
        """
        Инициализирует загрузчик данных.

        Параметры:
            data_dir (str): Путь к директории с данными (images, texts).
            split_csv_path (str): Путь к файлу метаданных распределения.
            split_type (str): Тип формируемой выборки ('train', 'val', 'test').
        """
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

        # Фильтрация файлов согласно принадлежности video_id к текущей выборке
        for file in os.listdir(self.img_dir):
            if file.endswith('.jpg'):
                video_id = file.split('_')[0]
                base_name = file.replace('.jpg', '')
                txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")

                if video_id in valid_ids and os.path.exists(txt_path):
                    category = self.df_split[self.df_split['video_id'] == video_id]['category'].iloc[0]
                    self.samples.append({
                        'base_name': base_name,
                        'label': self.class_to_idx[category]
                    })

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{sample['base_name']}.jpg")
        txt_path = os.path.join(self.txt_dir, f"{sample['base_name']}.txt")

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        enc = self.tokenizer(
            text, max_length=48, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'base_name': sample['base_name'],
            'img': img_tensor,
            'ids': enc['input_ids'].squeeze(0),
            'mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


class MultimodalClassifier(nn.Module):
    """
    Архитектура нейронной сети для классификации кросс-модальных данных
    на основе слияния признаков из ResNet-18 и DistilBERT.
    """

    def __init__(self, num_classes: int):
        """
        Инициализирует модули извлечения признаков и полносвязный классификатор.

        Параметры:
            num_classes (int): Количество целевых классов для предсказания.
        """
        super().__init__()
        # Визуальный энкодер
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # Текстовый энкодер
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Блок классификации (позднее слияние признаков: 512 + 768)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img_feat = self.resnet(img)
        txt_feat = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat((img_feat, txt_feat), dim=1))


def run_full_experiment():
    """
    Основной цикл эксперимента: инициализация данных, обучение модели
    и оценка производительности на тестовой выборке с построением матрицы ошибок.
    """
    CONFIG = {
        'data_dir': '/content/drive/MyDrive/Dataset',
        'split_csv': '/content/drive/MyDrive/Dataset/dataset_splits.csv',
        'epochs': 10,
        'batch_size': 16,
        'lr': 3e-5,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    train_ds = MultimodalExperimentDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'train')
    val_ds = MultimodalExperimentDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'val')
    test_ds = MultimodalExperimentDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'test')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = MultimodalClassifier(num_classes=train_ds.num_classes).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    print(f"Начало процесса обучения на устройстве: {CONFIG['device']}")

    for epoch in range(CONFIG['epochs']):
        model.train()
        t_correct, t_total = 0, 0

        for batch in train_loader:
            img = batch['img'].to(CONFIG['device'])
            ids = batch['ids'].to(CONFIG['device'])
            mask = batch['mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(img, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()

        model.eval()
        v_correct, v_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch['img'].to(CONFIG['device'])
                ids = batch['ids'].to(CONFIG['device'])
                mask = batch['mask'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])

                outputs = model(img, ids, mask)
                _, preds = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        print(
            f"Эпоха {epoch + 1} | Точность (Train): {100 * t_correct / t_total:.1f}% | Точность (Val): {100 * v_correct / v_total:.1f}%")

    # Этап тестирования и сбора метрик
    y_true, y_pred, y_probs = [], [], []
    misclassified_examples = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            img = batch['img'].to(CONFIG['device'])
            ids = batch['ids'].to(CONFIG['device'])
            mask = batch['mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])
            base_names = batch['base_name']

            outputs = model(img, ids, mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_examples.append({
                        'base_name': base_names[i],
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item()
                    })

    print("\nАнализ результатов тестирования:")
    target_names = [test_ds.idx_to_class[i] for i in range(test_ds.num_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names))

    k = min(3, test_ds.num_classes)
    if k > 1:
        top_k_preds = np.argsort(np.array(y_probs), axis=1)[:, -k:]
        correct_k = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
        top_k_acc = correct_k / len(y_true)
        print(f"Метрика Top-{k} Accuracy: {top_k_acc:.4f}")

    # Визуализация матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица классификационных ошибок')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    print("\nРеестр неверных классификаций (первые 10 случаев):")
    for idx, ex in enumerate(misclassified_examples[:10]):
        true_class = test_ds.idx_to_class[ex['true_label']]
        pred_class = test_ds.idx_to_class[ex['pred_label']]
        print(
            f"{idx + 1}. Идентификатор файла: {ex['base_name']} | Истинный класс: '{true_class}' | Предсказанный класс: '{pred_class}'")


if __name__ == '__main__':
    run_full_experiment()
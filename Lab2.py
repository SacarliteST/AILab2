import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict


class VideoMultimodalDataset(Dataset):
    """
    Класс набора данных для загрузки мультимодальных видеоклипов.
    Группирует извлеченные 2D-кадры в 3D-тензоры (последовательности) и склеивает текст субтитров.
    Реализует различные стратегии сэмплирования для обучения и тестирования.
    """

    def __init__(self, data_dir: str, split_csv_path: str, split_type: str = 'train', seq_len: int = 8):
        self.img_dir = os.path.join(data_dir, 'images')
        self.txt_dir = os.path.join(data_dir, 'texts')
        self.seq_len = seq_len
        self.is_test = (split_type == 'test')

        df = pd.read_csv(split_csv_path)
        df_split = df[df['split'] == split_type]

        unique_cats = sorted(df['category'].unique())
        self.class_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        self.idx_to_class = {i: cat for cat, i in self.class_to_idx.items()}
        self.num_classes = len(unique_cats)

        valid_ids = set(df_split['video_id'].tolist())
        self.video_groups = defaultdict(list)

        for file in sorted(os.listdir(self.img_dir)):
            if file.endswith('.jpg'):
                video_id = file.split('_')[0]
                base_name = file.replace('.jpg', '')
                txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")

                if video_id in valid_ids and os.path.exists(txt_path):
                    self.video_groups[video_id].append(base_name)

        self.valid_videos = []
        for vid_id, frames in self.video_groups.items():
            if len(frames) > 0:
                category = df_split[df_split['video_id'] == vid_id]['category'].iloc[0]
                self.valid_videos.append({
                    'video_id': vid_id,
                    'frames': frames,
                    'label': self.class_to_idx[category]
                })

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])

    def __len__(self) -> int:
        return len(self.valid_videos)

    def __getitem__(self, idx: int) -> dict:
        sample = self.valid_videos[idx]
        frame_names = sample['frames']
        total_frames = len(frame_names)

        video_clips = []

        if self.is_test:
            num_clips = 3
            step = max(1, total_frames // num_clips)
            clips_indices = []
            for i in range(num_clips):
                start = i * step
                end = min((i + 1) * step, total_frames) if i < num_clips - 1 else total_frames
                indices = torch.linspace(start, end - 1, self.seq_len).long()
                clips_indices.append(indices)
        else:
            clips_indices = [torch.linspace(0, total_frames - 1, self.seq_len).long()]

        combined_text = ""
        for base_name in frame_names:
            txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + " "

        for indices in clips_indices:
            clip_tensors = []
            for i in indices:
                img_path = os.path.join(self.img_dir, f"{frame_names[i]}.jpg")
                img = Image.open(img_path).convert('RGB')
                clip_tensors.append(self.transform(img))
            video_clips.append(torch.stack(clip_tensors, dim=1))

        final_video_tensor = torch.stack(video_clips, dim=0)

        enc = self.tokenizer(
            combined_text.strip(), max_length=64, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'video_id': sample['video_id'],
            'video': final_video_tensor,
            'ids': enc['input_ids'].squeeze(0),
            'mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


class MultimodalVideoClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.video_resnet = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.video_resnet.fc = nn.Identity()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, video: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vid_feat = self.video_resnet(video)
        txt_feat = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        combined_feat = torch.cat((vid_feat, txt_feat), dim=1)
        return self.classifier(combined_feat)


def get_model_and_optimizer(num_classes, lr, device):
    """Вспомогательная функция для чистой инициализации модели и оптимизатора"""
    model = MultimodalVideoClassifier(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer


def run_full_experiment():
    CONFIG = {
        'data_dir': '/content/drive/MyDrive/Dataset_Local',
        'split_csv': '/content/drive/MyDrive/dataset_splits.csv',
        'search_epochs': 20,  # Эпохи для анализа сходимости
        'batch_size': 4,
        'lr': 3e-5,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    train_ds = VideoMultimodalDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'train')
    val_ds = VideoMultimodalDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'val')
    test_ds = VideoMultimodalDataset(CONFIG['data_dir'], CONFIG['split_csv'], 'test')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()

    print(f"Используемое вычислительное устройство: {CONFIG['device']}")

    print(f"\n--- Запуск первичного цикла обучения (Анализ сходимости на {CONFIG['search_epochs']} эпох) ---")
    model, optimizer = get_model_and_optimizer(train_ds.num_classes, CONFIG['lr'], CONFIG['device'])

    best_val_loss = float('inf')
    optimal_epoch = 1

    for epoch in range(CONFIG['search_epochs']):
        model.train()
        for batch in train_loader:
            video = batch['video'].squeeze(1).to(CONFIG['device'])
            ids = batch['ids'].to(CONFIG['device'])
            mask = batch['mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(video, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].squeeze(1).to(CONFIG['device'])
                ids = batch['ids'].to(CONFIG['device'])
                mask = batch['mask'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])

                outputs = model(video, ids, mask)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        epoch_val_loss = v_loss / v_total if v_total > 0 else 0
        epoch_val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        print(
            f"Эпоха {epoch + 1}/{CONFIG['search_epochs']} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            optimal_epoch = epoch + 1

    print(f"\nАнализ функции потерь: Оптимальное количество эпох до переобучения — {optimal_epoch}.")

    print(f"\n--- Запуск финального цикла обучения ({optimal_epoch} эпох) ---")

    model, optimizer = get_model_and_optimizer(train_ds.num_classes, CONFIG['lr'], CONFIG['device'])

    for epoch in range(optimal_epoch):
        model.train()
        for batch in train_loader:
            video = batch['video'].squeeze(1).to(CONFIG['device'])
            ids = batch['ids'].to(CONFIG['device'])
            mask = batch['mask'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(video, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].squeeze(1).to(CONFIG['device'])
                ids = batch['ids'].to(CONFIG['device'])
                mask = batch['mask'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])

                outputs = model(video, ids, mask)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        epoch_val_loss = v_loss / v_total if v_total > 0 else 0
        epoch_val_acc = 100 * v_correct / v_total if v_total > 0 else 0
        print(f"Эпоха {epoch + 1}/{optimal_epoch} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%")

    y_true, y_pred, y_probs = [], [], []
    misclassified_examples = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            video = batch['video'].to(CONFIG['device'])
            batch_size, num_clips, c, t, h, w = video.size()
            video_input = video.view(batch_size * num_clips, c, t, h, w)

            ids = batch['ids'].to(CONFIG['device'])
            mask = batch['mask'].to(CONFIG['device'])
            ids_input = ids.repeat_interleave(num_clips, dim=0)
            mask_input = mask.repeat_interleave(num_clips, dim=0)

            labels = batch['label'].to(CONFIG['device'])
            video_ids = batch['video_id']

            outputs = model(video_input, ids_input, mask_input)
            probs = torch.softmax(outputs, dim=1).view(batch_size, num_clips, -1)
            avg_probs = torch.mean(probs, dim=1)

            _, preds = torch.max(avg_probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(avg_probs.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_examples.append({
                        'video_id': video_ids[i],
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item()
                    })

    print("\nАнализ результатов тестирования:")
    target_names = [test_ds.idx_to_class[i] for i in range(test_ds.num_classes)]
    labels_indices = list(range(test_ds.num_classes))

    print(classification_report(y_true, y_pred, labels=labels_indices, target_names=target_names, zero_division=0))

    k = min(3, test_ds.num_classes)
    if k > 1:
        top_k_preds = np.argsort(np.array(y_probs), axis=1)[:, -k:]
        correct_k = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
        top_k_acc = correct_k / len(y_true) if len(y_true) > 0 else 0
        print(f"Метрика Top-{k} Accuracy: {top_k_acc:.4f}")

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=labels_indices)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
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
                f"{idx + 1}. Видео ID: {ex['video_id']} | Истинный класс: '{true_class}' | Предсказанный класс: '{pred_class}'")


if __name__ == '__main__':
    run_full_experiment()
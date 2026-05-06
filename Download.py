import os
import glob
import random
import cv2
import webvtt
import pandas as pd
import yt_dlp
from sklearn.model_selection import train_test_split


def setup_directories(base_path: str) -> tuple:
    """
    Инициализирует иерархию директорий для хранения мультимодальных данных.

    Параметры:
        base_path (str): Корневая директория локального датасета.

    Возвращает:
        tuple: Кортеж, содержащий абсолютные пути к директориям
               изображений (img_dir), текстов (txt_dir) и субтитров (vtt_dir).
    """
    img_dir = os.path.join(base_path, 'images')
    txt_dir = os.path.join(base_path, 'texts')
    vtt_dir = os.path.join(base_path, 'subtitles')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(vtt_dir, exist_ok=True)

    return img_dir, txt_dir, vtt_dir


def extract_frames_and_text(
        video_path: str,
        vtt_path: str,
        video_id: str,
        img_dir: str,
        txt_dir: str,
        target_fps: float = 0.5
) -> bool:
    """
    Извлекает кадры из видеопотока с заданной частотой и сопоставляет их с текстом субтитров.

    Параметры:
        video_path (str): Путь к локальному видеофайлу.
        vtt_path (str): Путь к файлу субтитров формата WebVTT.
        video_id (str): Уникальный идентификатор видеоролика.
        img_dir (str): Директория для сохранения извлеченных кадров.
        txt_dir (str): Директория для сохранения текстовых аннотаций.
        target_fps (float): Целевая частота извлечения кадров (кадров в секунду).

    Возвращает:
        bool: True, если извлечен хотя бы один валидный кадр, иначе False.
    """
    try:
        subs = webvtt.read(vtt_path)
    except Exception:
        return False

    if not subs:
        return False

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps / target_fps)
    saved_count, current_frame = 0, 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = current_frame / fps
        active_text = ""

        # Временная синхронизация кадра с блоком субтитров
        for sub in subs:
            t_s = sub.start.split(':')
            start_sec = float(t_s[0]) * 3600 + float(t_s[1]) * 60 + float(t_s[2])
            t_e = sub.end.split(':')
            end_sec = float(t_e[0]) * 3600 + float(t_e[1]) * 60 + float(t_e[2])

            if start_sec <= current_sec <= end_sec:
                active_text = sub.text.replace('\n', ' ').strip()
                break

        if active_text:
            fname = f"{video_id}_f{current_frame}"
            cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), frame)

            with open(os.path.join(txt_dir, f"{fname}.txt"), 'w', encoding='utf-8') as f:
                f.write(active_text)
            saved_count += 1

        current_frame += step
        if saved_count >= 20:
            break

    cap.release()
    return saved_count > 0


def download_video(video_id: str, base_path: str, cookies_path: str) -> bool:
    """
    Осуществляет загрузку видео и субтитров с хостинга с последующим извлечением данных.

    Параметры:
        video_id (str): Уникальный идентификатор видеоролика.
        base_path (str): Корневая директория для сохранения файлов.
        cookies_path (str): Путь к файлу авторизационных cookie.

    Возвращает:
        bool: Статус успешности операции (True при успешном извлечении данных).
    """
    url = f'https://www.youtube.com/watch?v={video_id}'
    img_dir, txt_dir, vtt_dir = setup_directories(base_path)
    video_path, vtt_path = None, None

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best',
        'writeautomaticsub': True,
        'subtitleslangs': ['en.*'],
        'outtmpl': {
            'default': os.path.join(base_path, f'{video_id}.%(ext)s'),
            'subtitle': os.path.join(vtt_dir, f'{video_id}.%(ext)s')
        },
        'cookiefile': cookies_path,
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'quiet': True,
        'no_warnings': True,
        'retries': 5,
        'socket_timeout': 30,
    }

    success = False
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        possible_videos = glob.glob(os.path.join(base_path, f'{video_id}.*'))
        possible_vtts = glob.glob(os.path.join(vtt_dir, f'{video_id}*.vtt'))

        current_vtt = possible_vtts[0] if possible_vtts else None
        for file in possible_videos:
            if not file.endswith(('.vtt', '.txt', '.part')):
                video_path = file
                break

        if video_path and os.path.exists(video_path) and current_vtt:
            success = extract_frames_and_text(video_path, current_vtt, video_id, img_dir, txt_dir)
            vtt_path = current_vtt

    except Exception:
        pass
    finally:
        # Очистка временных медиафайлов для экономии дискового пространства
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except PermissionError:
                pass
        if vtt_path and os.path.exists(vtt_path):
            try:
                os.remove(vtt_path)
            except PermissionError:
                pass

    return success


if __name__ == '__main__':
    BASE_DIR = os.getcwd()
    CSV_PATH = os.path.join(BASE_DIR, 'HowTo100M_v1.csv')
    COOKIES_PATH = os.path.join(BASE_DIR, 'cookies.txt')
    DATASET_PATH = os.path.join(BASE_DIR, 'Dataset_Local')
    SPLITS_CSV = os.path.join(BASE_DIR, 'dataset_splits.csv')
    PROGRESS_FILE = os.path.join(BASE_DIR, 'download_progress.csv')

    VIDEOS_PER_CAT = 100
    NUM_CATEGORIES = 10

    if not os.path.exists(CSV_PATH) or not os.path.exists(COOKIES_PATH):
        print("Системная ошибка: Отсутствуют конфигурационные файлы.")
        exit()

    downloaded_vids = set()
    category_counts = {}
    selected_categories = []

    if os.path.exists(PROGRESS_FILE):
        prog_df = pd.read_csv(PROGRESS_FILE)
        downloaded_vids = set(prog_df['video_id'].tolist())
        selected_categories = prog_df['category'].unique().tolist()
        category_counts = prog_df['category'].value_counts().to_dict()
    else:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            f.write("video_id,category\n")

    df = pd.read_csv(CSV_PATH, usecols=['video_id', 'category_1']).dropna()
    all_categories = df['category_1'].unique().tolist()

    if len(selected_categories) < NUM_CATEGORIES:
        needed = NUM_CATEGORIES - len(selected_categories)
        available = list(set(all_categories) - set(selected_categories))
        selected_categories += random.sample(available, min(needed, len(available)))

    for category in selected_categories:
        current_count = category_counts.get(category, 0)
        if current_count >= VIDEOS_PER_CAT:
            continue

        video_ids = df[df['category_1'] == category]['video_id'].drop_duplicates().tolist()
        random.shuffle(video_ids)

        for vid in video_ids:
            if current_count >= VIDEOS_PER_CAT:
                break
            if vid in downloaded_vids:
                continue

            if download_video(vid, DATASET_PATH, COOKIES_PATH):
                current_count += 1
                downloaded_vids.add(vid)
                with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{vid},{category}\n")

    # Формирование независимых подвыборок по уникальным идентификаторам видео
    final_prog_df = pd.read_csv(PROGRESS_FILE)
    if not final_prog_df.empty:
        train_val, test = train_test_split(
            final_prog_df, test_size=0.15, stratify=final_prog_df['category'], random_state=42
        )
        train, val = train_test_split(
            train_val, test_size=0.176, stratify=train_val['category'], random_state=42
        )

        for dataset_split, split_name in zip([train, val, test], ['train', 'val', 'test']):
            dataset_split['split'] = split_name

        pd.concat([train, val, test]).to_csv(SPLITS_CSV, index=False)
        print("Формирование метаданных выборок успешно завершено.")
    else:
        print("Ошибка: Набор данных не сформирован.")
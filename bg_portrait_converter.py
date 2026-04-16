#!/usr/bin/env python3
"""
Baldur's Gate EE Portrait Converter
Конвертирует изображения в портреты для Baldur's Gate EE с использованием AI.
"""

import sys
import random
from pathlib import Path
from typing import Tuple, Optional, List
import warnings

import click
from PIL import Image
import cv2
import numpy as np

warnings.filterwarnings('ignore')

# Размеры портретов для BG EE
PORTRAIT_MODES = {
    'face': {
        'name': 'Только лицо',
        'large': (210, 330),
        'medium': (169, 266),
        'h_expansion': {'large': 2.2, 'medium': 1.8},  # Горизонтальное расширение
        'v_expansion': {'large': 2.2, 'medium': 1.8},  # Вертикальное расширение
        'v_offset': {'large': 0.0, 'medium': 0.0}
    }
}

# Default режим
DEFAULT_MODE = 'face'


class FaceDetector:
    """Детектор лиц с использованием OpenCV Haar Cascades"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_face(self, image_cv2: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Детектирует лицо на изображении.
        Возвращает (x, y, width, height) bounding box или None если лицо не найдено.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
        # Стандартные параметры - самые надежные
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Берем наибольшее обнаруженное лицо
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, width, height = face
        
        return (x, y, width, height)


def crop_to_face(image_path: str, face_detector: FaceDetector, target_size: Tuple[int, int], 
                 h_expansion: float = 1.8, v_expansion: float = 1.8) -> Optional[Image.Image]:
    """
    Обрезает изображение к лицу и масштабирует до целевого размера.
    h_expansion: горизонтальное расширение области вокруг лица (1.0 = только лицо)
    v_expansion: вертикальное расширение области вокруг лица (1.0 = только лицо)
    """
    # Загружаем изображение с OpenCV
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return None
    
    # Детектируем лицо
    face_bbox = face_detector.detect_face(img_cv2)
    
    h, w = img_cv2.shape[:2]
    
    if face_bbox:
        x, y, face_w, face_h = face_bbox
        
        # Центр лица
        face_center_x = x + face_w // 2
        face_center_y = y + face_h // 2
        
        # Расширяем область вокруг лица с использованием переданных параметров
        crop_w = int(face_w * h_expansion)
        crop_h = int(face_h * v_expansion)
        
        # Вычисляем координаты обрезки
        crop_x1 = int(face_center_x - crop_w // 2)
        crop_y1 = int(face_center_y - crop_h // 2)
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        
        # Убеждаемся что в границах изображения
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)
        
        # Если обрезка слишком мала, берем все изображение
        if (crop_x2 - crop_x1) < 50 or (crop_y2 - crop_y1) < 50:
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w, h
        
        img_cropped = img_cv2[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        # Если лицо не найдено, берем центральную часть
        img_cropped = img_cv2
    
    # Конвертируем в PIL
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    # Масштабируем изображение чтобы полностью заполнить портрет
    # Вычисляем коэффициент масштабирования
    img_ratio = pil_image.width / pil_image.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Изображение шире - масштабируем по высоте
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    else:
        # Изображение уже - масштабируем по ширине
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    
    # Изменяем размер
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Вырезаем центральную часть нужного размера
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    
    result = pil_image.crop((left, top, right, bottom))
    
    return result


def save_portraits(output_dir: Path, portrait_code: str, image_large: Image.Image, image_medium: Image.Image) -> Tuple[str, str]:
    """
    Сохраняет портреты в формате BMP 24-bit.
    
    Использует префикс 'A' и уникальный 4-значный код:
    - A[code]L.bmp — Large портрет (210x330)
    - A[code]M.bmp — Medium портрет (169x266)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Убеждаемся что изображения в RGB (для BMP 24-bit)
    if image_large.mode != 'RGB':
        image_large = image_large.convert('RGB')
    if image_medium.mode != 'RGB':
        image_medium = image_medium.convert('RGB')
    
    # Сохраняем Large портрет (суффикс L)
    large_filename = f"A{portrait_code}L.bmp"
    large_path = output_dir / large_filename
    image_large.save(large_path, format='BMP')
    
    # Сохраняем Medium портрет (суффикс M)
    medium_filename = f"A{portrait_code}M.bmp"
    medium_path = output_dir / medium_filename
    image_medium.save(medium_path, format='BMP')
    
    return str(large_path), str(medium_path)


def process_image(image_path: str, output_dir: Path, portrait_code: str, face_detector: FaceDetector, mode: str = DEFAULT_MODE) -> bool:
    """
    Обрабатывает одно изображение.
    Возвращает True если успешно, False если ошибка.
    """
    try:
        mode_config = PORTRAIT_MODES[mode]
        large_size = mode_config['large']
        medium_size = mode_config['medium']
        
        # Получаем параметры расширения
        large_h_exp = mode_config['h_expansion']['large']
        large_v_exp = mode_config['v_expansion']['large']
        medium_h_exp = mode_config['h_expansion']['medium']
        medium_v_exp = mode_config['v_expansion']['medium']
        
        # Обрезаем к лицу и масштабируем с нужными параметрами
        image_large = crop_to_face(image_path, face_detector, large_size, 
                                    h_expansion=large_h_exp, v_expansion=large_v_exp)
        image_medium = crop_to_face(image_path, face_detector, medium_size,
                                     h_expansion=medium_h_exp, v_expansion=medium_v_exp)
        
        if image_large is None or image_medium is None:
            return False
        
        # Сохраняем с уникальным кодом
        save_portraits(output_dir, portrait_code, image_large, image_medium)
        return True
    
    except Exception as e:
        click.echo(f"Ошибка при обработке {image_path}: {str(e)}", err=True)
        return False


def get_supported_formats() -> List[str]:
    """Возвращает поддерживаемые форматы изображений"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


@click.command()
@click.option(
    '--input',
    '-i',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help='Папка с изображениями для обработки'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help='Папка для сохранения портретов (по умолчанию: input_folder/bg_portraits)'
)
def main(input: Optional[str] = None, output: Optional[str] = None):
    """
    Конвертирует изображения в портреты для Baldur's Gate EE.
    
    Создает 2 размера портретов: Large (210x330) и Medium (169x266) в формате BMP 24-bit.
    """
    
    # Используем режим по умолчанию (face)
    mode = DEFAULT_MODE
    
    click.echo()
    click.echo(f"[*] Режим: {PORTRAIT_MODES[mode]['name']}")
    click.echo()
    
    # Если папка не указана, просим у пользователя
    if not input:
        # Определяем папку input в директории проекта по умолчанию
        default_input = Path(__file__).parent / 'input'
        input = click.prompt(
            'Введите путь к папке с изображениями',
            type=click.Path(exists=True),
            default=str(default_input)
        )
    
    input_dir = Path(input) if input else Path('.')
    
    # Проверяем что это папка
    if not input_dir.is_dir():
        click.echo("Ошибка: указанный путь не является папкой", err=True)
        sys.exit(1)
    
    # Определяем выходную папку
    if not output:
        output_dir = input_dir / 'bg_portraits'
    else:
        output_dir = Path(output)
    
    click.echo(f"[*] Входная папка: {input_dir}")
    click.echo(f"[*] Выходная папка: {output_dir}")
    click.echo()
    
    # Ищем все поддерживаемые изображения
    supported_formats = get_supported_formats()
    image_files = []
    for fmt in supported_formats:
        image_files.extend(input_dir.glob(f'*{fmt}'))
        image_files.extend(input_dir.glob(f'*{fmt.upper()}'))
    
    image_files = list(set(image_files))  # Убираем дубликаты
    image_files.sort()
    
    if not image_files:
        click.echo("Предупреждение: в папке не найдено изображений", err=True)
        sys.exit(1)
    
    click.echo(f"[+] Найдено {len(image_files)} изображение(й)")
    
    # Проверка на количество портретов (максимум 99999 для соблюдения лимита 7 символов)
    if len(image_files) > 99999:
        click.echo("Ошибка: слишком много изображений! Максимум 99,999 портретов (ограничение двигателя).", err=True)
        sys.exit(1)
    
    click.echo()
    
    # Инициализируем детектор лиц
    click.echo("[*] Инициализация детектора лиц...")
    face_detector = FaceDetector()
    
    # Генерируем уникальные коды для портретов (4-значные числа от 1000 до 9999)
    portrait_codes = random.sample(range(1000, 10000), min(len(image_files), 9000))
    
    # Обрабатываем изображения
    successful = 0
    failed = 0
    created_files = []
    
    with click.progressbar(image_files, label='Обработка') as bar:
        for idx, image_path in enumerate(bar):
            portrait_code = str(portrait_codes[idx])
            if process_image(str(image_path), output_dir, portrait_code, face_detector, mode=mode):
                successful += 1
                created_files.append(f"A{portrait_code}L.bmp / A{portrait_code}M.bmp")
            else:
                failed += 1
    
    click.echo()
    click.echo(f"[+] Успешно обработано: {successful}")
    if failed > 0:
        click.echo(f"[-] Ошибок: {failed}")
    
    if successful > 0:
        click.echo()
        click.echo(f"[*] Созданные портреты: {successful} пар")
        if successful <= 10:
            for filename in created_files:
                click.echo(f"  • {filename}")
        else:
            click.echo(f"  • {created_files[0]}")
            click.echo(f"  • {created_files[1]}")
            click.echo(f"  • ...")
            click.echo(f"  • {created_files[-1]}")
    
    click.echo()
    click.echo(f"[*] Портреты сохранены в: {output_dir}")


if __name__ == '__main__':
    main()

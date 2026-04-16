#!/usr/bin/env python3
"""
Ручное создание портретов с использованием GUI
Позволяет нарисовать прямоугольник вокруг нужной области лица
"""

import sys
import random
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import click

# Размеры портретов для BG EE
PORTRAIT_SIZES = {
    'large': (210, 330),
    'medium': (169, 266),
}

DEFAULT_MODE = 'face'


class PortraitGUI:
    """GUI для ручного выделения области портрета"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        
        # Масштабируем если слишком большое
        h, w = self.display_image.shape[:2]
        if w > 800 or h > 600:
            scale = min(800 / w, 600 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.display_image = cv2.resize(self.display_image, (new_w, new_h))
            self.scale = scale
        else:
            self.scale = 1.0
    
    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_start = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end = (x, y)
            # Показываем превью прямоугольника
            preview = self.display_image.copy()
            cv2.rectangle(preview, self.rect_start, self.rect_end, (0, 255, 0), 2)
            cv2.imshow('Portrait GUI', preview)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect_end = (x, y)
            preview = self.display_image.copy()
            cv2.rectangle(preview, self.rect_start, self.rect_end, (0, 255, 0), 2)
            cv2.imshow('Portrait GUI', preview)
    
    def show_interface(self, size_type: str = 'large') -> Optional[Tuple[int, int, int, int]]:
        """
        Показывает GUI для выделения области.
        size_type: 'large' или 'medium'
        Возвращает (x, y, width, height) в координатах оригинального изображения или None
        
        Управление:
        - ЛКМ: нарисовать прямоугольник
        - Enter: подтвердить выделение
        - ESC: отменить
        """
        cv2.namedWindow('Portrait GUI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Portrait GUI', self.mouse_callback)
        
        cv2.imshow('Portrait GUI', self.display_image)
        
        size_label = "ПОЛНЫЙ ПОРТРЕТ (L)" if size_type == 'large' else "ПРИБЛИЖЕННОЕ ЛИЦО (M)"
        
        print("\n" + "="*50)
        print(f"ВЫДЕЛЕНИЕ: {size_label}")
        print("УПРАВЛЕНИЕ:")
        print("  1. Нарисуйте прямоугольник мышкой")
        print("  2. Нажмите ENTER для подтверждения")
        print("  3. Нажмите ESC для отмены")
        print("="*50 + "\n")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            
            elif key == 13:  # ENTER
                if self.rect_start and self.rect_end:
                    cv2.destroyAllWindows()
                    
                    # Преобразуем координаты обратно в исходный размер
                    x1 = int(min(self.rect_start[0], self.rect_end[0]) / self.scale)
                    y1 = int(min(self.rect_start[1], self.rect_end[1]) / self.scale)
                    x2 = int(max(self.rect_start[0], self.rect_end[0]) / self.scale)
                    y2 = int(max(self.rect_start[1], self.rect_end[1]) / self.scale)
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 20 and height > 20:
                        return (x1, y1, width, height)
                    else:
                        print("Область слишком мала! Попробуйте еще раз.")
                else:
                    print("Сначала нарисуйте прямоугольник!")


def crop_and_save(image_path: str, crop_rects: dict, 
                  output_dir: Path, portrait_code: str) -> bool:
    """
    Обрезает изображение по указанным прямоугольникам и сохраняет портреты.
    crop_rects: {'large': (x, y, w, h), 'medium': (x, y, w, h)}
    """
    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            return False
        
        # Обрезаем и сохраняем каждый размер
        for size_name, (target_w, target_h) in PORTRAIT_SIZES.items():
            if size_name not in crop_rects:
                continue
            
            x, y, width, height = crop_rects[size_name]
            
            # Обрезаем
            img_cropped = img_cv2[y:y+height, x:x+width]
            
            # Преобразуем в PIL
            img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            
            # Вычисляем aspect ratio
            img_ratio = img_pil.width / img_pil.height
            target_ratio = target_w / target_h
            
            if img_ratio > target_ratio:
                # Изображение шире, масштабируем по высоте
                new_h = target_h
                new_w = int(target_h * img_ratio)
            else:
                # Изображение уже, масштабируем по ширине
                new_w = target_w
                new_h = int(target_w / img_ratio)
            
            img_scaled = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Центрируем и обрезаем до целевого размера
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            
            img_final = img_scaled.crop((left, top, right, bottom))
            
            # Определяем суффикс размера
            suffix = 'L' if size_name == 'large' else 'M'
            filename = f"A{portrait_code}{suffix}.bmp"
            filepath = output_dir / filename
            
            img_final.save(filepath, format='BMP')
            print(f"  Сохранено: {filename}")
        
        return True
    
    except Exception as e:
        print(f"Ошибка при обрезке: {str(e)}", file=sys.stderr)
        return False


@click.command()
@click.option('-i', '--input', 'input_dir', 
              type=click.Path(exists=True),
              help='Папка с входными изображениями')
@click.option('-o', '--output', 'output_dir',
              type=click.Path(),
              help='Папка для выходных портретов')
def main(input_dir: Optional[str], output_dir: Optional[str]):
    """
    GUI приложение для ручного создания портретов для Baldur's Gate EE
    """
    click.clear()
    click.echo("╔════════════════════════════════════════╗")
    click.echo("║   Baldur's Gate EE Portrait GUI        ║")
    click.echo("║   Ручное создание портретов             ║")
    click.echo("╚════════════════════════════════════════╝\n")
    
    # Определяем папку входа
    if not input_dir:
        default_input = Path(__file__).parent / 'input'
        input_path = Path(click.prompt('Папка с изображениями', default=str(default_input)))
    else:
        input_path = Path(input_dir)
    
    if not input_path.exists():
        click.echo(f"Ошибка: папка не найдена: {input_path}", err=True)
        return
    
    # Ищем изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        click.echo(f"Ошибка: в папке нет изображений: {input_path}", err=True)
        return
    
    image_files = sorted(image_files)
    
    # Определяем папку выхода
    if not output_dir:
        output_path = input_path / 'bg_portraits'
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Найдено {len(image_files)} изображение(й)\n")
    
    # Генерируем уникальные коды для всех изображений
    portrait_codes = random.sample(range(1000, 10000), len(image_files))
    
    processed = 0
    skipped = 0
    
    for image_file, portrait_code in zip(image_files, portrait_codes):
        click.echo(f"\n{'='*50}")
        click.echo(f"Файл: {image_file.name}")
        click.echo(f"Код портрета: A{portrait_code}")
        click.echo(f"{'='*50}")
        
        try:
            gui = PortraitGUI(str(image_file))
            
            # Сначала выделяем область для Large (полный портрет)
            click.echo("\n>>> Шаг 1 из 2: Выделение для LARGE (полный портрет)")
            crop_large = gui.show_interface('large')
            
            if crop_large is None:
                click.echo("⊘ Пропущено пользователем")
                skipped += 1
                continue
            
            # Сбрасываем состояние рисования для второго выделения
            gui.rect_start = None
            gui.rect_end = None
            gui.drawing = False
            
            # Теперь выделяем область для Medium (приближенное лицо)
            click.echo("\n>>> Шаг 2 из 2: Выделение для MEDIUM (приближенное лицо)")
            crop_medium = gui.show_interface('medium')
            
            if crop_medium is None:
                click.echo("⊘ Пропущено пользователем на шаге 2")
                skipped += 1
                continue
            
            # Сохраняем оба портрета
            crop_rects = {
                'large': crop_large,
                'medium': crop_medium
            }
            
            if crop_and_save(str(image_file), crop_rects, output_path, str(portrait_code)):
                click.echo(f"✓ Портреты созданы успешно!")
                processed += 1
            else:
                click.echo(f"✗ Ошибка при обработке файла", err=True)
                skipped += 1
        
        except Exception as e:
            click.echo(f"✗ Ошибка: {str(e)}", err=True)
            skipped += 1
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Завершено!")
    click.echo(f"Обработано: {processed} портретов")
    click.echo(f"Пропущено: {skipped} портретов")
    click.echo(f"Выходная папка: {output_path}")
    click.echo(f"{'='*50}\n")


if __name__ == '__main__':
    main()

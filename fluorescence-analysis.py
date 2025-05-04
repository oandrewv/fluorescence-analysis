import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import math
import re

class FluorescenceAnalyzer:
    def __init__(self):
        # Калибровочная кривая: известные концентрации и соответствующие интенсивности
        self.calibration_curve = None
        self.model = None
        self.roi_coords = None  # Координаты области интереса (ROI)
        self.r2 = None  # Коэффициент детерминации
        
    def select_roi(self, image_path):
        """Позволяет пользователю выбрать область интереса на изображении."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Конвертация в RGB для отображения
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Показать изображение и выбрать ROI
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.title("Кликните на 4 точки, чтобы выбрать область интереса (тест-полоску)")
        plt.axis('on')
        
        coords = plt.ginput(4, timeout=0)
        plt.close()
        
        if len(coords) != 4:
            raise ValueError("Необходимо выбрать ровно 4 точки")
        
        self.roi_coords = np.array(coords, dtype=np.int32)
        return self.roi_coords
    
    def preprocess_image(self, image_path):
        """Предобработка изображения и извлечение ROI."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Если ROI не выбран, используем все изображение
        if self.roi_coords is None:
            print("ROI не выбран, используем все изображение")
            return image
        
        # Создание маски ROI
        mask = np.zeros_like(image[:, :, 0])
        cv2.fillPoly(mask, [self.roi_coords], 255)
        
        # Применение маски
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image
    
    def extract_fluorescence_intensity(self, image_path):
        """Извлечение интенсивности флуоресценции из изображения."""
        processed_image = self.preprocess_image(image_path)
        
        # Преобразование в HSV для лучшего анализа
        hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        
        # Выделение наиболее ярких областей (флуоресценция обычно яркая)
        # Фокусируемся на V (яркость) и S (насыщенность) в HSV
        h, s, v = cv2.split(hsv_image)
        
        # Создаем маску ненулевых значений
        mask = np.where(v > 0, 1, 0).astype(np.uint8)
        
        # Если маска пуста, вернуть нулевую интенсивность
        if np.sum(mask) == 0:
            return 0
        
        # Расчет средней интенсивности в канале V для ненулевых пикселей
        mean_intensity = np.sum(v * mask) / np.sum(mask)
        
        # Также учитываем насыщенность цвета (S)
        mean_saturation = np.sum(s * mask) / np.sum(mask)
        
        # Комбинированная метрика (можно настроить веса)
        fluorescence_intensity = 0.7 * mean_intensity + 0.3 * mean_saturation
        
        return fluorescence_intensity
    
    def extract_concentration_from_filename(self, filename):
        """Извлекает значение концентрации из имени файла.
        Поддерживает форматы:
        - conc_1.23.jpg
        - conc-1.23.jpg
        - 1.23_conc.jpg
        - 1.23-conc.jpg
        - 1.23.jpg
        """
        # Сначала ищем числа в имени файла (без расширения)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Регулярное выражение для поиска чисел с десятичной точкой
        # Поддерживает форматы 1.23, 1,23 (заменяет запятую на точку)
        number_pattern = r'\d+[.,]?\d*'
        
        matches = re.findall(number_pattern, base_name)
        if not matches:
            raise ValueError(f"Не удалось извлечь концентрацию из имени файла: {filename}")
        
        # Берем первое найденное число и заменяем запятую на точку, если она есть
        concentration = float(matches[0].replace(',', '.'))
        return concentration
    
    def build_calibration_curve(self, calibration_images, known_concentrations=None):
        """Построение линейной калибровочной кривой в координатах 'интенсивность от log10(концентрация)'.
        Если known_concentrations не указаны, значения берутся из имен файлов.
        """
        # Если концентрации не предоставлены, извлекаем их из имен файлов
        if known_concentrations is None:
            known_concentrations = []
            for img_path in calibration_images:
                try:
                    conc = self.extract_concentration_from_filename(img_path)
                    known_concentrations.append(conc)
                    print(f"Извлечена концентрация {conc} из файла {os.path.basename(img_path)}")
                except ValueError as e:
                    raise ValueError(f"Ошибка при извлечении концентрации: {str(e)}")
        
        if len(calibration_images) != len(known_concentrations):
            raise ValueError("Количество изображений должно соответствовать количеству известных концентраций")
        
        # Проверка на отрицательные или нулевые концентрации перед логарифмированием
        for i, conc in enumerate(known_concentrations):
            if conc <= 0:
                raise ValueError(f"Концентрация #{i+1} ({conc}) должна быть положительной для логарифмического преобразования")
        
        intensities = []
        for image_path in calibration_images:
            intensity = self.extract_fluorescence_intensity(image_path)
            intensities.append(intensity)
        
        # Логарифмирование концентраций по основанию 10
        log_concentrations = [math.log10(conc) for conc in known_concentrations]
        
        # Сохранение данных для калибровочной кривой
        self.calibration_curve = {
            'concentrations': known_concentrations,
            'log_concentrations': log_concentrations,
            'intensities': intensities
        }
        
        # Создание линейной модели (y = интенсивность, x = log10(концентрация))
        # Обратите внимание, что мы меняем X и y местами, так как строим зависимость
        # интенсивность от log(концентрация), а не наоборот
        X = np.array(log_concentrations).reshape(-1, 1)
        y = np.array(intensities)
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.model = model
        
        # Вычисление R^2 для оценки качества модели
        y_pred = model.predict(X)
        self.r2 = r2_score(y, y_pred)
        
        # Сохраняем коэффициенты линейной регрессии для отображения
        self.slope = model.coef_[0]
        self.intercept = model.intercept_
        
        return self.r2
    
    def predict_concentration(self, image_path):
        """Предсказание концентрации аналита по изображению."""
        if self.model is None:
            raise ValueError("Сначала необходимо построить калибровочную кривую")
        
        intensity = self.extract_fluorescence_intensity(image_path)
        
        # Обратное преобразование интенсивности в log_concentration
        # Используем обратную формулу: log_conc = (intensity - intercept) / slope
        if self.slope == 0:
            raise ValueError("Наклон калибровочной кривой равен нулю, невозможно предсказать концентрацию")
        
        log_concentration = (intensity - self.intercept) / self.slope
        
        # Обратное логарифмирование для получения концентрации
        concentration = 10 ** log_concentration
        
        return concentration
    
    def visualize_calibration_curve(self, save_path=None):
        """Визуализация линейной калибровочной кривой в координатах 'интенсивность от log10(концентрация)'."""
        if self.calibration_curve is None:
            raise ValueError("Сначала необходимо построить калибровочную кривую")
        
        plt.figure(figsize=(10, 6))
        
        # Исходные точки
        plt.scatter(
            self.calibration_curve['log_concentrations'], 
            self.calibration_curve['intensities'], 
            color='blue', 
            label='Калибровочные точки'
        )
        
        # Построение линии с более высоким разрешением
        log_conc_range = np.linspace(
            min(self.calibration_curve['log_concentrations']), 
            max(self.calibration_curve['log_concentrations']), 
            100
        ).reshape(-1, 1)
        
        intensity_pred = self.model.predict(log_conc_range)
        
        # Построение линии регрессии
        plt.plot(log_conc_range, intensity_pred, color='red', 
                 label=f'Калибровочная линия: y = {self.slope:.2f}x + {self.intercept:.2f}')
        
        plt.xlabel('log₁₀(Концентрация)')
        plt.ylabel('Интенсивность флуоресценции')
        plt.title('Калибровочная кривая')
        plt.legend()
        plt.grid(True)
        
        # Добавляем значение R² крупно в углу графика
        plt.text(0.05, 0.95, f'R² = {self.r2:.4f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Калибровочная кривая сохранена в {save_path}")
        
        return plt.gcf()
    
    def analyze_sample_image(self, image_path, show_visualization=True, save_visualization=None):
        """Полный анализ изображения образца с визуализацией."""
        processed_image = self.preprocess_image(image_path)
        intensity = self.extract_fluorescence_intensity(image_path)
        concentration = self.predict_concentration(image_path)
        
        # Вычисляем log10 концентрации для отображения на графике
        log_concentration = math.log10(concentration) if concentration > 0 else float('-inf')
        
        if show_visualization:
            plt.figure(figsize=(12, 8))
            
            # Показываем исходное изображение
            plt.subplot(2, 2, 1)
            original_image = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            plt.imshow(original_rgb)
            plt.title('Исходное изображение')
            plt.axis('off')
            
            # Показываем обработанное изображение с ROI
            plt.subplot(2, 2, 2)
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            plt.imshow(processed_rgb)
            plt.title('Область интереса (ROI)')
            plt.axis('off')
            
            # Показываем интенсивность в виде гистограммы
            plt.subplot(2, 2, 3)
            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mask = np.where(v > 0, 1, 0).astype(np.uint8)
            non_zero_v = v[mask > 0]
            if len(non_zero_v) > 0:
                plt.hist(non_zero_v.flatten(), bins=50, color='blue', alpha=0.7)
                plt.axvline(x=np.mean(non_zero_v), color='red', linestyle='--', 
                           label=f'Средняя интенсивность: {np.mean(non_zero_v):.2f}')
                plt.legend()
            plt.title('Гистограмма яркости')
            plt.xlabel('Яркость')
            plt.ylabel('Количество пикселей')
            
            # Показываем калибровочную кривую с текущим результатом
            plt.subplot(2, 2, 4)
            log_conc_range = np.linspace(
                min(self.calibration_curve['log_concentrations']), 
                max(self.calibration_curve['log_concentrations']), 
                100
            ).reshape(-1, 1)
            
            intensity_pred = self.model.predict(log_conc_range)
            
            plt.plot(log_conc_range, intensity_pred, color='red', 
                     label=f'Калибровочная линия: y = {self.slope:.2f}x + {self.intercept:.2f}')
            plt.scatter(
                self.calibration_curve['log_concentrations'], 
                self.calibration_curve['intensities'], 
                color='blue', 
                label='Калибровочные точки'
            )
            plt.scatter(log_concentration, intensity, color='green', s=100,
                       label=f'Текущий образец: {concentration:.4e}')
            
            # Добавляем R² на график
            plt.text(0.05, 0.95, f'R² = {self.r2:.4f}', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
            
            plt.xlabel('log₁₀(Концентрация)')
            plt.ylabel('Интенсивность флуоресценции')
            plt.title('Расчет концентрации')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Сохраняем визуализацию, если указан путь
            if save_visualization:
                plt.savefig(save_visualization, dpi=300, bbox_inches='tight')
                print(f"Визуализация анализа сохранена в {save_visualization}")
            
            plt.show()
        
        return {
            'intensity': intensity,
            'concentration': concentration,
            'log_concentration': log_concentration
        }

    def save_calibration(self, filepath):
        """Сохраняет данные калибровки в файл."""
        if self.calibration_curve is None or self.model is None:
            raise ValueError("Сначала необходимо построить калибровочную кривую")
        
        calibration_data = {
            'slope': self.slope,
            'intercept': self.intercept,
            'r2': self.r2,
            'concentrations': self.calibration_curve['concentrations'],
            'log_concentrations': self.calibration_curve['log_concentrations'],
            'intensities': self.calibration_curve['intensities']
        }
        
        np.save(filepath, calibration_data)
        print(f"Калибровочные данные сохранены в {filepath}")
    
    def load_calibration(self, filepath):
        """Загружает данные калибровки из файла."""
        try:
            calibration_data = np.load(filepath, allow_pickle=True).item()
            
            self.slope = calibration_data['slope']
            self.intercept = calibration_data['intercept']
            self.r2 = calibration_data['r2']
            
            self.calibration_curve = {
                'concentrations': calibration_data['concentrations'],
                'log_concentrations': calibration_data['log_concentrations'],
                'intensities': calibration_data['intensities']
            }
            
            # Воссоздаем модель
            X = np.array(self.calibration_curve['log_concentrations']).reshape(-1, 1)
            y = np.array(self.calibration_curve['intensities'])
            
            model = LinearRegression()
            model.fit(X, y)
            self.model = model
            
            print(f"Калибровочные данные успешно загружены из {filepath}")
            print(f"Параметры калибровки: наклон = {self.slope:.4f}, пересечение = {self.intercept:.4f}, R² = {self.r2:.4f}")
            
            return True
        except Exception as e:
            print(f"Ошибка при загрузке калибровочных данных: {str(e)}")
            return False


# Пример использования класса
def main():
    analyzer = FluorescenceAnalyzer()
    
    # Путь к папке с калибровочными изображениями
    calibration_dir = "calibration_images"
    
    # Путь к папке с тестовыми образцами
    samples_dir = "sample_images"
    
    # Пути для сохранения результатов
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    calibration_curve_path = os.path.join(results_dir, "calibration_curve.png")
    calibration_data_path = os.path.join(results_dir, "calibration_data.npy")
    
    # Проверка существования директорий
    if not os.path.exists(calibration_dir):
        print(f"Директория {calibration_dir} не найдена. Создаём...")
        os.makedirs(calibration_dir)
        print(f"Поместите калибровочные изображения в папку {calibration_dir}. Имена файлов должны содержать концентрацию.")
        print("Примеры форматов имен файлов: conc_1.23.jpg, 1.23-conc.jpg, 1.23.jpg")
        return
    
    if not os.path.exists(samples_dir):
        print(f"Директория {samples_dir} не найдена. Создаём...")
        os.makedirs(samples_dir)
        print(f"Поместите изображения образцов в папку {samples_dir}")
        return
    
    # Проверка наличия сохраненной калибровки
    if os.path.exists(calibration_data_path):
        use_saved = input(f"Найдена сохраненная калибровка. Использовать её? (y/n): ").lower()
        if use_saved == 'y':
            if analyzer.load_calibration(calibration_data_path):
                # Визуализация загруженной калибровочной кривой
                analyzer.visualize_calibration_curve(save_path=calibration_curve_path)
                plt.show()
            else:
                print("Не удалось загрузить калибровку. Будет создана новая.")
        else:
            print("Будет создана новая калибровка.")
    
    # Если калибровка не загружена, создаем новую
    if analyzer.calibration_curve is None:
        # Получение списка калибровочных изображений
        calibration_images = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not calibration_images:
            print(f"Калибровочные изображения не найдены в {calibration_dir}")
            return
        
        # Сортировка изображений по имени
        calibration_images.sort()
        
        print(f"Найдено {len(calibration_images)} калибровочных изображений.")
        print("Концентрации будут извлечены из имен файлов.")
        
        # Выбор области интереса (ROI) на первом калибровочном изображении
        print("Выберите область интереса (ROI) на калибровочном изображении...")
        analyzer.select_roi(calibration_images[0])
        
        # Построение калибровочной кривой (концентрации извлекаются из имен файлов)
        try:
            r2_score = analyzer.build_calibration_curve(calibration_images)
            print(f"Калибровочная кривая построена. R² = {r2_score:.4f}")
            print(f"Уравнение линии: Интенсивность = {analyzer.slope:.4f} × log₁₀(Концентрация) + {analyzer.intercept:.4f}")
            
            # Визуализация калибровочной кривой
            analyzer.visualize_calibration_curve(save_path=calibration_curve_path)
            plt.show()
            
            # Сохранение калибровочных данных
            analyzer.save_calibration(calibration_data_path)
            
        except ValueError as e:
            print(f"Ошибка при построении калибровочной кривой: {str(e)}")
            print("Убедитесь, что все имена файлов содержат числовые значения концентраций.")
            return
    
    # Получение списка образцов для анализа
    sample_images = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not sample_images:
        print(f"Образцы изображений не найдены в {samples_dir}")
        return
    
    # Анализ образцов
    print(f"\nНайдено {len(sample_images)} образцов для анализа.")
    results = {}
    
    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        print(f"\nАнализ образца: {img_name}")
        
        # Путь для сохранения визуализации анализа
        sample_result_path = os.path.join(results_dir, f"analysis_{os.path.splitext(img_name)[0]}.png")
        
        result = analyzer.analyze_sample_image(img_path, save_visualization=sample_result_path)
        results[img_name] = result
        print(f"Концентрация аналита: {result['concentration']:.4e}")
        print(f"log₁₀(Концентрация): {result['log_concentration']:.4f}")
    
    # Вывод итоговых результатов
    print("\n--- Итоговые результаты ---")
    for img_name, result in results.items():
        print(f"{img_name}: концентрация = {result['concentration']:.4e}, log₁₀(конц.) = {result['log_concentration']:.4f}")
    
    # Сохранение итоговых результатов в текстовый файл
    result_file_path = os.path.join(results_dir, "analysis_results.txt")
    with open(result_file_path, 'w') as f:
        f.write(f"Параметры калибровки: наклон = {analyzer.slope:.4f}, пересечение = {analyzer.intercept:.4f}, R² = {analyzer.r2:.4f}\n\n")
        f.write("--- Результаты анализа ---\n")
        for img_name, result in results.items():
            f.write(f"{img_name}: концентрация = {result['concentration']:.4e}, log₁₀(конц.) = {result['log_concentration']:.4f}\n")
    
    print(f"\nРезультаты анализа сохранены в {result_file_path}")
    print(f"Графические результаты сохранены в папке {results_dir}")

if __name__ == "__main__":
    main()

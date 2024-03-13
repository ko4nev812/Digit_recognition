try:
    import onnxruntime
    import tkinter as tk
    import numpy as np
    from PIL import Image, ImageDraw
    # Загружаем обученную модель
    session = onnxruntime.InferenceSession('model.onnx')
    # Размеры окна и размеры матрицыs
    window_size = (280, 280)
    matrix_size = (28, 28)
    
    # Размеры ячейки
    cell_size = window_size[0] // matrix_size[0]
    
    # Размеры размытия
    blur_size = 1
    
    # Создаем окно Tkinter
    root = tk.Tk()
    root.title("Распознователь цифр")
    root.iconbitmap('icon.ico')
    # Создаем холст для рисования
    canvas = tk.Canvas(root, width=window_size[0], height=window_size[1], bg='white')
    canvas.pack()
    
    # Создаем матрицу для отслеживания степени черного цвета пикселей
    pixels_matrix = np.zeros((matrix_size[0], matrix_size[1]))
    color_squares = []
    # Переменные для отслеживания рисования
    drawing = False
    def update_color_squares(probabilities):
        colorse = []
        for i, probability in enumerate(probabilities):
            # Преобразовываем вероятность в диапазон от 0 до 1
            normalized_probability = probability
            # Вычисляем цвет в формате RGB (чем выше вероятность, тем зеленее)
            color = "#{:02x}{:02x}{:02x}".format(int(255 * (1 - normalized_probability)), int(255 * normalized_probability), 0)
            colorse.append(color)
        return colorse
    # Функция для предсказания цифры
    def predict_digit():
        global last_point
        image_array = pixels_matrix.astype(np.float32)/255.0
        # Выпрямляем ма
        # Добавляем размерность для использования в модели
        input_data = np.expand_dims(np.expand_dims(image_array, axis=-1), axis=0)
        # Предсказываем вероятности для каждой цифры с использованием модели
        probabilities = session.run(None, {'conv2d_input': input_data})[0][0]
        # Находим индекс цифры с максимальной вероятностью
        predicted_digit = np.argmax(probabilities)
        # Получаем саму максимальную вероятность
        max_probability = probabilities[predicted_digit]
        colors = update_color_squares(probabilities)
        # Отображаем вероятности
        for i, (colors_for, predic) in enumerate(zip(colors, probabilities)):
            result_labels[i].config(text=f"\u220e  {i}: {(predic*100):.4f}%",  font=("Helvetica", 10), foreground=colors_for)
        
        # Добавляем вывод самой высоковероятной цифры с большим шрифтом
        result_label.config(text=f"Скорее всего это цифра: {predicted_digit} с вероятностью: {(max_probability*100):.4f}%", font=("Helvetica", 16))
    
        # Сбрасываем последнюю точку
        last_point = None
    
    
    
    # Функция для обработки событий рисования
    def draw_handler(event):
        global drawing
        if drawing:
            x, y = event.x, event.y
            x_matrix = x // cell_size
            y_matrix = y // cell_size
            if 0 <= x_matrix < matrix_size[0] and 0 <= y_matrix < matrix_size[1]:
                for i in range(-blur_size, blur_size + 1):
                    for j in range(-blur_size, blur_size + 1):
                        new_x = min(max(x_matrix + i, 0), matrix_size[0] - 1)
                        new_y = min(max(y_matrix + j, 0), matrix_size[1] - 1)
                        if i == 0 and j == 0:
                            intensity = min(int(pixels_matrix[new_y, new_x] + 200), 255)
                        elif i == 0 or j == 0:
                            intensity = min(int(pixels_matrix[new_y, new_x] + 60), 255)
                        else:
                            intensity = min(int(pixels_matrix[new_y, new_x] + 20), 255)
                        pixels_matrix[new_y, new_x] = intensity
                        color = "#{:02x}{:02x}{:02x}".format(255 - intensity, 255 - intensity, 255 - intensity)
                        canvas.create_rectangle(new_x * cell_size, new_y * cell_size,
                                                (new_x + 1) * cell_size, (new_y + 1) * cell_size, fill=color, outline='')
    
    def clear_matrix():
        global pixels_matrix
        pixels_matrix = np.zeros((matrix_size[0], matrix_size[1]))
        canvas.delete("all")
    # Обработчики событий для рисования
    def on_press(event):
        global drawing
        drawing = True
        draw_handler(event)
        
    
    def on_release(event):
        global drawing
        drawing = False
        predict_digit()
    
    def on_motion(event):
        draw_handler(event)
    
    # Привязываем обработчики событий
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_release)
    # Добавляем кнопку очистки
    clear_button = tk.Button(root, text="Очистить", command=clear_matrix)
    clear_button.pack()
    # Метка для отображения вероятностей
    result_labels = []
    for i in range(10):
        result_label = tk.Label(root, text="")
        result_label.pack(side=tk.TOP)
        result_labels.append(result_label)
    
    predict_digit()
    result_label = tk.Label(root, text="")
    result_label.pack()
    
    # Запускаем Tkinter
    root.mainloop()
    pass
except Exception as e:
    print(f"Произошла ошибка: {e}")
    input("Нажмите Enter для выхода...")

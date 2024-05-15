import telebot
from telebot import types
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
from ultralytics import YOLO

# Токен вашего бота
bot = telebot.TeleBot('6774558652:AAF-EYc5Re_YllJ7T6h8-vSVqWPqrFpFtZE')

# Перевод названий классов на русский
class_names_russian = {
    'person': 'человек',
    'bicycle': 'велосипед',
    'car': 'машина',
    'motorcycle': 'мотоцикл',
    'airplane': 'самолет',
    'bus': 'автобус',
    'train': 'поезд',
    'truck': 'грузовик',
    'boat': 'лодка',
    'traffic light': 'светофор',
    'fire hydrant': 'пожарный гидрант',
    'stop sign': 'стоп знак',
    'parking meter': 'парковочный счетчик',
    'bench': 'скамейка',
    'bird': 'птица',
    'cat': 'кошка',
    'dog': 'собака',
    'horse': 'лошадь',
    'sheep': 'овца',
    'cow': 'корова',
    'elephant': 'слон',
    'bear': 'медведь',
    'zebra': 'зебра',
    'giraffe': 'жираф',
    'backpack': 'рюкзак',
    'umbrella': 'зонт',
    'handbag': 'сумка',
    'tie': 'галстук',
    'suitcase': 'чемодан',
    'frisbee': 'фрисби',
    'skis': 'лыжи',
    'snowboard': 'сноуборд',
    'sports ball': 'мяч',
    'kite': 'воздушный змей',
    'baseball bat': 'бита',
    'baseball glove': 'перчатка',
    'skateboard': 'скейтборд',
    'surfboard': 'доска для серфинга',
    'tennis racket': 'теннисная ракетка',
    'bottle': 'бутылка',
    'wine glass': 'бокал',
    'cup': 'чашка',
    'fork': 'вилка',
    'knife': 'нож',
    'spoon': 'ложка',
    'bowl': 'миска',
    'banana': 'банан',
    'apple': 'яблоко',
    'sandwich': 'бутерброд',
    'orange': 'апельсин',
    'broccoli': 'брокколи',
    'carrot': 'морковь',
    'hot dog': 'хот-дог',
    'pizza': 'пицца',
    'donut': 'пончик',
    'cake': 'торт',
    'chair': 'стул',
    'couch': 'диван',
    'potted plant': 'растение в горшке',
    'bed': 'кровать',
    'dining table': 'обеденный стол',
    'toilet': 'унитаз',
    'tv': 'телевизор',
    'laptop': 'ноутбук',
    'mouse': 'мышь',
    'remote': 'пульт',
    'keyboard': 'клавиатура',
    'cell phone': 'мобильный телефон',
    'microwave': 'микроволновка',
    'oven': 'духовка',
    'toaster': 'тостер',
    'sink': 'раковина',
    'refrigerator': 'холодильник',
    'book': 'книга',
    'clock': 'часы',
    'vase': 'ваза',
    'scissors': 'ножницы',
    'teddy bear': 'плюшевый медведь',
    'hair drier': 'фен',
    'toothbrush': 'зубная щетка'
}


def download_and_open_image(file_id):
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_stream = io.BytesIO(downloaded_file)
    image = Image.open(image_stream).convert('RGB')
    return np.array(image)


def detect_objects(image_array):
    model = YOLO("yolov8x.pt")  # Путь к вашей модели
    results = model(image_array)  # Предсказание

    object_counts = {}  # Счетчик объектов
    font = ImageFont.truetype("arial.ttf", 20)  # Загрузка шрифта Arial
    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            label_ru = class_names_russian.get(label, label)
            if label_ru in object_counts:
                object_counts[label_ru] += 1
            else:
                object_counts[label_ru] = 1
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, y1 - 20), label_ru, font=font, fill=(255, 0, 0))

    caption = '\n'.join([f"{label}: {count}" for label, count in object_counts.items()])
    return pil_image, caption


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("ℹ️ Информация о боте")
    btn2 = types.KeyboardButton("📷 Отправить фото для анализа")
    markup.add(btn1, btn2)

    welcome_text = (
        "👋 Привет! Я бот, который поможет тебе распознать объекты на изображениях с помощью модели YOLO. "
        "Отправь мне фотографию, и я верну тебе изображение с аннотациями обнаруженных объектов. 📸\n\n"
        "Также я покажу количество каждого обнаруженного объекта. 🔍\n\n"
        "Выберите одну из доступных опций ниже, чтобы начать. 👇"
    )
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)


@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text == "ℹ️ Информация о боте":
        info_text = (
            "ℹ️ Я использую модель YOLOV8 для распознавания объектов на изображениях. "
            "Отправь мне изображение, и я верну его с аннотациями. "
            "Я также покажу количество каждого обнаруженного объекта. 📊"
        )
        bot.send_message(message.chat.id, info_text)
    elif message.text == "📷 Отправить фото для анализа":
        bot.send_message(message.chat.id, "📷 Пожалуйста, отправьте фото для анализа.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    file_id = message.photo[-1].file_id

    # Отправка сообщения "Идет поиск..."
    processing_message = bot.reply_to(message, "⏳ Идет поиск...")

    image_array = download_and_open_image(file_id)
    image_with_boxes, caption = detect_objects(image_array)

    buffer = io.BytesIO()
    image_with_boxes.save(buffer, format='JPEG')
    buffer.seek(0)

    # Удаление сообщения "Идет поиск..."
    bot.delete_message(chat_id, processing_message.message_id)

    # Ответ на исходное сообщение с фотографией
    bot.send_photo(chat_id, buffer, caption=caption if caption else "Объекты не обнаружены.",
                   reply_to_message_id=message.message_id)


if __name__ == '__main__':
    bot.polling(none_stop=True)

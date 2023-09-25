# Базовый набор функций для создания запроса к Stable Diffusion

## Формат изображений

Предполагается, что используются изображения в формате `.png`, с названием в формате `id_{prompt}_dilparam.png`, где `id` - какой-либо идентификатор, например, id пользователя, `prompt` - промпт для генерации изображения, `dilparam` - числовой параметр для функции `dilation` из `process.py`. Пример названия файла - `12931230_women_polka_dot_t-shirt_5.png`

## process.py

Основан на **[Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)**
**[Arxiv](https://arxiv.org/pdf/1910.09777v1.pdf)**

Программа для получения масок одежды и верхних конечностей для изображений с людьми.

Для работы необходимо:
1) Склонировать репо https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

```shell
git https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git
```

2) Поместить `process.py` в корневую папку этого репозитория

```none
Self-Correction-Human-Parsing
├── ...
├── process.py
├── train.py
├── ...
```

3) Скачать **[чекпоинт](https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP/view?usp=sharing)** нейронной сети 

4) Запустить `process.py`, указав необходимые параметры:

    - --model-restore - путь до чекпоинта
    - --input-dir - папка со входными изображениями

```shell
python process.py --model-restore --input-dir
```

## api_req_wo_async.py

Программа для создания запросов и получения изображений от [Stable Diffusion API](http://platform.stability.ai/).

Для работы необходимо: сгенерировать ключ к API на платформе [Dreamstudio](http://dreamstudio.ai/) и поместить его в файл `config` в той же директории, что и файл `api_req_wo_async.py`, а затем запустить программу, указав необходимые параметры:

   - --output-dir - папка для хранения изображений, полученных от Stable Diffusion API
   - --search - папка для хранения изображений для поискового запроса по фотке

```shell
python api_req_wo_async.py --output-dir --search
```

### Оба файла должны находиться в одной папке!

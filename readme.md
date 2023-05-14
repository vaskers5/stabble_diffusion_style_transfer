### About

Основная идея была реализовать следующий пайплайн:

1) Генерация описаний content + style с помощью BLIP
2) Удаления фона с картинок
3) Style Transfer с помощью kadinsky на картинках с удаленным фоном
4) Генерация фона для 3), опять же с помощью Kadinsky

Также результаты будут представлены обычной модели kadinsky, без использования пайплайна выше

### Результаты работы:


### Используя пайплайн:

![Гигачад + Шрек](results/shrek_preproc.jpg)
![Боромир + вазовски](results/with_preproc.jpg)


### Без использования  пайплайна, обычный подбор гиперпараметров:

![Боромир + вазовски](results/results_waz_withour_post_proc.jpg)
![Гигачад + Шрек](results/results_giga.jpg)


### Installation:

### PreRequirements:
pytorch >=1.6.0 builds with cuda support

### Installation:

```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
pip install transformers -U
```

### Testing locally

```bash
python run.py images/boromir.jpeg images/vazovski.jpeg
```
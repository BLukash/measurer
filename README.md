# Measurer
Uses reference object - ruler and homography (not there yet) to get pixel to unit (cm/in/etc.) ratio and measure real distance between any two points on that image.

Ruler scale detection is based on my older research, details and some alternative approaches can be found there:
- [Method of automatic identification of ruler scales on images to solve the problem of estimating the geometric dimensions of certain objects](https://ieeexplore.ieee.org/abstract/document/9916468)
- [ScaleSense: удосконалений метод автоматичного визначення масштабу лінійки на зображеннях для вирішення завдання оцінювання геометричних розмірів певних об'єктів](https://nv.nltu.edu.ua/index.php/journal/article/view/2513)

## Usage

Requires [uv](https://docs.astral.sh/uv/). Run from the repo root:

```bash
uv sync
uv run python -m src.detect --input path/to/ruler.jpg --output ruler_annotated.jpg
```

Prints the pixels per cm/in and saves the image with detected numbers and the measured segment drawn on it. The output folder must already exist.

```bash
Seems that digit detection either works worse on ONNX export
or I have too optimistic memories on its performance before.
In either way I will update the model used in the near future
```

## Acknowledgments and License

This project utilizes a fine-tuned version of the YOLOv7 architecture based on the original work by [WongKinYiu/yolov7](https://github.com/wongkinyiu/yolov7).

In accordance with the upstream repository, this demo software and the accompanying model weights are licensed under the **GNU General Public License v3.0 (GPL-3.0)**. You can find the full license terms in the `LICENSE` file.

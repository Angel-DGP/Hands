# 🖐️ Detección y Mapeo de Dedos en Tiempo Real con OpenCV y MediaPipe

Este proyecto utiliza la cámara del dispositivo para detectar y rastrear las manos en tiempo real, resaltando puntos clave como la punta del pulgar e índice meñique, mediante la librería [MediaPipe](https://mediapipe.dev/) y [OpenCV](https://opencv.org/).

## 📸 ¿Qué hace?

- Conecta automáticamente a tu cámara web.
- Detecta hasta dos manos simultáneamente.
- Muestra en tiempo real la posición de los dedos.
- Resalta puntos específicos (ID 4 y 20).

## 🔧 Requisitos

- Python 3.10+
- Pip

### 📦 Instalación rápida (con entorno virtual recomendado)

```bash
# Clona el repositorio
git clone https://github.com/Angel-DGP/Hands.git
cd Hands

# Crea entorno virtual
python -m venv venv
# Actívalo
# En Windows
venv\Scripts\activate
# En Linux/Mac
source venv/bin/activate

# Instala dependencias
pip install -r requirements.txt

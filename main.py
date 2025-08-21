import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def dedos_extendidos(puntos, mano_label):
    dedos_ids = {
        "pulgar": [4, 3, 2],  
        "indice": [8, 6],
        "medio": [12, 10],
        "anular": [16, 14],
        "menique": [20, 18]
    }

    resultado = {}

    if mano_label == "Right":
        resultado["pulgar"] = puntos[4][0] > puntos[3][0]
    else:
        resultado["pulgar"] = puntos[4][0] < puntos[3][0]

    for dedo in ["indice", "medio", "anular", "menique"]:
        tip = dedos_ids[dedo][0]
        pip = dedos_ids[dedo][1]
        resultado[dedo] = puntos[tip][1] < puntos[pip][1]

    return resultado


dispositivoCaptura = cv2.VideoCapture(1)
historial = []  

mpManos = mp.solutions.hands
manos = mpManos.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.8,
                      min_tracking_confidence=0.8) 
mpDibujar = mp.solutions.drawing_utils


font = ImageFont.truetype("arial.ttf", 32)

ultimo_gesto = ""

while True:
    success, img = dispositivoCaptura.read()
    if not success:
        dispositivoCaptura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = manos.process(imgRGB)

    mensaje = ""

    if resultado.multi_hand_landmarks:
        for handLms, handType in zip(resultado.multi_hand_landmarks, resultado.multi_handedness):
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)

            alto, ancho, _ = img.shape
            puntos = {}
            for id, lm in enumerate(handLms.landmark):
                puntos[id] = (int(lm.x * ancho), int(lm.y * alto))

            mano_label = handType.classification[0].label
            dedos_estado = dedos_extendidos(puntos, mano_label)

            if dedos_estado["menique"] and dedos_estado["pulgar"] and not any(
                dedos_estado[d] for d in ["indice", "medio", "anular"]
            ):
                mensaje = "No"
            elif not dedos_estado["pulgar"] and not any(
                dedos_estado[d] for d in ["indice", "medio", "anular", "menique"]
            ):
                mensaje = "Adios"
            elif all(dedos_estado.values()):
                mensaje = "Hola"
            elif dedos_estado["indice"] and not dedos_estado["pulgar"] and not any(
                dedos_estado[d] for d in ["medio", "anular", "menique"]
            ):
                mensaje = "Si"

    if mensaje and mensaje != ultimo_gesto:
        historial.append(mensaje)
        ultimo_gesto = mensaje

    alto, ancho, _ = img.shape
    barra_altura = 70
    barra = np.ones((barra_altura, ancho, 3), dtype=np.uint8) * 255 

    barra_pil = Image.fromarray(barra)
    draw = ImageDraw.Draw(barra_pil)

    x = 20
    for gesto in historial[-4:]:  
        draw.text((x, 15), gesto, font=font, fill=(0, 0, 0))
        x += 150  

    barra = np.array(barra_pil)


    img_final = np.vstack((img, barra))

    cv2.imshow("Detector de Gestos", img_final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

dispositivoCaptura.release()
cv2.destroyAllWindows()

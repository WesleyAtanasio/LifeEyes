import cv2
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import ttk
import random
import sys

# Carrega os classificadores Haar globalmente para reutilização
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variável global para armazenar configurações e dados de calibração
config = {
    "eye_color": None,
    "selected_eye": None,  # 'left' ou 'right'
    "calibration": None     # dicionário com limites do movimento do pupilo
}

def detect_pupil(frame, selected_eye):
    """
    Dado um frame (imagem capturada da webcam), detecta o rosto, os olhos e, a partir do olho
    selecionado, retorna o centro (cx, cy) do pupilo. Se não conseguir detectar, retorna None.
    """
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None

    # Usaremos o primeiro rosto detectado
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(face_roi)
    if len(eyes) == 0:
        return None

    # Seleciona o olho com base na configuração:
    # Para "left", escolhe o olho com menor coordenada x; para "right", o de maior.
    if selected_eye == "left":
        selected = min(eyes, key=lambda e: e[0])
    else:
        selected = max(eyes, key=lambda e: e[0])
    ex, ey, ew, eh = selected

    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
    # Ajusta o threshold conforme a cor dos olhos (valores podem ser ajustados)
    if config["eye_color"].lower() in ["azul", "verde"]:
        thresh_val = 40
    elif config["eye_color"].lower() == "castanho":
        thresh_val = 30
    else:
        thresh_val = 35

    eye_roi_blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)
    _, thresh = cv2.threshold(eye_roi_blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None

def calibrate(num_points=10):
    """
    Realiza a calibração exibindo uma tela fullscreen branca com um ponto preto em
    posições aleatórias. Para cada ponto, o usuário deve olhar para ele e pressionar a
    tecla 'c'. Ao pressionar, a webcam captura um frame e detecta a posição do pupilo.
    Retorna uma lista de dicionários, cada um com a posição do ponto na tela e a posição
    do pupilo detectada.
    """
    # Obtém resolução da tela
    screen_w, screen_h = pyautogui.size()
    calibration_data = []

    # Cria uma imagem branca com o tamanho da tela
    white_img = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

    # Abre a webcam para capturar os frames durante a calibração
    cap_cal = cv2.VideoCapture(0)
    if not cap_cal.isOpened():
        print("Erro ao acessar a webcam para calibração.")
        return calibration_data

    cv2.namedWindow("Calibração", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibração", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i in range(num_points):
        # Gera posição aleatória com margem para não ficar nas bordas
        margin = 50
        pt_x = random.randint(margin, screen_w - margin)
        pt_y = random.randint(margin, screen_h - margin)

        img = white_img.copy()
        # Desenha o ponto de calibração (círculo preto)
        cv2.circle(img, (pt_x, pt_y), 20, (0, 0, 0), -1)
        # Exibe contador no topo
        text = f"Ponto {i+1} de {num_points}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(img, text, ((screen_w - tw)//2, th+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow("Calibração", img)

        # Aguarda o usuário pressionar 'c' para capturar a posição do pupilo
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                ret, frame = cap_cal.read()
                if not ret:
                    print("Falha ao capturar frame da webcam.")
                    continue
                # Detecta a posição do pupilo usando a função auxiliar
                pupil_pos = detect_pupil(frame, config["selected_eye"])
                if pupil_pos is None:
                    print("Não foi possível detectar o pupilo. Tente novamente.")
                    continue
                # Armazena: posição do ponto na tela e a posição do pupilo detectada
                calibration_data.append({
                    "screen_x": pt_x,
                    "screen_y": pt_y,
                    "pupil_x": pupil_pos[0],
                    "pupil_y": pupil_pos[1]
                })
                break
            elif key == ord('q'):
                cap_cal.release()
                cv2.destroyAllWindows()
                return calibration_data

    cap_cal.release()
    cv2.destroyAllWindows()
    return calibration_data

def run_calibration():
    """
    Atualiza as configurações com os valores selecionados na interface
    e executa a calibração.
    """
    # Atualiza as configurações a partir dos campos da interface
    config["eye_color"] = eye_color_var.get()
    config["selected_eye"] = eye_choice.get()
    
    print("Iniciando calibração...")
    calib_points = calibrate(num_points=10)
    if calib_points:
        # Extrai os valores capturados do pupilo para cada ponto
        pupil_x_vals = [pt['pupil_x'] for pt in calib_points]
        pupil_y_vals = [pt['pupil_y'] for pt in calib_points]
        config["calibration"] = {
            "pupil_x_min": min(pupil_x_vals),
            "pupil_x_max": max(pupil_x_vals),
            "pupil_y_min": min(pupil_y_vals),
            "pupil_y_max": max(pupil_y_vals)
        }
        print("Calibração concluída:", config["calibration"])
    else:
        print("Calibração não realizada ou cancelada.")

def iniciar_programa():
    """
    Salva as configurações da janela de configuração e fecha-a.
    """
    config["eye_color"] = eye_color_var.get()
    config["selected_eye"] = eye_choice.get()
    root.destroy()

# Janela de Configuração (Tkinter)
root = tk.Tk()
root.title("Configuração - Controle do Mouse com os Olhos")
root.geometry("300x320")
root.resizable(False, False)

tk.Label(root, text="Selecione a cor dos olhos:", font=("Arial", 10)).pack(pady=5)
eye_color_var = tk.StringVar(value="Castanho")
eye_color_options = ["Castanho", "Azul", "Verde", "Outro"]
eye_color_menu = ttk.Combobox(root, textvariable=eye_color_var, values=eye_color_options, state="readonly")
eye_color_menu.pack(pady=5)

tk.Label(root, text="Escolha com qual olho operar:", font=("Arial", 10)).pack(pady=5)
eye_choice = tk.StringVar(value="left")
frame_radio = tk.Frame(root)
frame_radio.pack(pady=5)
tk.Radiobutton(frame_radio, text="Esquerdo", variable=eye_choice, value="left").pack(side="left", padx=10)
tk.Radiobutton(frame_radio, text="Direito", variable=eye_choice, value="right").pack(side="left", padx=10)

tk.Button(root, text="Calibrar", command=run_calibration, bg="#2196F3", fg="white").pack(pady=10)
tk.Button(root, text="Iniciar", command=iniciar_programa, bg="#4CAF50", fg="white").pack(pady=10)

root.mainloop()

print("Configuração escolhida:")
print("Cor dos olhos:", config["eye_color"])
print("Olho utilizado:", config["selected_eye"])
print("Calibração:", config["calibration"])

# Inicializa a captura de vídeo para o controle do mouse
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    sys.exit()

# Obtém o tamanho da tela
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi)
        if len(eyes) == 0:
            continue

        if config["selected_eye"] == "left":
            selected = min(eyes, key=lambda e: e[0])
        else:
            selected = max(eyes, key=lambda e: e[0])
        ex, ey, ew, eh = selected
        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

        if config["eye_color"].lower() in ["azul", "verde"]:
            thresh_val = 40
        elif config["eye_color"].lower() == "castanho":
            thresh_val = 30
        else:
            thresh_val = 35

        eye_roi_blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)
        _, thresh = cv2.threshold(eye_roi_blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (x+ex+cx, y+ey+cy), 3, (0, 0, 255), -1)

                # Se a calibração foi realizada, mapeia o movimento do pupilo para a tela
                if config["calibration"]:
                    pminx = config["calibration"]["pupil_x_min"]
                    pmaxx = config["calibration"]["pupil_x_max"]
                    pminy = config["calibration"]["pupil_y_min"]
                    pmaxy = config["calibration"]["pupil_y_max"]

                    # Evita divisão por zero
                    if pmaxx - pminx == 0 or pmaxy - pminy == 0:
                        continue

                    rel_x = (cx - pminx) / (pmaxx - pminx)
                    rel_y = (cy - pminy) / (pmaxy - pminy)

                    target_x = int(screen_w * rel_x)
                    target_y = int(screen_h * rel_y)
                    pyautogui.moveTo(target_x, target_y)
        break

    cv2.imshow("Controle do Mouse com os Olhos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import face_recognition
import sys
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from threading import Thread

# Diretório para armazenar as imagens e codificações
KNOWN_FACES_DIR = "known_faces"

# Verificar se o diretório existe, se não, criá-lo
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Caminho para os arquivos de modelo
MODEL_DIR = "C:\\Users\\willi\\PycharmProjects\\ProjetoFacial3"
proto_txt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

# Certifique-se de que esses arquivos estão no caminho correto
if not os.path.isfile(proto_txt_path) or not os.path.isfile(model_path):
    print("Erro: Não foi possível encontrar os arquivos de modelo necessários.")
    sys.exit(1)

net = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

# Variável global para controle da execução do reconhecimento facial
recognition_running = False

# Função para carregar codificações conhecidas
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        if os.path.isfile(filepath):
            # Carregar a imagem
            image = face_recognition.load_image_file(filepath)
            # Codificar o rosto na imagem
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename.split(".")[0])

    return known_face_encodings, known_face_names

# Função para reconhecer rostos e detectar objetos
def recognize_faces_and_objects():
    global recognition_running
    recognition_running = True

    # Carregar as codificações conhecidas
    known_face_encodings, known_face_names = load_known_faces()

    # Inicializar a webcam
    video_capture = cv2.VideoCapture(0)

    # Lista de classes do modelo MobileNet SSD
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

    while recognition_running:
        # Capturar um único frame da webcam
        ret, frame = video_capture.read()

        # Reduzir o tamanho do frame para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Encontrar todas as faces e suas codificações no frame atual
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Verificar se a face encontrada corresponde a uma das faces conhecidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        # Detectar objetos
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Exibir apenas o nome do objeto detectado na tela
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Exibir os resultados de reconhecimento facial
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Colocar o nome abaixo do rosto
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Exibir a imagem resultante
        cv2.imshow('Video', frame)

        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognition_running = False
            break

    # Liberar a webcam e fechar janelas
    video_capture.release()
    cv2.destroyAllWindows()

# Função para parar o reconhecimento facial
def stop_recognition():
    global recognition_running
    recognition_running = False

# Função para cadastrar um novo rosto
def register_face():
    # Pedir o nome da pessoa
    name = simpledialog.askstring("Cadastro", "Digite o nome da pessoa:")

    if name:
        # Inicializar a webcam
        video_capture = cv2.VideoCapture(0)

        while True:
            # Capturar um único frame da webcam
            ret, frame = video_capture.read()

            # Exibir o frame
            cv2.imshow('Cadastro - Pressione C para Capturar', frame)

            # Pressionar 'c' para capturar a imagem
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Salvar a imagem capturada
                image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                cv2.imwrite(image_path, frame)
                break

        # Liberar a webcam e fechar janelas
        video_capture.release()
        cv2.destroyAllWindows()

        # Informar que o cadastro foi bem-sucedido
        messagebox.showinfo("Cadastro", f"{name} cadastrado com sucesso!")

# Função principal para iniciar a interface gráfica
def main():
    root = tk.Tk()
    root.title("Reconhecimento Facial e Detecção de Objetos")

    tk.Button(root, text="Iniciar Reconhecimento Facial e Detecção de Objetos", command=lambda: Thread(target=recognize_faces_and_objects).start()).pack(pady=10)
    tk.Button(root, text="Parar Reconhecimento Facial e Detecção de Objetos", command=stop_recognition).pack(pady=10)
    tk.Button(root, text="Cadastrar Novo Rosto", command=register_face).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

import cv2
import face_recognition
import sys

# Caminho da imagem de referência
image_path = "meu_rosto.jpg"

# Carregar uma imagem de referência
my_image = cv2.imread(image_path)

# Verifique se a imagem foi carregada corretamente
if my_image is None:
    print(f"Erro ao carregar a imagem {image_path}. Verifique se o caminho está correto.")
    sys.exit(1)

# Converta a imagem para RGB
rgb_my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)

# Codificar o rosto na imagem
face_encodings = face_recognition.face_encodings(rgb_my_image)

# Verifique se alguma face foi detectada na imagem
if len(face_encodings) == 0:
    print("Nenhuma face foi detectada na imagem de referência. Verifique se a imagem contém uma face visível.")
    sys.exit(1)

my_face_encoding = face_encodings[0]

known_face_encodings = [my_face_encoding]
known_face_names = ["Meu Nome"]

# Inicializar a webcam
video_capture = cv2.VideoCapture(0)

while True:
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

    # Exibir os resultados
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
        break

# Liberar a webcam e fechar janelas
video_capture.release()
cv2.destroyAllWindows()
import cv2
import face_recognition

# Caminhos das imagens de referência
image_paths = ["rosto1.jpg", "rosto2.jpg", "rosto3.jpg"]

# Carregar e codificar as imagens de referência
face_encodings = []
for path in image_paths:
    # Carregar a imagem
    image = cv2.imread(path)

    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem {path}. Verifique se o caminho está correto.")
        continue

    # Converta a imagem para RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Codificar o rosto na imagem
    encodings = face_recognition.face_encodings(rgb_image)

    # Verificar se alguma face foi detectada na imagem
    if len(encodings) == 0:
        print(f"Nenhuma face foi detectada na imagem {path}. Verifique se a imagem contém uma face visível.")
        continue

    # Adicionar a codificação do rosto à lista
    face_encodings.append(encodings[0])

# Verificar se temos três codificações de rosto
if len(face_encodings) != 3:
    print("Não foi possível codificar todas as três imagens. Certifique-se de que todas as imagens contenham uma face visível.")
    exit(1)

# Comparar as codificações faciais
results = []
for i in range(len(face_encodings)):
    for j in range(i + 1, len(face_encodings)):
        match = face_recognition.compare_faces([face_encodings[i]], face_encodings[j])[0]
        results.append((i + 1, j + 1, match))

# Exibir os resultados da comparação
for i, j, match in results:
    result = "Match" if match else "No Match"
    print(f"Comparação entre rosto {i} e rosto {j}: {result}")

# Exibir as imagens
for i, path in enumerate(image_paths):
    image = cv2.imread(path)
    cv2.imshow(f'Rosto {i + 1}', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

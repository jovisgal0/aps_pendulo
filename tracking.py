import cv2 # type: ignore
import pandas as pd # type: ignore
import numpy as np

video = cv2.VideoCapture("./pendulo.mp4")

# Criar um vetor para armazenar os dados obtidos
dados = []

# Variável para guardar o tempo
t = 0

# Guarda o número de frames por segundo do vídeo
fps = video.get(cv2.CAP_PROP_FPS)

# Função para remover o fundo no vídeo, deixar apenas objetos que se movem
detectorDeObjetos = cv2.createBackgroundSubtractorKNN(history=3000, dist2Threshold=900, detectShadows=False)

# Enquanto o vídeo roda ou até apertar esc para fechar
while True:
    working, frame = video.read()
    # Caso o vídeo termine, fecha a janela
    if not working:
        break

    # Ajusta o tamanho da janela para um valor adequado
    frame = cv2.resize(frame, (540, 960))

    # Aplica o detector de objetos em uma máscara
    mask = detectorDeObjetos.apply(frame)

    # Desenhar um contorno dos nossos objetos do vídeo
    contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 200:
            # Posição x e y, largura e altura do contorno
            x, y, w, h = cv2.boundingRect(cnt)

            # Contorno retangular
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

            # Coloca no final do vetor de dados uma coluna de tempo e a outra da posição do centro do retângulo
            dados.append({"t": t, "x": (x + w/2)})
            break

    # Abre uma janela para o vídeo
    cv2.imshow("Janela", frame)

    # Atualiza o valor do tempo
    t += 1.0/fps

    # Mostrar quanto tempo já passou no terminal
    print(f"Time: {t}")

    # Se apertar esc, também fecha a janela
    tecla = cv2.waitKey(30)
    if tecla == 27:
        break

# Fechar o vídeo e todas as janelas que foram abertas
video.release()
cv2.destroyAllWindows()

# Transforma o vetor dados em uma tabela usando a biblioteca pandas
tabela = pd.DataFrame(dados)

# Salvar essa tabela em um arquivo
tabela.to_csv("posicoes.csv", index=False)

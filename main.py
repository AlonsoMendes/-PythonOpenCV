#Importando as bibliotecas
import cv2
import matplotlib.pyplot as plt

#Definindo variaveis para o acesso dos arquivos ** não esqueça de colocar o caminho na sua maquina
#Estrutura da rede neural
arquivo_proto = "X:/PYTHON/MapadeCalor/pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
#Pesos da rede neural
arquivo_pesos = "X:/PYTHON/MapadeCalor/pose/body/mpi/pose_iter_160000.caffemodel"

#Carregando as imagens como um array NumPy
img = cv2.imread("X:/PYTHON/MapadeCalor/imagens/body/multiple/multiple_1.jpeg")
print(img.shape) #Verificando a quantidade de pixels da imagem
cv2.imshow('preview', img)
cv2.waitKey(0)

#Criação das variaveis para armazenamento da largura e altura da imagem
img_largura = img.shape[1]
img_altura = img.shape[0]

#Lendo o modelo de criação e peso da rede neural
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

# Definir novas dimensões da imagem de entrada.
altura_entrada = 368
largura_entrada = int((altura_entrada / img_altura) * img_largura)#Formula para deixar a imagem proporcional

#Converter a imagem do formato openCV para o formato blob Caffe
blob_entrada = cv2.dnn.blobFromImage(image=img, scalefactor=1.0 / 255, size=(largura_entrada, altura_entrada),
                                     mean=(0, 0, 0), swapRB=False, crop=False)

#Saída
modelo.setInput(blob_entrada)# imagem convertida
saida = modelo.forward()# retornar as previsão da rede neural

print(saida.shape)# mostra os valores sendo o primeiro o ID da imagem
print(saida[0].shape)
print(saida[0][1]) # gerando o mapa de confiança

#MAPA DE CONFIANÇA
ponto = 15
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca, (img_largura, img_altura))

plt.figure(figsize=[14, 10])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(mapa_confianca, alpha=0.6)
plt.axis("off") #limpar os eixos
plt.show()#Para mostrar o gráfico na tela

#MAPA DE AFINIDADE
# Ápos o ponto 16 começa o mapa de afinidade que termina no ponto 43.


















import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#carregar os dados CSV
data = pd.read_csv(r"dataset.csv")

'''---------------------------TREINAMENTO----------------------------------------------------------'''
#converter valores categóricos em valores numéricos (vai ser usado no KNN)
convertVar = LabelEncoder()

#aplica LabelEncoder em todas as colunas categóricas de X
for column in data.columns[:-1]:  # Exclui a última coluna 'Disorder'
    data[column] = convertVar.fit_transform(data[column])

#codificar a variável target (Disorder)
data['Disorder'] = convertVar.fit_transform(data['Disorder'])

#dividir os dados em features (X) e rótulos (y)
X = data.drop(columns=['Disorder'])  # Todas as colunas, exceto a última (entrada)
Y = data['Disorder']  # A última coluna (saída)

#divide os dados em conjunto de treino e teste
xTreino, xTeste, yTreino, yTeste = train_test_split(X, Y, test_size=0.2, random_state=42)

#inicializar o modelo
modelo = KNeighborsClassifier()

#treinar o modelo
modelo.fit(xTreino, yTreino)

#avaliar o desempenho do modelo
precisao = modelo.score(xTeste, yTeste)
print("Precisão do modelo(KNN): {}".format(precisao))

'''-------------------------------CHATBOT-----------------------------------'''
#perguntas baseadas nas colunas do dataset
conversas =[
    "Voce está se sentindo nervoso?",
    "Voce está tendo ataques de panico?",
    "Sua respiração está rápida?",
    "Voce está sundo?",
    "Está tendo problemas para se concentrar?",
    "Está tendo dificuldades para dormir",
    "Está tendo problemas no trabalho?",
    "Você se sente sem esperança?",
    "Você esta com raiva?",
    "Você tende a exagerar?",
    "Você percebe mudanças nos seus habitos alimentares?",
    "Você tem pensamentos suicidas?",
    "Você se sente cansado?",
    "Você tem um amigo proximo?",
    "Você tem vicio em redes sociais?",
    "Você ganhou peso recentemente?",
    "Você valoriza muito as posses materiais?",
    "Você se considera introvertido?",
    "Lembranças estressantes estão surgindo?",
    "Você tem pesadelos?",
    "Você evita pessoas ou atividades?",
    "Você está se sentindo negativo?",
    "Está com problemas de concentraçao?",
    "Você tende a se culpar por coisas?"
]

#coletar respostas do usuário e fazer a predição
def coletarRespostas():
    respostas = []

    for pergunta in conversas:
        resposta = input("{} (sim/não): ".format(pergunta))
        if resposta == 'sim':
            respostas.append(1)
        else:
            respostas.append(0)

    #fazer a predição com base nas respostas do usuário
    predicao = modelo.predict([respostas])

    #converter a predição de volta para o transtorno correspondente
    transtornoPredito = convertVar.inverse_transform(predicao)[0]

    #diagnosticos
    if transtornoPredito == 'Normal':
        print("Com base nas suas respostas, você pode ta suave (normal)")

    if transtornoPredito == 'Stress':
        print("Com base nas informações dadas voce pode estar com estresse, lembresse de procurar um profissional especializado para ter certeza")

    if transtornoPredito == 'Loneliness':
        print("Com base nas informacoes voce pode estar se sentindo sozinho")

    if transtornoPredito == 'Depression':
        print("Com base nas informaçoes que voce forneceu, voce indica sintomas de depressão, procure um profissional para ter certeza que está tudo bem ")

    if transtornoPredito == 'Anxiety':
        print("Com base nas informações que voce forneceu coce parece estar com ansiedade, procure um profissional para ")

'''-------------------------------INTERACAO USUARIO MAQUINA-----------------------------------'''
#loop de interação com o usuário
while True:
    pergunta = input("Você: ")
    if pergunta.lower() == 'sair':
        break
    elif pergunta.lower() == 'avaliar':
        coletarRespostas()
    else:
        print("Por favor, digite 'avaliar' para começar.")

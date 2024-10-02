import unittest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch


# Função auxiliar para carregar e preparar os dados
def preparar_dados(caminho_csv):
    data = pd.read_csv(caminho_csv)
    convertVar = LabelEncoder()

    for column in data.columns[:-1]:
        data[column] = convertVar.fit_transform(data[column])
    data['Disorder'] = convertVar.fit_transform(data['Disorder'])

    X = data.drop(columns=['Disorder'])
    Y = data['Disorder']
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def inicializarModelo():
    return KNeighborsClassifier()

class TestChatbotModel(unittest.TestCase):

    # 1.teste de carregamento de dataset
    def testCarregamentoDataset(self):
        try:
            data = pd.read_csv("dataset.csv")
        except FileNotFoundError:
            self.fail("Arquivo dataset.csv não foi encontrado")

    # 2-teste de ausência do dataset
    def testAusenciaDataset(self):
        with self.assertRaises(FileNotFoundError):
            pd.read_csv("arquivo_inexistente.csv")

    # 3-teste a conversão das variáveis
    def testConversaoVar(self):
        data = pd.read_csv("dataset.csv")
        convertVar = LabelEncoder()

        for column in data.columns[:-1]:
            data[column] = convertVar.fit_transform(data[column])
            self.assertIn(data[column].dtype, ['int32', 'int64'])

    # 4-testa a divisão dos dados
    def testDivDados(self):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        self.assertGreater(len(xTreino), 0)
        self.assertGreater(len(xTeste), 0)
        self.assertGreater(len(yTreino), 0)
        self.assertGreater(len(yTeste), 0)

    # 5.teste de inicialização do modelo KNN
    def testInicializacaoKnn(self):
        modelo = inicializarModelo()
        self.assertIsInstance(modelo, KNeighborsClassifier)

    # 6-teste de treinamento do modelo
    def testTreinamento(self):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        modelo = inicializarModelo()
        modelo.fit(xTreino, yTreino)
        self.assertIsNotNone(modelo)  # ve se o modelo foi treinado sem erros

    # 7-teste de precisao do modelo
    def testPreciso(self):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        modelo = inicializarModelo()
        modelo.fit(xTreino, yTreino)
        precisao = modelo.score(xTeste, yTeste)
        self.assertTrue(0 <= precisao <= 1)  # deve ser um valor entre 0 e 1

    # 8-teste para 'Stress' so com sim
    @patch('builtins.input', side_effect=['sim'] * 24)
    def testStress(self, mock_input):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        modelo = inicializarModelo()
        modelo.fit(xTreino, yTreino)
        respostas = [1] * 24  # Simulando respostas "sim" para todas as perguntas
        predicao = modelo.predict([respostas])
        self.assertIsNotNone(predicao)  # Verifica se a predição ocorre sem erro

    # 9-teste de predição respostas inconsistentes simulando entre s e n as respostas
    @patch('builtins.input', side_effect=['sim', 'não'] * 12)
    def test_predicao_inconsistente(self, mock_input):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        modelo = inicializarModelo()
        modelo.fit(xTreino, yTreino)
        respostas = [1, 0] * 12  # Simulan respostas alternadas
        predicao = modelo.predict([respostas])
        self.assertIsNotNone(predicao)  #verifica se a prediçao ocorre sem erro

    # 10-teste de novos dados não vistos durante o treinamento
    def testNovosDados(self):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        modelo = inicializarModelo()
        modelo.fit(xTreino, yTreino)

        #simula novas de respostas do usuario
        novas_respostas = [1 if i % 2 == 0 else 0 for i in range(24)]  #respostas alternadas entre 1 e 0
        predicao = modelo.predict([novas_respostas])

        # Verifica se a predição é válida (não deve ser nula)
        self.assertIsNotNone(predicao)
        self.assertTrue(predicao[0] in yTreino.unique())  #verifica se a predicao esta entre os possíveis rótulos

if __name__ == '__main__':
    unittest.main()

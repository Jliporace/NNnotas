import numpy as np
from numpy.linalg import inv


class NeuralNet(object):
    def __init__(self, topologia):
        np.random.seed(0)
        self.topologia = topologia
        self.parametros = topologia[0]
        self.saida = topologia[-1]
        self.hidden = topologia[1:-1]
        self.thetas = []
        for i in range(len(self.hidden)+1):
            self.thetas.append(self.Constroimatriz(i))

    def Constroimatriz(self,n):
        theta = (np.random.rand(self.topologia[n], self.topologia[n+1]))
        #print(theta)
        return theta

    def feedForward(self, x, at):
        # Retorna o y calculado com os thetas atuais 
        self.x = x
        z = x
        if at == True:
            self.z = []
            self.a = []
      

        for i in range(len(self.topologia) - 1):
            #z = np.dot(NeuralNet.addbias(z), self.thetas[i])
            z = np.dot(z, self.thetas[i])
            if at == True:
                self.z.append(z)

            a = NeuralNet.sigmoid(z)
            if at == True:
                self.a.append(a)
        self.y = a.flatten()
        return a # a = y
             
    def backpropagate(self, x, y, alpha):
        #Calcula os gradientes e pesos
        dif = (y - self.y)[np.newaxis].T
        delta3 = np.multiply(-dif,self.sigmoidPrime(self.z[1]))
        dw2 = self.a[0].T.dot(delta3)

        delta2 = delta3.dot(self.thetas[1].T) * self.sigmoidPrime(self.z[0])
        dw1 = (x.T).dot(delta2)

        self.thetas[0] -= alpha * dw1
        self.thetas[1] -= alpha * dw2
        
            
        
       
    def train(self, x, y, alpha, maxEpochs, minError):
        #Calcula o gradiente e o custo, at� que se atinja o maxEpochs ou custo < minError
        
        epoch = 0
        error = self.Cost(y, self.feedForward(x, True))
        while(epoch < maxEpochs and error > minError):
            self.backpropagate(x, y, alpha)
            error = self.Cost(y, self.feedForward(x, True))    
            epoch += 1
        pass
    
    @staticmethod
    def Cost(targetY, computedY):
        dif = (targetY-computedY)
        dif = np.power(dif, 2)
        return dif.sum() * 0.5

    @staticmethod
    def addbias(matriz):
        return np.column_stack(([1] * matriz.shape[0], matriz))
        #matriz = matriz.T
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoidPrime(x):
        f = NeuralNet.sigmoid(x)
        d = 1-f
        r = f * (d)
        return r


net = NeuralNet([3, 2, 1])

p1 = [0.7, 0.5, 0.4]
p2 = [0.8,0.4,0.3]
p3 = [0.9,1,0]
p4 = [0.2,1,0.9]
p5 = [0.3,0.3,0]
p6 = [0.5,0.5,0.6]
p7 = [0.6,0.7,0.5]
p8 = [0,1,0.3]

pt = [0.8,0.8,0.0]

r = [0.7,0.7,0.9,0.8,0.5,1,0.5,0.8]
lista = [p1, p2, p3, p4, p5, p6, p7, p8]
x = np.array(lista)
y = np.array(r)
y1 = np.array(r)
y2 = np.array([[7,7,10,8]])
print(lista)
lista2 = [pt] + lista[1:]
print(lista2)
print(np.array(lista2))
net.train(x, y, 0.3, 10000, 0.2)
print(net.y)
print(net.feedForward(lista2, False))

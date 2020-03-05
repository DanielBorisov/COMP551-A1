import numpy as np
import pandas as pd

# functions to "modify" the data (square, log etc) transformation
def asis(aX):
    return aX

def squared(aX):
    aX2=np.square(aX)
    jointaX = aX.join(aX2, rsuffix = "_sqr")#pd.concat([aX,aX2], axis=1)#pd.merge(aX, aX2)
    return jointaX

def loged(aX):
    return np.log(aX+1)

def logedplus(aX):
    aX2 = np.log(aX+1)
    jointaX = aX.join(aX2, rsuffix="_log")
    return jointaX

def get_covar(aX, aY):
    classes = [0,1]#pd.unique(aY)#[0, 1]
    #print(aX)
    N = len(aY)
    class_u = pd.DataFrame(columns=classes)
    jointdf = aX.join(pd.Series(aY, name="class"))
    #print(jointdf)
    for i, rows in jointdf.groupby("class"):
        # remove the class
        rows = rows.drop(["class"], axis=1)
        class_u[i] = rows.mean()

    f_size = len(aX.columns)
    Sw = np.zeros((f_size, f_size))
    for i, rows in jointdf.groupby("class"):
        # remove the class
        rows = rows.drop(["class"], axis=1)
        # empty base
        s = np.zeros((f_size, f_size))

        for j, row in rows.iterrows():
            x = row.values.reshape(f_size, 1)
            ux = class_u[i].values.reshape(f_size, 1)

            s += (x - ux).dot((x - ux).T)
        Sw += s

    cova = Sw / (N - 2)
    return cova

# This is the one we use for Linear Discriminant Analysis
def cross_valid(pX,class_var,leave_prop,iter,myfunc=asis):
    sample_size = len(pX)
    leave_size = int(sample_size/iter*1.0)#*sample_size
    training_set ={}
    validation_set ={}
    set_try = {}
    pX = pX.sample(frac=1).reset_index(drop=True)
    #generate the sets
    for i in range(iter):
        #set seed, or shuffle

        leave_set = range(i*leave_size,(i+1)*leave_size)#random.sample(range(sample_size),leave_size)
        #print(leave_set)
        training_set[i] = pX.drop(leave_set)
        validation_set[i] = pX.iloc[leave_set, :]
        #print(training_set[i])
        #print(validation_set[i])
        #------

    #run the sets
    acc_list = []
    for i in range(iter):
        #print(str(i) + " -------------------------")
        variables = training_set[i].drop(class_var, axis=1)
        aX = variables
        aY = training_set[i][class_var]
        #print(aX)

        set_try[i] = LDA(myfunc)
        set_try[i].fit(aX, aY)
        #-------
        #do prediction
        # n for "new" (validation variables)
        nvariables = validation_set[i].drop(class_var, axis=1)
        nX = nvariables
        nY = validation_set[i][class_var]

        set_prediction = list(set_try[i].predict(nX))
        acc_list.append(evaluate_acc(aX, nY, set_prediction))

    #just to keep a copy of both covariance approach
    #global covar0, covar1;
    #covar0 = set_try[i].COVAR
    #covar1 = set_try[i].covarv2
    #Show results
    print("cross-validation accuracies")
    print(acc_list)
    print("accuracy")
    print (np.mean(acc_list))

    return set_try, acc_list        

def evaluate_acc(aX, aY, tY):
    #t for target

    #print(tY)
    #print (list(aY))
    validity = [int(list(aY)[element] == tY[element]) for element in range(len(aY))]
    #print(validity)
    accuracy = np.mean(validity)
    #print(accuracy)
    return accuracy

def I(value):
    return value.astype(int)
        
class LDA():

    def __init__(self,amF=asis):
        #Only takes the transformation function
        self.model = amF

        #covar4 = np.cov(aX, rowvar=False)
        self.aX = 0
        self.N0 = 0
        self.N1 = 0
        self.P0 = 0 # probablilty of class 0
        self.P1 = 0  # probability of calss 1
        self.COVAR = 0
        self.mean = 0
        self.predict_y = 0
        self.predict_y_bin = 0


        print("")
        #return self

    def fit(self, pX, aY):

        aX = self.model(pX)
        classval = pd.unique(aY)
        #print(classval)
        N0 = aY[aY == 0].count()
        N1 = aY[aY == 1].count()
        P0 = N0 / (N0 + N1 * 1.0)  # probablilty of class 0
        P1 = N1 / (N0 + N1 * 1.0)  # probability of calss 1
        #print(aX)

        COVAR = get_covar(aX,aY)#np.cov(aX, rowvar=False)#
        covarv2 = get_covar(aX, aY)
        mean = {}
        mean[0] = aX.apply(lambda x: sum(x * I(aY[x.index] == 0)) / N0)  # sum(axis = 0)
        mean[1] = aX.apply(lambda x: sum(x * I(aY[x.index] == 1)) / N1)

        self.aX = aX
        self.aY = aY
        self.N0 = N0
        self.N1 = N1
        self.P0 = P0  # probablilty of class 0
        self.P1 = P1  # probability of calss 1
        self.COVAR = COVAR
        self.covar2 = covarv2
        self.mean = mean


    def predict(self, nX):
        P0 =self.P0
        P1 = self.P1
        mean = self.mean
        COVAR = self.COVAR

        inputX = self.model(nX)

        #Predicts y
        predict_y = np.log(P1 / P0) - 1 / 2.0 * (np.dot(mean[1].T, np.linalg.inv(COVAR)).dot(mean[1])) + \
        1 / 2.0 * (np.dot(mean[0].T, np.linalg.inv(COVAR)).dot(mean[0])) + \
        (np.dot(inputX, np.linalg.inv(COVAR)).dot((mean[1] - mean[0])))

        #turns probabilities to binary, treshold at 0.5
        predict_y_bin = (predict_y > 0.5).astype(int)

        self.predict_y = predict_y
        self.predict_y_bin = predict_y_bin
        #print(predict_y_bin)
        return predict_y_bin

    def show(self):
        print("aX:"+str(self.aX))
        print("aY:" + str(self.aY))
        print("N0:" + str(self.N0))
        print("N1:" + str(self.N1))
        print("P0:" + str(self.P0))
        print("P1:" + str(self.P1))
        print("COVAR:" + str(self.COVAR))
        print("covar2:" + str(self.covar2))
        print("Mean:" + str(self.mean))
        print("predict:" + str(self.predict_y))
        print("predict_bin:" + str(self.predict_y_bin))


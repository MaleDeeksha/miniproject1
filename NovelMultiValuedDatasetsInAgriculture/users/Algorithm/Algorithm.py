from django.conf import settings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import Normalizer


class Algorithms:
    path = settings.MEDIA_ROOT + "\\" + "Agriculture.csv"
    data = pd.read_csv(path, delimiter=',')
    data = data.drop(["ID"], axis=1)
    print(data)

    x = data.iloc[:, :8]
    y = data.iloc[:, -1]
    x = pd.get_dummies(x)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=1, stratify=y)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    x_train = pd.DataFrame(x_train)
    x_train.head()

    def Nb(self):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn import metrics

        mnb = MultinomialNB()

        self.x_train = np.abs(self.x_train)

        # print("c:", self.x_train)
        # print("d:", self.y_train)
        mnb.fit(self.x_train, self.y_train, sample_weight=1)
        y_pred = mnb.predict(self.x_test)

        print("a:", y_pred)
        # print("e:", self.y_test)

        dt_acc = metrics.accuracy_score(self.y_test, y_pred)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=[10, 6], dpi=100, )
        plt.title('Naive Bayes Comparision Chart')
        plt.xlabel('Algorithms')
        plt.ylabel('Accuracy')
        pf = sns.barplot(x=['dt_acc', 'mae', 'mse', 'r2'],
                         y=[83547, 24262, 19056, 17601],
                         palette='dark')
      
        plt.show()
        return dt_acc,  mae, mse, r2, pf

    def Mlp(self):
        from sklearn.neural_network import MLPClassifier

        mlpclassifier = MLPClassifier(random_state=2, max_iter=550)
        mlpclassifier.fit(self.x_train, self.y_train)
        y_pred = mlpclassifier.predict(self.x_test)
        from sklearn import metrics
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=[10, 6], dpi=100, )
        plt.title('MultiLayer Perceptron Comparision Chart')
        plt.xlabel('Algorithms')
        plt.ylabel('Accuracy')
        pf = sns.barplot(x=['dt_acc', 'mae', 'mse', 'r2'],
                         y=[69059, 52063, 37981, 30373],
                         palette='dark')
        plt.show()

        return accuracy, mse, mae, r2, pf

    def Approch(self):
        from sklearn.naive_bayes import MultinomialNB
        mnb = MultinomialNB()

        self.x_train = np.abs(self.x_train)
        mnb.fit(self.x_train, self.y_train)
        y_pred = mnb.predict(self.x_test)
        mae1 = mean_absolute_error(self.y_test, y_pred)
        mse1 = mean_squared_error(self.y_test, y_pred)

        from sklearn.neural_network import MLPClassifier
        mlpclassifier = MLPClassifier(random_state=2, max_iter=550)
        mlpclassifier.fit(self.x_train, self.y_train)
        y_pred = mlpclassifier.predict(self.x_test)

        mae2 = mean_absolute_error(self.y_test, y_pred)
        mse2 = mean_squared_error(self.y_test, y_pred)

        mae = np.array([[mae1], [mae2]])
        mae = np.mean(mae)
        mse = np.array([[mse1], [mse2]])
        mse = np.mean(mse)

        return mae, mse

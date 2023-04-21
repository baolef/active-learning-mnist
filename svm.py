# Created by Yuan Chen on 03/26/2023

import matplotlib.pyplot as plt
from sklearn.svm import SVC
import data


class SVM:
    def __init__(self, X_train, X_test, y_train, y_test, C=1, kernel='rbf'):
        """
        Initialize input and parameter
        :param C: specify "C" for regularization
        :param kernel: choices of kernel
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.C = C
        self.kernel = kernel


    def train(self):
        """
        train SVC model
        """
        model = SVC(C=self.C, kernel=self.kernel)
        model.fit(self.X_train, self.y_train)
        return model

    
    def predict(self):
        """
        make predictions on the test set
        """
        model = self.train()
        print("Finished training.")

        prediction = model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)
        print("Score: {}.".format(score))

        return prediction, score


    def visualize_test(self):
        """
        a plot for y_test
        :return: None
        """
        plt.figure(figsize=(8, 6))

        # plot y_test
        test_color = self.y_test.astype('int')

        plt.title("Classes of y_test after PCA", fontsize="small")
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], s=1, c=test_color)

        plt.savefig("SVM_y")
        plt.show()
        

    def visualize_prediction(self):
        """
        a plot for prediction
        :return: None
        """
        plt.figure(figsize=(8, 6))

        # plot prediction
        prediction, score = self.predict()
        prediction_color = prediction.astype('int')
        plt.title("Classification for X_test", fontsize="small")
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], s=1, c=prediction_color)

        plt.savefig("SVM_prediction")
        plt.show()


if __name__ == '__main__':
    dataset = data.Dataset()
    X_test, y_test = dataset.get_test()
    X_train, y_train = dataset.get_train()

    classifier = SVM(X_train, X_test, y_train, y_test, 1, 'poly')
    classifier.visualize_prediction()

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import statistics as stat

class CustomGradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.initial_prediction_ = None
        self.trees_ = []
    
    def fit(self, X, y):
        
        # Empezamos con la lista de arboles vacía
        self.trees_ = []

        # Calcular la predicción inicial
        self.initial_prediction_ = stat.mean(y)


        current_prediction = np.full(
            len(y),
            self.initial_prediction_

        )

        for epoch in range (self.n_estimators):

            # Calcular los residuos

            residuals = y - current_prediction

            # Crear arbol devil

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state,
                                         )
            tree.fit(X, residuals)

            tree_predictions = tree.predict(X)

            current_prediction += (
                self.learning_rate * tree_predictions
            )

            self.trees_.append(tree)

        return self

    
    def predict(self, X):
        if self.initial_prediction_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        predictions = np.full(
            len(X),
            self.initial_prediction_
        )

        for tree in self.trees_:
            tree_predictions = tree.predict(X)

            predictions += self.learning_rate * tree_predictions

        return predictions
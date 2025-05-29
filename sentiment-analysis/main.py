import os, json
import numpy as np
from article_embedding import embeddingDB
from sentiment_analysis import customNN
import time

if __name__ == "__main__":
    db = embeddingDB()

    x_train, train_text, y_train = db.load_local_db(True)
    x_test, test_text, y_test = db.load_local_db(False)
    self_x, self_text, self_y = db.load_local_db(False, True)

    print(f"Data loaded.")

    nn = customNN(
        input_dim=x_train.shape[1],
        x_train=x_train,
        y_train=y_train
    )

    print(f"Training model...")

    (model, hp) = nn.search()

    print(f"Best val_acc model:")
    print(f"Structure: {hp.get('structure')}; Learning rate: {hp.get('learning_rate')}")

    (test_ac, test_cm, test_cr) = nn.evaluate(x_test, y_test)
    (self_ac, self_cm, self_cr) = nn.evaluate(self_x, self_y)

    print(f"Test accuracy on keras test set: {test_ac}")
    print(f"Test accuracy on self test set: {self_ac}")
    print(f"Test confusion matrix on keras test set:\n{test_cm}")
    print(f"Test confusion matrix on self test set:\n{self_cm}")
    print(f"Test classification report on keras test set:\n{test_cr}")
    print(f"Test classification report on self test set:\n{self_cr}")

    current_time = time.strftime("%Y%m%d_%H%M%S")
    print(f"Saving model to models/{hp.get('structure')}_{hp.get('learning_rate')}_{current_time}.keras")
    nn.save_model(f"models/{hp.get('structure')}_{hp.get('learning_rate')}_{current_time}.keras")
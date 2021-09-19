"""
author: Keshab
email:keshab@abc.com
"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s, %(levelname)s, %(module)s, %(lineno)s]  %(message)s"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"runlog.log"), level=logging.INFO, format=logging_str, filemode="a")


def main(data, eta, epoch, modelFileName, plotFileName):
    df_and = pd.DataFrame(data)
    model_and = Perceptron(eta, epoch)
    x, y = prepare_data(df_and)
    model_and.fit(x, y)
    _ = model_and.total_loss()
    save_model(model_and, modelFileName)
    save_plot(df_and, plotFileName, model_and)


if __name__ == "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]
    }
    ETA = 0.3
    EPOCHS = 1000
    try:
        logging.info(">>>>> Starting <<<<<<")
        main(AND, ETA, EPOCHS, "and.model", "and.png")
        logging.info(">>>>> Stoping <<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e

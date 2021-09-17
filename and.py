from utils.all_utils import save_model, save_plot
from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

def main(data, eta, epoch, modelFileName, plotFileName):
    df_and=pd.DataFrame(data)
    model_and = Perceptron(eta, epoch)
    x,y = prepare_data(df_and)
    model_and.fit(x,y)
    _=model_and.total_loss()
    save_model(model_and, modelFileName)
    save_plot(df_and, plotFileName, model_and)

if __name__ == "__main__":
    AND={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(AND, ETA, EPOCHS, "and.model", "and.png")
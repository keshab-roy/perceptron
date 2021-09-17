from utils.all_utils import save_model, save_plot
from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd


AND={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}
df_and=pd.DataFrame(AND)
ETA = 0.3
EPOCHS = 10
model_and = Perceptron(ETA, EPOCHS)
x,y = prepare_data(df_and)
model_and.fit(x,y)
_=model_and.total_loss()
save_model(model_and, "and.model")
save_plot(df_and, "and.png", model_and)
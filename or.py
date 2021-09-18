"""
author: Keshab
email:keshab@abc.com
"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd


OR={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
}
df_or=pd.DataFrame(OR)
ETA = 0.3
EPOCHS = 10
model_or = Perceptron(ETA, EPOCHS)
x,y = prepare_data(df_or)
model_or.fit(x,y)
_=model_or.total_loss()
save_model(model_or, "or.model")
save_plot(df_or, "or.png", model_or)
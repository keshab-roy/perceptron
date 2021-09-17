import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
import os

def prepare_data(df):
    x = df.drop("y", axis=1)
    y = df["y"]
    return x, y

def save_model(model, filename):
    model_dir="models"
    os.makedirs(model_dir, exist_ok=True)
    filepath=os.path.join(model_dir, filename)
    joblib.dump(model, filepath)

def save_plot(df, file_name, model):
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2",c="y",s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure=plt.gcf() # Get current figure
        figure.set_size_inches(10,8)
        
    def _plot_decision_region(x,y,classifier,resolution=0.02):
        colors=("red","blue","lightgreen","grey","cyan")
        cmap=ListedColormap(colors[:len(np.unique(y))])
        
        x=x.values #as a array
        x1_min, x1_max=x[:,0].min() -1, x[:,0].max() +1
        x2_min, x2_max=x[:,1].min() -1, x[:,1].max() +1
        print(f"x1_min: {x1_min}")
        print(f"x1_max: {x1_max}")
        print(f"x2_min: {x2_min}")
        print(f"x2_max: {x2_max}")
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z=z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()
    
    x,y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_region(x,y,model)
    plot_dir="plots"
    os.makedirs(plot_dir, exist_ok=True)
    plotpath=os.path.join(plot_dir, file_name)
    plt.savefig(plotpath)
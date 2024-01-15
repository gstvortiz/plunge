import os
import shutil
import datetime
from PIL import Image
from itertools import combinations
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

class MiningVisualizer():
    def __init__(self, data, subset = ['X', 'Y', 'Z'], figsize = 8, fontsize = 15, s = 30, elev = 30, azim = -75, labelpad = 10, cmap = 'turbo', colorbar = True):
        self.data = data.dropna(subset = subset)
        self.subset = subset
        self.figsize = figsize
        self.fontsize = fontsize
        self.labelpad = labelpad
        self.s = s
        self.elev = elev
        self.azim = azim
        self.cmap = cmap
        self.colorbar = colorbar

    def Plot(self, var = None, var_kind = None):
        fig = plt.figure(figsize=(self.figsize, self.figsize))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(var, fontsize = self.fontsize, pad = self.labelpad)
        ax.set_xlabel(self.subset[0], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.set_ylabel(self.subset[1], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.set_zlabel(self.subset[2], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.view_init(elev = self.elev, azim = self.azim)

        if var_kind == 'Discreta':
            scatter_list = []
            data = self.data.dropna(subset = var)
            grupos = sorted(data[var].unique())
            colors = sns.color_palette(self.cmap, n_colors=len(grupos))
            for i, grupo in enumerate(grupos):
                grupo_data = data[data[var] == grupo]
                scatter = ax.scatter(data=grupo_data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], s = self.s, color = colors[i], depthshade=False, label = grupo)
                scatter_list.append(scatter)
            ax.legend(fontsize=self.fontsize*0.75, markerscale=2)

        elif var_kind == 'Contínua':    
            data = self.data.dropna(subset = var)
            scatter = ax.scatter(data = data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], c = var, cmap = self.cmap, s = self.s, depthshade=False)
            if self.colorbar:
                plt.subplots_adjust(left = 0, right = 0.8, bottom = 0, top = 1)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.6])
                cbar = fig.colorbar(scatter, cax = cbar_ax)
                
        else:
            scatter = ax.scatter(data = self.data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], s = self.s, depthshade=False)
            ax.set_title(None)
        return fig, ax
    
    def Expand(self, hue):
        fig, axs = plt.subplots(1, 3, figsize = (20, 6))
        fig.suptitle(f'Geospatial Visualization: {hue}')
        combinacoes = list(combinations(self.subset, 2))
        for i, (x, y) in enumerate(combinacoes):
            sns.scatterplot(data = self.data, x = x, y = y, hue = hue, palette='turbo', s = 40, ax = axs.ravel()[i])

class Correlation():
    def __init__(self, data, round = 2):
        self.data = data
        self.corr = data.corr(numeric_only=True)
        self.cmap = sns.diverging_palette(10, 150, n=1000, center='light') 
        self.fmt = f".{round}f"

    def Heatmap(self, figsize, fontsize, ticksfontsize):
        ax = sns.heatmap(data=self.corr, vmin=-1, vmax=1, annot=True, cmap=self.cmap, fmt = self.fmt, xticklabels=self.corr.columns, yticklabels=self.corr.columns, annot_kws={"size": ticksfontsize})
        ax.figure.set_size_inches(figsize)
        ax.set_title('Matriz de Correlação', fontdict={'fontsize':fontsize}, pad=16)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize = ticksfontsize, rotation=-75)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = ticksfontsize)
        return ax

    def Columns(self, colsX, colsY, lim = 0.5, figsize = (5, 6), wspace = 4):
        fig, axs = plt.subplots(1, len(colsY), figsize=figsize)   
        fig.subplots_adjust(hspace=0.3, wspace=wspace)
        for col, ax in zip(colsY, axs.ravel()):
            dados = self.data[list(set(colsX + [col]))].corr(numeric_only = True)
            data = dados[col].sort_values(ascending = False)[1:]
            cell = data[(data < -lim) | (data > lim)].to_frame()
            sns.heatmap(cell, vmin=-1, vmax=1, annot=True, cmap=self.cmap, fmt = self.fmt, cbar = False, ax = ax)
            ax.set_title(col, fontsize = 14, y = 1.02)
        return fig, ax
    
    def getTopCorrelations(self, col, qtd):
        data = self.corr[col].abs().sort_values(ascending=False).dropna()[1:qtd+1]
        return list(data.index)

def nullData(data, cols, title, width, height, fontsize):
    df = data.shape[0] - data[cols].isna().sum()
    ax = sns.barplot(x = df.index, y = df.values, color = 'royalblue')
    ax.figure.set_size_inches(width, height)
    ax.set_title(title, fontsize = fontsize*1.5)
    ax.set_xlabel('Columns', fontsize = fontsize)
    ax.set_ylabel('Count', fontsize = fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = -90, fontsize = fontsize)
    plt.show()

def dictValueCounts(series, normalize):
    data = series.value_counts(normalize = normalize).to_dict()
    return {str(key): value for key, value in data.items()}

def countplot(data, x, ax = False, palette = 'tab10', fontsize = 8, limit = 8):
    ax = plt.subplots()[1] if not ax else ax 

    sns.countplot(data = data, x = x, hue = x, palette = palette, dodge = False, legend = True, ax = ax)
    ax.set_title(ax.get_xlabel(), fontsize = fontsize)
    ax.set_xlabel(None)
    ax.set_ylabel('Count', fontsize = fontsize)
    ax.set_ylim(0, data.shape[0])

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) < limit: 
        numericalCounts = dictValueCounts(data[x], False)
        for i, label in enumerate(labels):
            ax.text(i, numericalCounts[label], f'{numericalCounts[label]}', ha='center', va='bottom', fontsize = fontsize * 0.75)
            
        normalizedCounts = dictValueCounts(data[x], True)
        labels = ['{:.2f} %'.format(normalizedCounts[str(i)] * 100) for i in labels]
        ax.legend(handles = handles, labels = labels, loc = 'best', fontsize = fontsize * 0.75)

    else:
        ax.get_legend().remove()

def histplot(data, x, ax = False, fontsize = 8, gap = 1):
    ax = plt.subplots()[1] if not ax else ax
    stats = 'N: {} \nM: {:.2f} \nx̄: {:.2f} \nσ: {:.2f} \ncv: {:.2f} \n'.format(
            data[x].dropna().shape[0], 
            data[x].median(),
            data[x].mean(),
            data[x].std(),
            data[x].std()/data[x].mean())
    ax = sns.histplot(data = data, x = x, kde = True, ax = ax)
    ax.set_title(ax.get_xlabel(), fontsize = fontsize)
    ax.set_xlabel(None)
    ax.set_ylabel('Count', fontsize = fontsize)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]*gap)
    ax.text(s = stats, x = ax.get_xlim()[1], y = ax.get_ylim()[1]*0.99, fontsize = fontsize, ha='right', va='top')

def scatterplot(data, x, y, ax = False, fontsize = 8, gap = 1.3, color = None):
    Corr = lambda method: data[[x, y]].corr(method = method).iloc[0, 1]
    stats = 'N: {} \nPsn: {:.2f} \nSpm: {:.2f} \nKnd: {:.2f} '.format(
            data[[x, y]].dropna().shape[0],
            Corr('pearson'), 
            Corr('spearman'),
            Corr('kendall'))
    color = sns.diverging_palette(10, 150, n=100, center='light')[int(50*(Corr('pearson')**3+1))] if not color else color
    
    ax = plt.subplots()[1] if not ax else ax
    ax = sns.scatterplot(data = data, x = x, y = y, color = color, ax = ax)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]*gap)
    ax.text(s = stats, x = ax.get_xlim()[1], y = ax.get_ylim()[1]*0.99, fontsize = fontsize * 0.8, ha='right', va='top')
    ax.set_title(ax.get_xlabel(), fontsize = fontsize)
    ax.set_xlabel(None)
    ax.set_ylabel(ax.get_ylabel(), fontsize = fontsize)

def PredictionError(y_test, y_pred, ax = None, color = 'forestgreen'):
    ax = plt.subplots()[1] if not ax else ax

    sns.regplot(x = y_test, y = y_pred, color = color, ax = ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim, xlim, 'r--')
    text = '\n R²: {:.2f}\n RMSE: {:.2f}\n MAPE: {:.2f}'.format(
        r2_score(y_test, y_pred),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        mean_absolute_percentage_error(y_test, y_pred)
    )
    ax.text(s = text, x = xlim[0], y = ylim[1], fontsize = 10, ha='left', va='top')
    ax.set_title(f'Prediction Error ({len(y_pred)})')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')

def FeatureImportances(data, regressor, target, n = 7, ax = False):
    ax = plt.subplots()[1] if not ax else ax 
    
    df = pd.DataFrame()
    df['Features'] = data.drop(target, axis = 1).columns
    df['Importances'] = abs(regressor.feature_importances_)
    df['Correlation'] = df['Features'].map(data.corr()[target]).apply(lambda x: 'Positive' if x >=0 else 'Negative')

    df = df.sort_values(by = 'Importances', ascending = False)[:n].T
    df['Others'] = ['Others', 1 - df.loc['Importances'].sum(), '']

    sns.barplot(data = df.T, x = 'Features', y = 'Importances', hue = 'Correlation', ax = ax)
    ax.set_title(f'Feature Importances ({data.shape[1]})')

def reencoder(data, columns, sep = '-'):
    items = {}
    for item in columns:
        key, value = item.split(sep)
        items.setdefault(key, []).append(item)
    df = data.copy()
    for item in items:
        df[item] = (df[items[item]] == 1).idxmax(1).apply(lambda x: x.split(sep)[1])
    return df, list(items.keys())
import os
import shutil
from PIL import Image
import datetime
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
        cols = {'XY': {'x': 'X', 'y': 'Y'}, 'XZ': {'x': 'X', 'y': 'Z'}, 'YZ': {'x': 'Y', 'y': 'Z'}}
        fig, axs = plt.subplots(1, 3, figsize = (20, 6))
        fig.suptitle(f'Visualização Geoespacial: {hue}')
        for i, col in enumerate(cols):
            sns.scatterplot(data = self.data, x = cols[col]['x'], y = cols[col]['y'], hue = hue, palette='turbo', s = 40, ax = axs.ravel()[i])
            axs.ravel()[i].set_title(col)
        plt.show()

    def GIF(self, col, datapath = 'dados', kind = 'Discreta'):
        folder = f'{datapath}/{col}'
        GIF(folder).Preparar()
        for azim in range(0, 360, 2):
            self.azim = azim
            self.elev = 30 + 15*np.sin(azim/57.29)
            self.Plot(col, kind)
            plt.savefig('{}/{}'.format(folder, azim))
            plt.close()
        GIF(folder).Make(col)





class GIF():
    def __init__(self, folder = 'dados/img'):
        self.folder = folder

    def Preparar(self):
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.mkdir(self.folder)
    
    def OrderedPaths(self):
        paths = []
        imagens = os.listdir(self.folder)
        for imagem in imagens:
            caminho_completo = os.path.join(self.folder, imagem)
            data_criacao = datetime.datetime.fromtimestamp(os.path.getctime(caminho_completo))
            paths.append((caminho_completo, data_criacao))
        paths.sort(key=lambda x: x[1])
        return np.array(paths)[:, 0]
    
    def Make(self, name):
        imagens = []
        for path in self.OrderedPaths():
            imagem = Image.open(path)
            imagens.append(imagem)
        imagens[0].save(f'{name}.gif', save_all=True, append_images=imagens[1:], duration=70, loop=0)





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
    




def dadosNulos(dados, colunas, titulo, width, height, fontsize):
    data = dados.shape[0] - dados[colunas].isna().sum()
    ax = sns.barplot(x=data.index, y=data.values, color = 'royalblue')
    ax.figure.set_size_inches(width, height)
    ax.set_title(titulo, fontsize = fontsize*1.5)
    ax.set_xlabel('Variáveis do Conjunto', fontsize = fontsize)
    ax.set_ylabel('Frequência Absoluta', fontsize = fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = -90, fontsize = fontsize)
    plt.show()





def countplot(data, x, ax, fontsize = 8):
    ax = sns.countplot(data = data, x = x, hue = x, dodge = False, ax = ax)
    ax.set_title(ax.get_xlabel())
    ax.set_xlabel(None)
    ax.set_ylabel('Quantidade')
    ax.set_ylim(0, data.shape[0])

    dictLegend = data[x].value_counts(normalize=True).round(2).to_dict()
    handles, labels = ax.get_legend_handles_labels()
    labels = ['{:.2f} %'.format(dictLegend[float(i)] * 100) for i in labels]
    ax.legend(handles = handles, labels = labels, loc = 'best')

    for valor, contagem in data[x].value_counts().items():
            ax.text(valor, contagem, f'{contagem}', ha='center', va='bottom')





def histplot(data, x, ax, fontsize = 8, gap = 1):
    stats = 'N: {} \nM: {:.2f} \nx̄: {:.2f} \nσ: {:.2f} \ncv: {:.2f} \n'.format(
            data[x].dropna().shape[0], 
            data[x].median(),
            data[x].mean(),
            data[x].std(),
            data[x].std()/data[x].mean())
    ax = sns.histplot(data = data, x = x, kde = True, ax = ax)
    ax.set_title(ax.get_xlabel())
    ax.set_xlabel(None)
    ax.set_ylabel('Quantidade')
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]*gap)
    ax.text(s = stats, x = ax.get_xlim()[1], y = ax.get_ylim()[1]*0.99, fontsize = fontsize, ha='right', va='top')





def scatterplot(data, x, y, ax, fontsize = 8, gap = 1.3, color = None):
    Corr = lambda method: data[[x, y]].corr(method = method).iloc[0, 1]
    stats = 'N: {} \nPsn: {:.2f} \nSpm: {:.2f} \nKnd: {:.2f} '.format(
            data[[x, y]].dropna().shape[0],
            Corr('pearson'), 
            Corr('spearman'),
            Corr('kendall'))
    color = sns.diverging_palette(10, 150, n=100, center='light')[int(50*(Corr('pearson')**3+1))] if not color else color
    ax = sns.scatterplot(data = data, x = x, y = y, color = color, ax = ax)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]*gap)
    ax.text(s = stats, x = ax.get_xlim()[1], y = ax.get_ylim()[1]*0.99, fontsize = fontsize, ha='right', va='top')





def PredictionError(y_test, y_pred, ax = None, color = 'forestgreen'):
    ax = plt.subplots()[1] if not ax else ax
    sns.regplot(x = y_test, y = y_pred, color = color, ax = ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim, xlim, 'r--')
    text = f'\nR²: {r2_score(y_test, y_pred):.2f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\nMAPE: {mean_absolute_percentage_error(y_test, y_pred):.2f}'
    ax.text(s = text, x = xlim[0], y = ylim[1], fontsize = 10, ha='left', va='top')
    ax.set_title(f'Prediction Error ({len(y_pred)})')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Previsto')





def FeatureImportances(data, variavelInteresse, regressor, ax, qtdOutras = 5, rotation = 0):
    index = data.drop(variavelInteresse, axis = 1).columns
    feature_importances = pd.DataFrame(data = regressor.feature_importances_, index = index, columns = ['Importâncias'])
    feature_importances.sort_values(by = 'Importâncias', ascending = False, inplace =  True)
    outras = feature_importances[qtdOutras:].sum()[0]
    feature_importances = feature_importances[:qtdOutras].T
    feature_importances['Outras'] = outras
    correlacoes = [data[[coluna, variavelInteresse]].corr().iloc[0][1] for coluna in feature_importances.columns[:-1]]
    cores = ['olivedrab' if correlacao >= 0 else 'brown' for correlacao in correlacoes] + ['teal']

    sns.barplot(feature_importances, palette = cores, ax = ax)
    ax.set_title(f'Feature Importances')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
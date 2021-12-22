




# -*- coding: UTF-8 -*-

import sys

print(sys.path)
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/lib-dynload')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9')
sys.path.append('/opt/local/lib/libgs')

import sys

if (sys.stdout.encoding is None):
    print >> sys.stderr, "please set python env PYTHONIOENCODING=UTF-8, example: export PYTHONIOENCODING=UTF-8, when write to stdout."
    exit(1)

import selenium
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import bs4
from bs4 import BeautifulSoup
import requests
import os
import tabula
import tabula.io
import tabulate
import PyPDF2
import shutil
import camelot
import camelot.io as camelot
import _tkinter
import pandas as pd
from scipy import nan
import numpy as np
import openpyxl
import os
import io
import re
import errno
# import ghostscript
import locale
import fitz
import warnings
import xlrd
import xlrd3
from itertools import chain
import functools
import operator
import pylab
import matplotlib

matplotlib.use('TkAgg')
import pyreadstat

import tabula
import tabula.io
import tabulate
import PyPDF2
import shutil
import camelot
import camelot.io as camelot
import _tkinter
import pandas as pd
from scipy import nan
import numpy as np
import openpyxl
import os
import io
import re
import errno
# import ghostscript
import locale
import fitz
import warnings
import xlrd
import xlrd3
from itertools import chain
import functools
import operator
import pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from io import StringIO
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
import csv
import glob
import os
import re
import sys
import pandas as pd
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
from tika import parser
import json
import pyperclip, re
from itertools import chain
import regex

pd.options.mode.chained_assignment = None
import collections
from matplotlib.pyplot import figure

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import scipy.cluster.hierarchy as sch

import networkx as nx

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from sklearn.manifold import TSNE
import umap
import json

os.chdir('/Users/janoschkorell/Desktop/Wissenschaft/Statistik/Python/Netzwerktest/')
path = '/Users/janoschkorell/Desktop/Wissenschaft/Statistik/Python/Netzwerktest/'
path_export = '/Users/janoschkorell/Desktop/Wissenschaft/Statistik/Python/Netzwerktest/'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

network_df = pd.read_csv("characNS.csv", index_col=0)
network_df2 = pd.read_csv("rslt_df1.csv", index_col=0)

#### Netzwerk Koordinaten ####


G = nx.Graph()
G = nx.from_pandas_edgelist(network_df2, 'Sitzungsnummer', 'Drucksache_Nummer', create_using=nx.Graph())
pos = nx.spring_layout(G)

nodes = []
for p in pos:
    nodes.append(p)

vals = pos.values()
array = np.stack(vals, axis=0)
x_list = []
y_list = []

for node in array[:, 0]:
    node = node * 1000
    x_list.append(node)
for node in array[:, 1]:
    node = node * 1000
    y_list.append(node)

coordi = pd.DataFrame(nodes, columns=['Drucksache_Nummer'])
coordi['x'] = x_list
coordi['y'] = y_list

# Merge: Koordinaten hinzufügen

network_df3 = coordi.merge(network_df, on="Drucksache_Nummer")
network_df3['ThemaVD'] = network_df3['ThemaVD'].str.replace('ü', 'ü')
network_df3['ThemaVD'] = network_df3['ThemaVD'].str.replace('ä', 'ä')
network_df3['TitelVD'] = network_df3['TitelVD'].str.replace('Große', 'Grosse')

###!!!Vorher ändern!!!

network_df3.loc[network_df3['TitelVD_ID'] == -1, 'Dokumentart'] = '1'
network_df3.loc[network_df3['TitelVD_ID'] != -1, 'Dokumentart'] = '2'

###### Elements erstellen#######

nodes = network_df3.filter(
    ["Drucksache_Nummer", 'color', 'ThemaVD', 'TitelVD', 'year', 'month', 'day', 'x', 'y', 'Occurrences_aktuell',
     'Dokumentart'])

mapping = {nodes.columns[0]: 'id'}
nodes = nodes.rename(columns=mapping)
nodes['label'] = nodes['id']
d = nodes
afile = open(r'nodesBasis', 'wb')
pickle.dump(d, afile)
afile.close()

### Filter nodes


network_merge = network_df3.copy()
del network_merge['Sitzungsnummer']
edges1 = network_merge.merge(network_df2, on="Drucksache_Nummer")

try:
    edges1 = edges1.rename(columns={'TitelVD_x': 'TitelVD'})
    edges1 = edges1.rename(columns={'ThemaVD_x': 'ThemaVD'})

except:
    pass

edges = edges1.filter(["Sitzungsnummer", 'Drucksache_Nummer', 'TitelVD', 'ThemaVD'])
### Filter edges



### Ausnahme der Protokolle


size = 30

# Nodes
startup_elm_list = []
for a, b, them, tit, year, month, day, occ, x, y, doc in zip(nodes['id'], nodes['label'], nodes['ThemaVD'],
                                                             nodes['TitelVD'], nodes['year'], nodes['month'],
                                                             nodes['day'], nodes['Occurrences_aktuell'], nodes['x'],
                                                             nodes['y'], nodes['Dokumentart']):
    dict = {'data': {'id': str(a), 'label': str(b), 'titel': str(tit), 'thema': str(them),
                     'datum': str(year) + '-' + str(month) + '-' + str(day), 'node_size': float(occ * size)},
            'position': {'x': float(x), 'y': float(y)},
            'classes': str(tit) + ' ' + str(them) + ' ' + 'node' + ' ' + str(doc)}
    startup_elm_list.append(dict)

# Edges

for a, b, c in zip(edges["Sitzungsnummer"], edges["Drucksache_Nummer"], edges['TitelVD']):
    dict = {'data': {'source': str(a), 'target': str(b)}, 'classes': str(c)}
    startup_elm_list.append(dict)

###############################App#######################################


####################Style#####################


col_swatch = px.colors.qualitative.Dark24

def_stylesheet = []
for i, v in zip(range(len(nodes["TitelVD"].dropna().unique())), nodes["TitelVD"].dropna().unique()):
    a = {"selector": '.' + str(v), "style": {"background-color": col_swatch[i], "opacity": 0.65}}
    def_stylesheet.append(a)

for i, v in zip(range(len(edges["TitelVD"].dropna().unique())), edges["TitelVD"].dropna().unique()):
    a = {"selector": '.' + str(v), "style": {"line-color": col_swatch[i]}}
    def_stylesheet.append(a)

def_stylesheet += [

    # Class selectors
    {
        'selector': '.2',
        'style': {
            "width": "data(node_size)",
            "height": "data(node_size)"
        }
    },
    {
        'selector': '.1',
        'style': {
            'background-color': "black"
        }
    },

]


###########################################


#############Kat_dropdown plus Anzahl###############

def dropdowns(nodes):
    nodesTopicCount = nodes.filter(["ThemaVD", 'TitelVD']).dropna()

    nodesTopicCount['ThemaVD'] = nodesTopicCount['ThemaVD'].str.replace('ü', 'ü')
    nodesTopicCount['ThemaVD'] = nodesTopicCount['ThemaVD'].str.replace('ä', 'ä')
    nodesTopicCount['TitelVD'] = nodesTopicCount['TitelVD'].str.replace('Große', 'Grosse')

    zTitel1 = nodes.TitelVD.value_counts(ascending=False)
    zTitel = zTitel1.to_dict()
    nodesTopicCount['CountTitel'] = nodesTopicCount['TitelVD'].map(zTitel)

    zTitel1 = nodes.ThemaVD.value_counts(ascending=False)
    zTitel = zTitel1.to_dict()
    nodesTopicCount['CountThema'] = nodesTopicCount['ThemaVD'].map(zTitel)

    nodesTopicCountTitel = nodesTopicCount.filter(["CountTitel", 'TitelVD']).dropna()
    nodesTopicCountTitel = nodesTopicCountTitel.drop_duplicates(subset=["TitelVD"], keep='first').sort_values(
        by='CountTitel', ascending=False)
    nodesTopicCountThema = nodesTopicCount.filter(["CountThema", 'ThemaVD']).dropna()
    nodesTopicCountThema = nodesTopicCountThema.drop_duplicates(subset=["ThemaVD"], keep='first').sort_values(
        by='CountThema', ascending=False)

    cat_startPSL = []
    for a, b in zip(nodesTopicCountTitel['TitelVD'], nodesTopicCountTitel['CountTitel']):
        cat_startPS = []
        cat_startPS.append(a)
        cat_startPS.append(b)
        cat_startPSL.append(cat_startPS)

    catThem_startPSL = []
    for a, b in zip(nodesTopicCountThema['ThemaVD'], nodesTopicCountThema['CountThema']):
        catThem_startPS = []
        catThem_startPS.append(a)
        catThem_startPS.append(b)
        catThem_startPSL.append(catThem_startPS)

    cat_start = [str(i[0]) + '  ' + '(' + str(i[1]) + ')' for i in cat_startPSL]
    catThem_start = [str(i[0]) + '  ' + '(' + str(i[1]) + ')' for i in catThem_startPSL]

    return cat_start, catThem_start


cat_start, catThem_start = dropdowns(nodes)

d = catThem_start
afile = open(r'them2_alt', 'wb')
pickle.dump(d, afile)
afile.close()

d = cat_start
afile = open(r'kat2_alt', 'wb')
pickle.dump(d, afile)
afile.close()

cat_startdict = []
for i in cat_start:
    a = {'label': str(i), 'value': str(i)}
    cat_startdict.append(a)

catThem_startdict = []
for i in catThem_start:
    a = {'label': str(i), 'value': str(i)}
    catThem_startdict.append(a)


d = ""
afile = open(r'edgesVordavorAb', 'wb')
pickle.dump(d, afile)
afile.close()

d = ""
afile = open(r'edgesVordavorAuf', 'wb')
pickle.dump(d, afile)
afile.close()

d = ''
afile = open(r'edgesVordavorAufCTL', 'wb')
pickle.dump(d, afile)
afile.close()


d = []
afile = open(r'edgesVordavorCatDrop', 'wb')
pickle.dump(d, afile)
afile.close()

#Input davor frei machen

# Klick aktuell speichern
d = ''
afile = open(r'input_id', 'wb')
pickle.dump(d, afile)
afile.close()


# edges
d = ''
afile = open(r'edges', 'wb')
pickle.dump(d, afile)
afile.close()



def set_Them_options(new_kat, new_them):

    cat_Sub1 = []
    for i in new_kat:
        a = re.sub(r" \(\d+\)", "", i).rstrip()
        cat_Sub1.append(a)

    # Entfernen von Klammer und Zahl von cat_start Liste
    cat_start1 = []
    for i in cat_start:
        a = re.sub(r" \(\d+\)", "", i).rstrip()
        cat_start1.append(a)

    # Schauen welche Eingabe fehlt, um diese wieder mit (0) aus Liste hinzuzufügen
    for i in cat_start1:
        if i not in cat_Sub1:
            new_kat.append(i + '  ' + '(0)')

    # Wieder ursprüngliche Ordnung von Liste wiederherstellen
    usr_catLOrd = []
    for v in cat_start:
        v = re.sub(r" \(\d+\)", "", v).rstrip()
        for i in new_kat:
            if i.startswith(v):
                usr_catLOrd.append(i)


    # Endgültige Liste für Option Dict
    usr_catLOrdMA = []
    for i in new_kat:
        for v in cat_start:
            b = re.sub(r" \(\d+\)", "", v).rstrip()
            if i.startswith(b) and i.endswith('(0)'):
                i = v
        usr_catLOrdMA.append(i)


    # Endgültige Value_liste wieder alle mit (0) aus Value_Liste rauslöschen

    kat2 = []
    for i in usr_catLOrd:
        if not i.endswith('(0)'):
            kat2.append(i)

    #########################################################################################################

    # Entfernen von Klammer und Zahl von Eingabe Liste

    them_Sub1 = []
    for i in new_them:
        a = re.sub(r" \(\d+\)", "", i).rstrip()
        them_Sub1.append(a)

    # Entfernen von Klammer und Zahl von cat_start Liste
    them_start1 = []
    for i in catThem_start:
        a = re.sub(r" \(\d+\)", "", i).rstrip()
        them_start1.append(a)

    # Schauen welche Eingabe fehlt, um diese wieder mit (0) aus Liste hinzuzufügen
    for i in them_start1:
        if i not in them_Sub1:
            new_them.append(i + '  ' + '(0)')

    # Wieder ursprüngliche Ordnung von Liste wiederherstellen
    usr_themLOrd = []
    for v in catThem_start:
        v = re.sub(r" \(\d+\)", "", v).rstrip()
        for i in new_them:
            a = re.sub(r" \(\d+\)", "", i).rstrip()
            if a == v:
                usr_themLOrd.append(i)

    # Endgültige Liste für Option Dict
    usr_themLOrdMA = []
    for i in new_them:
        for v in catThem_start:
            b = re.sub(r" \(\d+\)", "", v).rstrip()
            a = re.sub(r" \(\d+\)", "", i).rstrip()
            if a == b and i.endswith('(0)'):
                i = v
        usr_themLOrdMA.append(i)

    print(len(usr_themLOrdMA))

    # Endgültige Value_liste wieder alle mit (0) aus Value_Liste rauslöschen

    them2 = []
    for i in usr_themLOrd:
        if not i.endswith('(0)'):
            them2.append(i)


    # Endgültiges Options_dict

    # Endgültiges Options_dict
    new_katdict1 = [{'label': c, 'value': c} for c in usr_catLOrdMA]
    new_themdict1 = [{'label': c, 'value': c} for c in usr_themLOrdMA]


    return kat2, new_katdict1, them2, new_themdict1



def filter_data(usr_catL, usr_catThemL):


    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        usr_catL1 = []
        for i in usr_catL:
            a = re.sub(r" \(\d+\)", "", i).rstrip()
            usr_catL1.append(a)

    except:
        pass

    try:
        usr_catThemL1 = []
        for i in usr_catThemL:
            a = re.sub(r" \(\d+\)", "", i).rstrip()
            usr_catThemL1.append(a)
    except:
        pass

    ###### Elements erstellen#######

    ###!!!Vorher ändern!!!

    network_df3.loc[network_df3['TitelVD_ID'] == -1, 'Dokumentart'] = '1'
    network_df3.loc[network_df3['TitelVD_ID'] != -1, 'Dokumentart'] = '2'

    nodes = network_df3.filter(
        ["Drucksache_Nummer", 'color', 'ThemaVD', 'TitelVD', 'year', 'month', 'day', 'x', 'y', 'Occurrences_aktuell',
         'Dokumentart'])

    mapping = {nodes.columns[0]: 'id'}
    nodes = nodes.rename(columns=mapping)
    nodes['label'] = nodes['id']

    network_merge = network_df3.copy()
    del network_merge['Sitzungsnummer']
    edges1 = network_merge.merge(network_df2, on="Drucksache_Nummer")

    try:
        edges1 = edges1.rename(columns={'TitelVD_x': 'TitelVD'})
        edges1 = edges1.rename(columns={'ThemaVD_x': 'ThemaVD'})
    except:
        pass

    edgesVor = edges1.copy()
    edgesVor2 = edges1.copy()
    edgesVor3 = edges1.copy()

    edgesVor.to_csv('edgesVorAnfang.csv')


    ###########################Filter für Drops#########################


    edgesVor3 = edges1.copy()


    #logger.info(f"kat2_alt: {len(kat2_a)}")
    #logger.info(f"usr_catL1: {len(usr_catL1)}")
    #logger.info(f"cat_start: {len(cat_start)}")

    #logger.info(f"them2_a: {len(them2_a)}")
    #logger.info(f"usr_catThemL1: {len(usr_catThemL1)}")
    #logger.info(f"catThem_start: {len(catThem_start)}")

##############################################################################
##############################################################################
##############################################################################



    ### Filter edges


    try:
        file2 = open(r'edges', 'rb')
        edges = pickle.load(file2)
        file2.close()
    except:
        pass

    if edges != '':
        print('Referenz Anfang')
        print(len(edges))

        ### Edges welche gelöscht werden soll

        print('Delete Edges')

        # Rausfinden welche Titel
        edgesDel1 =  edgesVor3[(edgesVor3.Drucksache_Nummer.isin(edges))]
        edgesDel1 = edgesDel1.drop_duplicates(subset=['Drucksache_Nummer'])
        edgesDel1L = edgesDel1['TitelVD'].tolist()

        print('Titel zum Beibehalten')
        print(edgesDel1L)

        # Löschen von allem außer der TitelVDListe
        edgesDel =  edgesVor3[(edgesVor3.TitelVD.isin(edgesDel1L))]
        edgesDel =  edgesDel[(~edgesDel.Drucksache_Nummer.isin(edges))]
        edgesDel = edgesDel.drop_duplicates(subset=['Drucksache_Nummer'])
        edgesDelL = edgesDel['Drucksache_Nummer'].tolist()
        edgesDel.to_csv('edgesDel.csv')
        print(len(edgesDelL))





    #Komplett

    komplett = edgesVor3['Drucksache_Nummer'].tolist()

    #Referenz Beginn

    if edges == '':

        try:
            referenzDT = edgesVor3[(edgesVor3.Drucksache_Nummer.isin(edges))]
            referenz = referenzDT['Drucksache_Nummer'].tolist()
        except:
            referenz = komplett

    # Normalfall der Referenz
    if edges != '':

        print('Normalfall Referenz')
        referenzDT = edgesVor3[(edgesVor3.Drucksache_Nummer.isin(edges))]
        #referenzDT = referenzDT.drop_duplicates(subset=['Drucksache_Nummer'])
        referenz = referenzDT['Drucksache_Nummer'].tolist()


#################################################################################
#################################################################################

    if input_id == "cat_dropdown":

        # Normalfall der aktuellen Drucksachen
        aktuellCatthemDT = edgesVor3[(edgesVor3.TitelVD.isin(usr_catL1))]
        #aktuellCatthemDT = aktuellCatthemDT.drop_duplicates(subset=['Drucksache_Nummer'])
        aktuellCatthem = aktuellCatthemDT['Drucksache_Nummer'].tolist()


    if input_id == "catThem_dropdown":
        # Normalfall der aktuellen Drucksachen
        aktuellCatthemDT = edgesVor3[(edgesVor3.ThemaVD.isin(usr_catThemL1))]
        # aktuellCatthemDT = aktuellCatthemDT.drop_duplicates(subset=['Drucksache_Nummer'])
        aktuellCatthem = aktuellCatthemDT['Drucksache_Nummer'].tolist()

#################################################################################
#################################################################################

    # Spezialfall wenn einer der beiden Inputs off ist.
    if (len(aktuellCatthem) == 0 and usr_catThemL == []) or (len(aktuellCatthem) == 0 and usr_catL1 == []) :
        print('OFF')
        aktuellCatthemDT = edgesVor3[(edgesVor3.TitelVD.isin(usr_catL1))]
        #aktuellCatthemDT = aktuellCatthemDT.drop_duplicates(subset=['Drucksache_Nummer'])
        aktuellCatthem = aktuellCatthemDT['Drucksache_Nummer'].tolist()

    # Referenz wieder aktualisieren, wenn alle Inputs involviert sind
    if len(aktuellCatthem) == len(komplett):
        referenz = komplett


    print('referenz')
    print(len(referenz))

    print('aktuellCatthem')
    print(len(aktuellCatthem))


    if input_id == "catThem_dropdown":

        try:
            file2 = open(r'edges', 'rb')
            edges = pickle.load(file2)
            file2.close()
        except:
            pass

        if (bool(len(aktuellCatthem) < len(referenz))) == True:

            print('CatThem Ab')



            edgesVor = edgesVor[(edgesVor.ThemaVD.isin(usr_catThemL1))]

            try:
                edgesVor = edgesVor[(edgesVor.Drucksache_Nummer.isin(edges))]
            except:
                pass


            edges = edgesVor['Drucksache_Nummer'].tolist()

            # Speichern der edges

            d = edges
            afile = open(r'edges', 'wb')
            pickle.dump(d, afile)
            afile.close()


        if bool(len(aktuellCatthem) >=  len(referenz))  == True:


            print('CatThem Auf')

            edgesVor = edgesVor[(edgesVor.ThemaVD.isin(usr_catThemL1))]
            edges = edgesVor['Drucksache_Nummer'].tolist()

            d = edges
            afile = open(r'edges', 'wb')
            pickle.dump(d, afile)
            afile.close()




##############################################################################
##############################################################################
##############################################################################


    if input_id == "cat_dropdown":

        try:
            file2 = open(r'edges', 'rb')
            edges = pickle.load(file2)
            file2.close()
        except:
            pass

        if (bool(len(aktuellCatthem) < len(referenz))) == True:

            print('Cat Ab')


            edgesVor = edgesVor[(edgesVor.TitelVD.isin(usr_catL1))]

            try:
                edgesVor = edgesVor[(edgesVor.Drucksache_Nummer.isin(edges))]

            except:
                pass

            edges = edgesVor['Drucksache_Nummer'].tolist()

            # Speichern der edges

            d = edges
            afile = open(r'edges', 'wb')
            pickle.dump(d, afile)
            afile.close()

        if bool(len(aktuellCatthem) >=  len(referenz))  == True:

            print('Cat Auf')

            edgesVor = edgesVor[(edgesVor.TitelVD.isin(usr_catL1))]
           # edgesVor = edgesVor.drop_duplicates(subset=['Drucksache_Nummer'])
            edgesVor = edgesVor[(~edgesVor.Drucksache_Nummer.isin(edgesDelL))]
            edges = edgesVor['Drucksache_Nummer'].tolist()


            d = edges
            afile = open(r'edges', 'wb')
            pickle.dump(d, afile)
            afile.close()


            print('Cat Auf Edges')
            print(len(edges))

            edgesVor.to_csv('edgesVor.csv')


##############################################################################
##############################################################################
##############################################################################




    if bool(usr_catL1 == [] or usr_catThemL1 == []) and (len(referenz) != 0) == True:


        d = []
        afile = open(r'edges', 'wb')
        pickle.dump(d, afile)
        afile.close()

        referenz = ' '


        logger.info(f"Neustart")




##############################################################################
##############################################################################
##############################################################################



    d = usr_catL1
    afile = open(r'kat2_alt', 'wb')
    pickle.dump(d, afile)
    afile.close()

    d = usr_catThemL1
    afile = open(r'them2_alt', 'wb')
    pickle.dump(d, afile)
    afile.close()

    try:
        if bool(x == 1) == True:

            d = ""
            afile = open(r'kat2_alt', 'wb')
            pickle.dump(d, afile)
            afile.close()

            d = ""
            afile = open(r'them2_alt', 'wb')
            pickle.dump(d, afile)
            afile.close()
    except:
        pass




    try:
        d = input_id
        afile = open(r'Eingabe', 'wb')
        pickle.dump(d, afile)
        afile.close()
    except:
        pass


    ### Ausnahme der Protokolle

    protoL = []
    for a, b in zip(edgesVor["Sitzungsnummer"], edgesVor["Drucksache_Nummer"]):
        protoL.append(a)
        protoL.append(b)

    s = []
    for i in protoL:
        if i not in s:
            s.append(i)

    nodes = nodes[(nodes.id.isin(s))]
    nodes4 = nodes.copy()

    # Nodes
    elm_list = []
    for a, b, them, tit, year, month, day, occ, x, y, doc in zip(nodes['id'], nodes['label'], nodes['ThemaVD'],
                                                                 nodes['TitelVD'], nodes['year'], nodes['month'],
                                                                 nodes['day'], nodes['Occurrences_aktuell'], nodes['x'],
                                                                 nodes['y'], nodes['Dokumentart']):
        dict = {'data': {'id': str(a), 'label': str(b), 'titel': str(tit), 'thema': str(them),
                         'datum': str(year) + '-' + str(month) + '-' + str(day), 'node_size': float(occ * size)},
                'position': {'x': float(x), 'y': float(y)},
                'classes': str(tit) + ' ' + str(them) + ' ' + 'node' + ' ' + str(doc)}

        elm_list.append(dict)

    # Edges

    for a, b, c in zip(edgesVor["Sitzungsnummer"], edgesVor["Drucksache_Nummer"], edgesVor['TitelVD']):
        dict = {'data': {'source': str(a), 'target': str(b)}, 'classes': str(c)}
        elm_list.append(dict)

    return elm_list, usr_catThemL1, usr_catL1 ,nodes4


##############################################################################


app.layout = html.Div([
    cyto.Cytoscape(
        id='core_19_cytoscape',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '400px'},
        elements=startup_elm_list,
        stylesheet=def_stylesheet,
        minZoom=0.06,
    ),
    html.P(id='cytoscape-tapNodeData-output'),
    html.P(id='cytoscape-tapEdgeData-output'),
    html.P(id='cytoscape-mouseoverNodeData-output'),
    html.P(id='cytoscape-mouseoverEdgeData-output'),
    dbc.Col(
        [
            dbc.Badge(
                "Kategorien Thema:", color="info", className="mr-1"
            ),
            dbc.FormGroup(
                [
                    dcc.Dropdown(
                        id="catThem_dropdown",
                        options=catThem_startdict,
                        value=catThem_start,
                        multi=True,
                        style={"width": "100%"},
                    ),

                ]
            ),
            dbc.Badge(
                "Kategorien Titel:", color="info", className="mr-1"
            ),
            dbc.FormGroup(
                [
                    dcc.Dropdown(
                        id="cat_dropdown",
                        options=cat_startdict,
                        value=cat_start,
                        multi=True,
                        style={"width": "100%"},
                    ),

                ]
            ),

        ],
        sm=12,
        md=4,
    )
])


###############################################################


##########Callback Kat-Auswahl##############

@app.callback(
    Output("core_19_cytoscape", "elements"),
    Output('cat_dropdown', 'value'),
    Output('cat_dropdown', 'options'),
    Output('catThem_dropdown', 'value'),
    Output('catThem_dropdown', 'options'),
    [Input("cat_dropdown", "value")],
    [Input("catThem_dropdown", "value")],
    prevent_initial_call=True
)
def elements(usr_catL, usr_catThemL):

    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # if bool((len(usr_catThemL) != len(catThem_start)) or (len(usr_catL) != len(cat_start))) == True:

    logger.info(f"!!!Beginn: {input_id}")

    # Zusammenstellung von Netzwerk
    elements, p, q, nodesele = filter_data(usr_catL, usr_catThemL)
    # Neue Options/Values
    new_kat ,new_them = dropdowns(nodesele)
    # Neue Options/Values in richtiger Reihenfolge mit dict
    kat2, new_katdict1, them2, new_themdict1 = set_Them_options(new_kat, new_them)


    logger.info(f"usr_catL: {usr_catL}")
    logger.info(f"usr_catThemL: {usr_catThemL}")

    logger.info(f"new_themdict1: {new_themdict1}")

    """

    logger.info(f"them2: {them2}")

    logger.info(f"new_themdict1: {print(len(new_themdict1))}")

    logger.info(f"new_katdict1: {print(len(new_katdict1))}")
    """
    """
    logger.info(f"new_kat: {new_kat}")
    logger.info(f"new_them: {new_them}")

    logger.info(f"new_themdict1: {new_themdict1}")
    logger.info(f"new_katdict1: {new_katdict1}")

    logger.info(f"!!!Ende: {input_id}")
    """
    return elements, kat2, new_katdict1, them2, new_themdict1







#####################Tap Infos#######################


@app.callback(
    Output("node-data", "children"), [Input("core_19_cytoscape", "selectedNodeData")]
)
@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              Input('core_19_cytoscape', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        return "Informationen über das Dokument: " + data['label'] + '  ' + data['titel'] + '  ' + data[
            'thema'] + '  ' + data['datum']


@app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
              Input('core_19_cytoscape', 'mouseoverNodeData'))
def displayTapNodeData(data):
    if data:
        return "Dokument: " + data['label'] + '  ' + data['titel'] + '  ' + data['thema'] + '  ' + data['datum']
    if not data:
        return ''


if __name__ == "__main__":
    state_data = fetch_csv(STATE_DATA_URL)
    state_data = clean(state_data)

    plot_cumulative_state(state_data, "index.html")







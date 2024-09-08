import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
import re
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
import warnings
from sklearn.metrics import accuracy_score
from pycaret.classification import get_config
warnings.filterwarnings("ignore")


def load_data(self):
    path = "C:/Users/mayum/OneDrive/Email attachments/Documentos/Universidad/Octavo semestre/Python/Parcial 1/"
    df = pd.read_csv(path + 'train.csv')
    prueba = pd.read_csv(path + "test.csv")
    
    categoricas = [
        'Marital status', 'Nacionality', 'Mother\'s occupation', 'Father\'s occupation',
        'Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Application mode', 'Application order'
    ]

        # Convertir las columnas a tipo categórico en 'df' y 'prueba'
    for k in categoricas:
        if k in df.columns:
                df[k] = df[k].astype('category')
                prueba[k] = prueba[k].astype('category')
    return df, prueba
    
def criterio(df,columns):
    for k in columns:
        df[k]=df[k].map(prueba_kr)
        df["criterio"] = np.sum(df.get(columns),axis = 1)
        df["criterio"] = df.apply(lambda row:1 if row["criterio"]==3 else 0, axis = 1)
        return df    
    
def indicadora(x):
    if x == True:
        return 1
    else:
        return 0
    


def label_tg(self, x):
    if x == "Graduate":
        return 0
    elif x == "Enrolled":
        return 1
    else:
        return 2

def label_tg_inverse(self, x):
    if x == 0:
        return "Graduate"
    elif x == 1:
        return "Enrolled"
    else:
        return "Dropout"

def prueba_kr(self, x):
    return 1 if x <= 0.10 else 0

def nombre(x):
    return "C"+ str(x)


class Mt_flow:
    def __init__(self):
        pass
    def load_data(self):
        df = pd.read_csv("train.csv",sep = ";",decimal=",")
        prueba = pd.read_csv("test.csv",sep = ";",decimal=",")
        
        categoricas = [
            'Marital status', 'Nacionality', 'Mother\'s occupation', 'Father\'s occupation',
            'Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
            'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Application mode', 'Application order'
        ]

            # Convertir las columnas a tipo categórico en 'df' y 'prueba'
        for k in categoricas:
            if k in df.columns:
                df[k] = df[k].astype('O')
                prueba[k] = prueba[k].astype('O')
        return df, prueba
    
    def formato(self, df):
        formato = pd.DataFrame({'Variable': list(df.columns), 'Formato': df.dtypes})
        cuantitativas = list(formato.loc[formato["Formato"] != "category", "Variable"])
        cuantitativas = [x for x in cuantitativas if x not in ["id", "Target"]]
        categoricas = list(formato.loc[formato["Formato"] == "category", "Variable"])
        categoricas = [x for x in categoricas if x not in ["id", "Target"]]
        base_cuadrado = df.get[cuantitativas].copy()
        base_cuadrado["Target"] = df["Target"].copy()
        var_names2, pvalue1, pvalue2, pvalue3 = [], [],[], []

        for k in cuantitativas:
            base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2

            # Prueba de Kruskal sin logaritmo
            mue1 = base_cuadrado.loc[base_cuadrado["Target"] == "Graduate", k+"_2"].to_numpy()
            mue2 = base_cuadrado.loc[base_cuadrado["Target"] == "Enrolled", k+"_2"].to_numpy()
            mue3 = base_cuadrado.loc[base_cuadrado["Target"] == "Dropout", k+"_2"].to_numpy()

            p1 = stats.kruskal(mue1, mue2)[1]
            p2 = stats.kruskal(mue2, mue3)[1]
            p3 = stats.kruskal(mue1, mue3)[1]
            # Guardar p-values y variables
            var_names2.append(k+"_2")
            pvalue1.append(np.round(p1, 2))
            pvalue2.append(np.round(p2, 2))
            pvalue3.append(np.round(p3, 2))        
        pcuadrado1 = pd.DataFrame({'Variable2': var_names2, 'p value1': pvalue1, 'p value3': pvalue3,'p value2': pvalue2})
        pcuadrado1 = criterio(pcuadrado1,["p value1", "p value2", "p value3"])       
        
        ## Interacciones cuantitativas
        
        lista_inter = list(combinations(cuantitativas, 2))
        base_interacciones = df.get[cuantitativas].copy()
        var_interaccion, pv1, pv2, pv3 = [], [],[], []
        base_interacciones["Target"] = df["Target"].copy()
  
        for k in lista_inter:
            base_interacciones[k[0] + "__" + k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]

            # Prueba de Kruskal
            mue1 = base_interacciones.loc[base_interacciones["Target"] == "Graduate", k[0] + "__" + k[1]].to_numpy()
            mue2 = base_interacciones.loc[base_interacciones["Target"] == "Enrolled", k[0] + "__" + k[1]].to_numpy()
            mue3 = base_interacciones.loc[base_interacciones["Target"] == "Dropout", k[0] + "__" + k[1]].to_numpy()
            
            p1 = stats.kruskal(mue1, mue2)[1]
            p2 = stats.kruskal(mue2, mue3)[1]
            p3 = stats.kruskal(mue1, mue3)[1]
            # Guardar p-values y variables

            var_interaccion.append(k[0] + "__" + k[1])
            pv1.append(np.round(p1, 2))
            pv2.append(np.round(p2, 2))
            pv3.append(np.round(p3, 2))

        pxy = pd.DataFrame({'Variable': var_interaccion, 'p value': pv1 , 'p value 2': pv2,'p value 3': pv3})
        pxy = criterio(pxy,["p valor1", "p valor2", "p valor3"])
        # Razones
        
        raz1 = [(x, y) for x in cuantitativas for y in cuantitativas]
        base_razones1 = df.get[cuantitativas].copy()
        base_razones1["Target"] = df["Target"].copy()

        var_raz, pval1,pval2, pval3 = [], [],[], []

        for j in raz1:
            if j[0] != j[1]:
                base_razones1[j[0] + "__coc__" + j[1]] = base_razones1[j[0]] / (base_razones1[j[1]] + 0.01)

                # Prueba de Kruskal
                mue1 = base_razones1.loc[base_razones1["Target"] == "Graduate", j[0] + "__coc__" + j[1]].to_numpy()
                mue2 = base_razones1.loc[base_razones1["Target"] == "Enrolled", j[0] + "__coc__" + j[1]].to_numpy()
                mue3 = base_razones1.loc[base_razones1["Target"] == "Dropout", j[0] + "__coc__" + j[1]].to_numpy()
                p1 = stats.kruskal(mue1, mue2)[1]
                p2 = stats.kruskal(mue2, mue3)[1]
                p3 = stats.kruskal(mue1, mue3)[1]
                # Guardar p-values y variables
                pval1.append(np.round(p1, 2))
                pval2.append(np.round(p2, 2))
                pval3.append(np.round(p3, 2))
                # Guardar valores
                var_raz.append(j[0] + "__coc__" + j[1])
       

        prazones = pd.DataFrame({'Variable': var_raz, 'p value': pval1 , 'p value 2': pval2,'p value 3': pval3 })
        prazones = criterio(prazones,["p value", "p value 2", "p value 3"])
        
        # Interacciones categoricas
        cb = list(combinations(categoricas, 2))
        p_value, modalidades, nombre_var = [], [], []
        base2 = df[categoricas].copy()

        for k in base2.columns:
            base2[k] = base2[k].map(nombre)

        for k in range(len(cb)):
            base2[cb[k][0]]=base2[cb[k][0]]
            base2[cb[k][1]]=base2[cb[k][1]]
            
            base2[cb[k][0] + "__" + cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]
            c1 = pd.DataFrame(pd.crosstab(base2["Target"],base2[cb[k][0]] + "__" + base2[cb[k][1]]))
            p1 = stats.chi2_contingency(c1)[1]
            
        ## Numero por importancia
        mod = len(base2[cb[k][0]+"__"+cb[k][1]].unique())
        nombre_var.append(mod)
        p_value.append(p1)
        pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})
        seleccion1 = list(pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=8),"Variable"])
        sel1 = base2.get(seleccion1)
        contador = 0
        for k in sel1:
            if contador==0:
                lb1 = pd.get_dummies(sel1[k],drop_first=True)
                lb1.columns = [k + "_" + x for x in lb1.columns]
            else:
                lb2 = pd.get_dummies(sel1[k],drop_first=True)
                lb2.columns = [k + "_" + x for x in lb2.columns]
                lb1 = pd.concat([lb1,lb2],axis=1)
            contador = contador + 1
        for k in lb1.columns:
            lb1[k] = lb1[k].map(indicadora)

            lb1["Target"] = df["Target"].copy()
     #Cuantitativa categorica
        cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]
        v1, v2, pvalores_min, pvalores_max  = [], [], [], []

        for j in cat_cuanti:
            k1 = j[0]
            k2 = j[1]

            g1 = pd.get_dummies(df[k2])
            lt1 = list(g1.columns)

            for k in lt1:
                g1[k] = g1[k] * df[k1]

            g1["Target"] = df["Target"].copy()

            pvalues_c = []
            for y in lt1:
                mue1 = g1.loc[g1["Target"]=="Graduate",y].to_numpy()
                mue2 = g1.loc[g1["Target"]=="Dropout",y].to_numpy()
                mue3 = g1.loc[g1["Target"]=="Enrolled",y].to_numpy()

                try:
                    pval = (stats.kruskal(mue1,mue2)[1]<=0.20)
                except ValueError:
                    pval = 0
                try:
                    pval2 = (stats.kruskal(mue3,mue2)[1]<=0.20)
                except ValueError:
                    pval2 = 0
                try:
                    pval3 = (stats.kruskal(mue1,mue3)[1]<=0.20)
                except ValueError:
                    pval3 = 0
                pvalues_c.append(np.rpund(np.sum(np.array([pval, pval2, pval3]))))
            min_ = np.min(pvalues_c) # Se revisa si alguna de las categorías no es significativa
            max_ = np.max(pvalues_c) # Se revisa si alguna de las categorías es significativa
            v1.append(k1) # nombre de la variable 1
            v2.append(k2) # nombre de la variable 2
            pvalores_min.append(np.round(min_,2))
            pvalores_max.append(np.round(max_,2))
            pc2 = pd.DataFrame({'Cuantitativa':v1,'Categórica':v2,'p value':pvalores_min, 'p value max':pvalores_max})
            v1 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Cuantitativa"])
            v2 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Categórica"])
            for j in range(len(v1)):

                if j==0:
                    g1 = pd.get_dummies(df[v2[j]],drop_first=True)
                    lt1 = list(g1.columns)
                    for k in lt1:
                        g1[k] = g1[k] * df[v1[j]]
                    g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                else:
                    g2 = pd.get_dummies(df[v2[j]],drop_first=True)
                    lt1 = list(g2.columns)
                    for k in lt1:
                        g2[k] = g2[k] * df[v1[j]]
                    g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                    g1 = pd.concat([g1,g2],axis=1)

                    g1["Target"] = df["Target"].copy()
                
                # Seleccion de variables

                var_cuad = list(pcuadrado1["Variable2"])
                base_modelo1 = base_cuadrado.get(var_cuad+["Target"])
                base_modelo1["Target"] = base_modelo1["Target"].map(label_tg)
                cov = list(base_modelo1.columns)
                cov = [x for x in cov if x not in ["Target"]]

                X1 = base_modelo1.get(cov)
                y1 = base_modelo1.get(["Target"])

                modelo1 = XGBClassifier()
                modelo1 = modelo1.fit(X1,y1)

                importancias = modelo1.feature_importances_
                imp1 = pd.DataFrame({'Variable':X1.columns,'Importancia':importancias})
                imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
                imp1 = imp1.sort_values(["Importancia"],ascending=False)
                imp1.index = range(imp1.shape[0])
        #Interacciones cuanti
                var_int = list(pxy["Variable"])
                base_modelo2 = base_interacciones.get(var_int+["Target"])
                base_modelo2["Target"] = base_modelo2["Target"].map(label_tg)
                cov = list(base_modelo2.columns)
                cov = [x for x in cov if x not in ["Target"]]

                X2 = base_modelo2.get(cov)
                y2 = base_modelo2.get(["Target"])

                modelo2 = XGBClassifier()
                modelo2 = modelo2.fit(X2,y2)

                importancias = modelo2.feature_importances_
                imp2 = pd.DataFrame({'Variable':X2.columns,'Importancia':importancias})
                imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
                imp2 = imp2.sort_values(["Importancia"],ascending=False)
                imp2.index = range(imp2.shape[0])
                
            #Razones
                var_raz = list(prazones["Variable"])
                base_modelo3 = base_razones1.get(var_raz+["Target"])
                base_modelo3["Target"] = base_modelo3["Target"].map(label_tg)
                cov = list(base_modelo3.columns)
                cov = [x for x in cov if x not in ["Target"]]

                X3 = base_modelo3.get(cov)
                y3 = base_modelo3.get(["Target"])

                modelo3 = XGBClassifier()
                modelo3 = modelo3.fit(X3,y3)

                importancias = modelo3.feature_importances_
                imp3 = pd.DataFrame({'Variable':X3.columns,'Importancia':importancias})
                imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
                imp3 = imp3.sort_values(["Importancia"],ascending=False)
                imp3.index = range(imp3.shape[0])
            #Categoricas
                lb1["Target"] = lb1["Target"].map(label_tg)
                cov = list(lb1.columns)
                cov = [x for x in cov if x not in ["Target"]]

                X4 = lb1.get(cov)
                y4 = lb1.get(["Target"])

                modelo4 = XGBClassifier()
                modelo4 = modelo4.fit(X4,y4)

                importancias = modelo4.feature_importances_
                imp4 = pd.DataFrame({'Variable':X4.columns,'Importancia':importancias})
                imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
                imp4 = imp4.sort_values(["Importancia"],ascending=False)
                imp4.index = range(imp4.shape[0])
        # Cuanti - categ
                g1["Target"] = g1["Target"].map(label_tg)
                cov = list(g1.columns)
                cov = [x for x in cov if x not in ["Target"]]

                X5 = g1.get(cov)
                y5 = g1.get(["Target"])

                modelo5 = XGBClassifier()
                modelo5 = modelo5.fit(X5,y5)

                importancias = modelo5.feature_importances_
                imp5 = pd.DataFrame({'Variable':X5.columns,'Importancia':importancias})
                imp5["Importancia"] = imp5["Importancia"] * 100 / np.sum(imp5["Importancia"])
                imp5 = imp5.sort_values(["Importancia"],ascending=False)
                imp5.index = range(imp5.shape[0])

        # variables importantes
                c2 = list(imp1.iloc[0:3,0]) # Variables al cuadrado
                cxy = list(imp2.iloc[0:8,0]) # Interacciones cuantitativas
                razxy = list(imp3.iloc[0:8,0]) # Razones
                catxy = list(imp4.iloc[0:3,0]) # Interacciones categóricas
                cuactxy = list(imp5.iloc[0:8,0]) # Interacción cuantitativa y categórica

                # Variables cuantitativas (Activar D1)
                D1 = df.get(cuantitativas).copy()

                # Variables categóricas
                D2 = df.get(categoricas).copy()
                for k in categoricas:
                    D2[k] = D2[k].map(nombre)
                    D4 = D2.copy()

                # Variables al cuadrado (Activar D1)
                cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
                cuadrado = [x[0] for x in cuadrado]

                for k in cuadrado:
                    D1[k+"_2"] = D1[k] ** 2

                # Interacciones cuantitativas (Activar D1)
                result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]

                for k in result:
                    D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]

                # Razones
                result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
                for k in result2:
                    k2 = k[0]
                    D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)

                    # Interacciones categóricas
                    result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
                for k in result3:
                    D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]

                    # Interacción cuantitativa vs categórica
                    D5 = df.copy()
                    result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
                    contador = 0
                for k in result4:
                    col1, col2 = k[1], k[0] # categórica, cuantitativa
                if contador == 0:
                    D51 = pd.get_dummies(D5[col1],drop_first=True)
                    for j in D51.columns:
                        D51[j] = D51[j] * D5[col2]
                        D51.columns = [col2+""+col1+""+ str(x) for x in D51.columns]
                else:
                    D52 = pd.get_dummies(D5[col1],drop_first=True)
                    for j in D52.columns:
                        D52[j] = D52[j] * D5[col2]
                        D52.columns = [col2+""+col1+""+ str(x) for x in D52.columns]
                        D51 = pd.concat([D51,D52],axis=1)
                        contador = contador + 1
                        B1 = pd.concat([D1,D4],axis=1)
                        base_modelo = pd.concat([B1,D51],axis=1)
                        base_modelo["Target"] = df["Target"].copy()
                        base_modelo["Target"] = base_modelo["Target"].map(label_tg)
                        return base_modelo, cuantitativas, categoricas, cuadrado, result,  result2, result3, result4
                    
    def Automl(self,base_modelo, modelo):
        formatos = pd.DataFrame(base_modelo.dtypes).reset_index()
        formatos.columns = ["Variable","Formato"]
        cuantitativas_bm = list(formatos.loc[formatos["Formato"]!="object",]["Variable"])
        categoricas_bm = list(formatos.loc[formatos["Formato"]=="object",]["Variable"])
        cuantitativas_bm = [x for x in cuantitativas_bm if x not in ["Target"]]
        categoricas_bm = [x for x in categoricas_bm if x not in ["Target"]]
        exp_clf101 = setup(data=base_modelo,target='Target',session_id=123,train_size=0.7,numeric_features = cuantitativas_bm,categorical_features = categoricas_bm)
        numeric_features = cuantitativas_bm,categoricas_features = categoricas_bm
    
        if modelo == 1:
            dt = create_model("lightgbm")
            param_grid_bayesian = {'n_estimators': [50,100,200],'max_depth': [3,5,7],'min_child_samples': [50,150,200]}
            tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
        elif modelo==2:
            turned_dt = create_model("XGBClassifier")
        elif modelo ==3 :
            dt = create_model("gbc")
            param_grid_bayesian = {'n_estimators': [50,100,200],'max_depth': [3,5,7],'min_child_samples': [50,150,200]}
            tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
        predictions_test = predict_model(tuned_dt,data=exp_clf101.get_config('X_test'))
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train')) 
        y_train = get_config('y_train')
        y_test = get_config('y_test')
        acc_train = accuracy_score(y_train,predictions_train["prediction_label"])
        acc_test = accuracy_score(y_train,predictions_test["prediction_label"])
        final_dt = finalize_model(tuned_dt)
        return final_dt,acc_test,acc_train
    
    def final(self, ingenieria,modelo):
    
        try:
            df, prueba =self.load.data()
            if ingenieria == True:
                base_modelo, cuantitativas, categoricas, cuadrado, result,  result2, result3, result4 = self.formato(df)
            else:          
                 base_modelo = df 
                   
            final_dt,acc_test,acc_train = self.Automl(base_modelo, modelo)
          
            return{'success':True, 'Accuracy train':acc_train,'Accuracy test':acc_test}
        except Exception as e:
            return {'success':False,'message':str(e)}
    
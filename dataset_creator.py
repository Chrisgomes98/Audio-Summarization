import pandas as pd
import numpy as np
import librosa, librosa.display #librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

class dataset_creator:
    def __init__(self,path,prefix,afe,label_classifier,label):
        self.path=path
        self.prefix=prefix
        self.afe=afe
        self.label_classifier=label_classifier
        self.label=label
    def generate_dataset(self):
        dataset=pd.read_csv(self.path)
        names=dataset['Name']
        labels=dataset['Label']
        signaln = int(661794/6)
        for j in range(len(names)):
            df={'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], 'Zero Crossing Rate':[], 'Energy':[], 'RMSE':[], 'Label':[]}
            name="audio\\"+names[j]+".wav"
            print(name)
            signal, sample_rate = librosa.load(name, sr=22050)#plot signal
            label=labels[j]
            df_feature_row={}
            for i in range(signaln*2,len(signal),signaln):
                feature_row = self.afe.get_feature_row(signal[i:i+signaln],sample_rate)
                f_row={}
                for key in feature_row.keys():
                    f_row[key]=sum(feature_row[key][0])/len(feature_row[key][0])
               
                for key in f_row.keys():
                    df[key].append(f_row[key])

                if(self.label=="Defined"):
                    df['Label'].append(label)
                else:
                    df['Label'].append(list(self.label_classifier.detect(list(f_row.values())))[0])

                self.afe.reset()
            df = pd.DataFrame(df)
            df.to_csv(self.prefix+str(j)+'.csv')


        l=[]
        for i in range(len(names)):
            df=pd.read_csv(self.prefix+str(i)+".csv")
            df = df.iloc[: , 1:]
            l.append(df)
        df_master = pd.concat(l)
        df_master.to_csv(self.prefix+"dataset.csv",index=False)
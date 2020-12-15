# ニューラルネットワーク２つ
import nn as NN 
# データローダー
import loader as L 

import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os 
import cv2 
import numpy as np
import torchvision.transforms as transforms

if __name__ == "__main__":
  ######------hyperParam-----#####
    epoch       = 40
    imgSize     = 160
    Neuron      = 320
    hidden1     = 320
    hidden2     = 160
    lr          = 0.000005

    batchSize   = 1
    ptName      = "models/nn1.pt"
    lossPng     = "lossImg/loss1.png"
    accPng      = "accImg/acc1.png"
    
    ######------モデルの設定(損失関数も変える)-----#####
    # model       = NN.Net_log_softmax(num=6,inputSize=imgSize,Neuron=Neuron)
    model       = NN.CNN(num=6,inputSize=imgSize,hidden1=hidden1,hidden2=hidden2)

    optimizer   = torch.optim.Adam(params=model.parameters(),lr=lr)

    ######------個別推論用ディレクトリ設定-----#####
    andoDir     = "../ReadOnlyDataSet2/Resources/test/ando"
    higashiDir  = "../ReadOnlyDataSet2/Resources/test/higashi"
    kataokaDir  = "../ReadOnlyDataSet2/Resources/test/kataoka"
    kodamaDir   = "../ReadOnlyDataSet2/Resources/test/kodama"
    masudaDir   = "../ReadOnlyDataSet2/Resources/test/masuda"
    suetomoDir  = "../ReadOnlyDataSet2/Resources/test/suetomo"

    ######------学習用ディレクトリ設定-----#####
    dirImgTrainPath = "new-dataset"
    dirImgTestPath  = "../ReadOnlyDataSet2/Resources" + "/test/"

    # 学習結果保存用
    R = {
        "trainLoss":[],
        "testLoss":[],
        "testAcc":[],}
    A = {
        "ando":[],
        "higashi":[],
        "kataoka":[],
        "kodama":[],
        "masuda":[],
        "suetomo":[],}



    DL = L.MyDataLoader(trainRootDir=dirImgTrainPath,testRootDir=dirImgTestPath,imgSize=imgSize,batchSize=batchSize,num_workers=0)
    loader = DL.getDataLoaders()


    for e in range(epoch):
        for data in tqdm(loader["train"]):
            inputs,label = data 
            optimizer.zero_grad()
            output = model(inputs)
            ######------損失関数-----#####
            loss = F.nll_loss(output,label)
            
            loss.backward()
            optimizer.step()
        R["trainLoss"].append(loss)

        # 学習停止
        model.eval()
        testLoss = 0
        total = 0
        correct = 0
        andoAcc = 0
        higashiAcc = 0
        kataokaAcc = 0
        kodamaAcc = 0
        masudaAcc = 0
        suetomoAcc = 0

        with torch.no_grad():
            for data in tqdm(loader["test"]):
                inputs,label = data  
                #   予測値
                p   =   model.forward(inputs).exp()
                p   =   p.to('cpu').detach().numpy().copy()
                p   =   p[0]
                
                #   すべての中で最も大きい値
                p       =   p.argmax()
                #   予測値と正解ラベルが同じとき
                if p == label:
                    if p == 0:
                        if andoAcc == 200:
                            break
                        andoAcc     += 1
                    if p == 1:
                        if higashiAcc == 200:
                            break
                        higashiAcc  += 1
                    if p == 2:
                        if kataokaAcc == 200:
                            break
                        kataokaAcc  += 1
                    if p == 3:
                        if kodamaAcc == 200:
                            break
                        kodamaAcc   += 1
                    if p == 4:
                        if masudaAcc == 200:
                            break
                        masudaAcc   += 1
                    if p == 5:
                        if suetomoAcc == 200:
                            break
                        suetomoAcc  += 1
            andoAcc     =   andoAcc/200*100
            higashiAcc  =   higashiAcc/200*100
            kataokaAcc  =   kataokaAcc/200*100
            kodamaAcc   =   kodamaAcc/200*100
            masudaAcc   =   masudaAcc/200*100
            suetomoAcc  =   suetomoAcc/200*100
            

        A["ando"].append(andoAcc)
        A["higashi"].append(higashiAcc)
        A["kataoka"].append(kataokaAcc)
        A["kodama"].append(kodamaAcc)
        A["masuda"].append(masudaAcc)
        A["suetomo"].append(suetomoAcc)


    plt.figure()
    plt.plot(range(1, epoch+1), R['trainLoss'], label='trainLoss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(lossPng)
    plt.figure()
    plt.plot(range(1,epoch+1),A["ando"],label="ando")
    plt.plot(range(1,epoch+1),A["higashi"],label="higashi")
    plt.plot(range(1,epoch+1),A["kataoka"],label="kataoka")
    plt.plot(range(1,epoch+1),A["kodama"],label="kodama")
    plt.plot(range(1,epoch+1),A["masuda"],label="masuda")
    plt.plot(range(1,epoch+1),A["suetomo"],label="suetomo")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(accPng)
    # Save
    torch.save(model.state_dict(),ptName)
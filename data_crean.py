import torch
import cv2
import sys 
import numpy as np 
import torchvision.transforms as transforms
import os 

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)
        
    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class Pre:
    dataMax = 10
    imgDir = "DataSet/"
    def __init__(self):
        pass 

    def setMax(self,data_max):
        Pre.dataMax = data_max
        pass 

    def setDir(self,imgDir):
        Pre.imgDir  = imgDir

    def test(self,name):
        self.model = NN.Net(num=6,inputSize=160,Neuron=320)
        PATH = "models/nn1.pt"
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

        cnt     = 0
        ando    = 0
        higashi = 0
        kataoka = 0
        kodama  = 0
        masuda  = 0
        suetomo = 0
        mDir    = Pre.imgDir + name + "/"

        while True:
            if cnt == Pre.dataMax:
                print("")
                break

            img = cv2.imread(mDir+str(cnt)+".jpg")
            p = self.maesyori(img,160)

            if name == "ando":
                if p == 0:
                    ando += 1
                    sys.stdout.write("\r {} {}/{}".format(name,ando,Pre.dataMax))
            if name == "higashi":
                if p == 1:
                    higashi += 1
                    sys.stdout.write("\r {} {}/{}".format(name,higashi,Pre.dataMax))
            if name == "kataoka":
                if p == 2:
                    kataoka += 1
                    sys.stdout.write("\r {} {}/{}".format(name,kataoka,Pre.dataMax))
            if name == "kodama":
                if p == 3:
                    kodama += 1
                    sys.stdout.write("\r {} {}/{}".format(name,kodama,Pre.dataMax))
            if name == "masuda":
                if p == 4:
                    masuda += 1
                    sys.stdout.write("\r {} {}/{}".format(name,masuda,Pre.dataMax))
            if name == "suetomo":
                if p == 5:
                    suetomo += 1
                    sys.stdout.write("\r {} {}/{}".format(name,suetomo,Pre.dataMax))
            cnt += 1

        pass 

    def dataCrean(self,name):
        mDir = Pre.imgDir + name + "/"
        tmpDir = mDir + "tmp" + "/"
        try:
            os.makedirs(tmpDir)
        except FileExistsError:
            pass 

        filelist = []
        for f in os.listdir(mDir):
            if os.path.isfile(os.path.join(mDir,f)):
                filelist.append(mDir+f)

        cnt = 0

        # tmpフォルダに保存
        for path in filelist:
            os.rename(path,mDir+"tmp/"+str(cnt)+".jpg")
            cnt += 1

        # 元のフォルダに移動
        cnt = 0
        for f in os.listdir(mDir+"tmp/"):
            os.rename(mDir+"tmp/"+f,mDir+str(cnt)+".jpg")
            cnt += 1

        # tmpフォルダの削除
        os.rmdir(mDir+"tmp/")

        print(mDir+"に保存されているファイル名を揃えました。")
        pass 

    def maesyori(self,imgCV,imgSize):
        # チャンネル数を１
        imgGray = cv2.cvtColor(imgCV,cv2.COLOR_BGR2GRAY)
        

        #リサイズ
        img = cv2.resize(imgGray,(imgSize,imgSize))
        

        # リシェイプ
        img = np.reshape(img,(1,imgSize,imgSize))

        # transpose h,c,w
        img = np.transpose(img,(1,2,0))

        # ToTensor 正規化される
        img = img.astype(np.uint8)
        mInput = transforms.ToTensor()(img)  
        #print(mInput)

        #推論
        #print(mInput.size())
        output = self.model(mInput[0])

        #予測値
        p = self.model.forward(mInput)

        #予測値出力
        # print(p)
        # print(p.argmax())
        # print(type(p))

        # 戻り値は予測値
        return p.argmax()

if __name__ == "__main__":
    pre   = Pre()
    pre.setMax(100)
    pre.setDir("new-dataset/")

    # データをそろえる
    pre.dataCrean("ando")
    pre.dataCrean("higashi")
    pre.dataCrean("kataoka")
    pre.dataCrean("kodama")
    pre.dataCrean("masuda")
    pre.dataCrean("suetomo")

    # 推論結果
    # pre.test("ando")
    # pre.test("higashi")
    # pre.test("kataoka")
    # pre.test("kodama")
    # pre.test("masuda")
    # pre.test("suetomo")
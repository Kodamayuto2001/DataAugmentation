import torch 
import cv2 
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data=[1,2,3,4,5,6],label=[0,1,0,1,0,1],transform=None):
        self.transform = transform
        self.data = data
        self.label = label

        self.datanum = 6
        pass 

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data,out_label

class Dataset2(torch.utils.data.Dataset):
    def __init__(self,data,label,data_len,transform=None):
        self.data       = data 
        self.label      = label
        self.data_len   = data_len 
        self.transform  = transform 
        pass 

    def __len__(self):
        return self.data_len 

    def __getitem__(self,idx):
        out_data    = self.data 
        out_label   = self.label 

        if self.transform:
            out_data = self.transform(out_data)

        return out_data,out_label


if __name__ == "__main__":
    img = cv2.imread("new-dataset/ando/0.jpg")
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgSize = 160
    img = cv2.resize(img,(imgSize,imgSize))

    img = np.reshape(img,(1,imgSize,imgSize))

    # print(img.shape)    # (1, 160, 160)

    img = [img]
    img = np.array(img)

    # print(img.shape)    # (1, 1, 160, 160)
    img = img.tolist()

    dataset2 = Dataset2(
        data    =img,
        label   =0,
        data_len=1,
    )
    print(len(dataset2))
    print(dataset2[0])



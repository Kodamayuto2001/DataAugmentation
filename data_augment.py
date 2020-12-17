import albumentations as A
import numpy as np 
import random 
import cv2 
import os 

class DataAugmentation:
    MIN_TABLE       = 50
    MAX_TABLE       = 205 
    DIFF_TABLE      = MAX_TABLE - MIN_TABLE
    GANMA1          = 0.75 
    GANMA2          = 1.5 
    AVERAGE_SQUARE  = (10,10)
    MEAN            = 0
    SIGMA           = 15
    ORGDIR          = ""
    SAVEDIR         = "data-augment-set" + "/"
    def __init__(self):
        pass 

    def setSaveDir(self,saveDir):
        self.SAVEDIR = saveDir 

    def setOriginalDataSet(self,orgDir):
        self.ORGDIR = orgDir

    def save(self):
        fileList    = os.listdir(self.ORGDIR)
        for cnt,fname in enumerate(fileList):
            self.__saveNewFile(self.ORGDIR+fname    ,cnt)

    def __saveNewFile(self,img_path,cnt):
        try:
            os.makedirs(self.SAVEDIR)
        except FileExistsError:
            pass 

        # 画像読み込み
        img = cv2.imread(img_path)
        
        # 画像保存
        # cv2.imwrite(self.SAVEDIR+"original_"        +str(cnt)   +".jpg" ,img)
        # cv2.imwrite(self.SAVEDIR+"hicontrast_"      +str(cnt)   +".jpg" ,self.__hi_contrast(img))
        # cv2.imwrite(self.SAVEDIR+"locontrast_"      +str(cnt)   +".jpg" ,self.__lo_contrast(img))
        # cv2.imwrite(self.SAVEDIR+"ganma1_"          +str(cnt)   +".jpg" ,self.__ganma_1(img))
        # cv2.imwrite(self.SAVEDIR+"ganma2_"          +str(cnt)   +".jpg" ,self.__ganma_2(img))
        # cv2.imwrite(self.SAVEDIR+"blur_"            +str(cnt)   +".jpg" ,self.__blur(img))
        # cv2.imwrite(self.SAVEDIR+"gauss_"           +str(cnt)   +".jpg" ,self.__gauss(img))
        # cv2.imwrite(self.SAVEDIR+"spnoise_"         +str(cnt)   +".jpg" ,self.__sp_noise(img,0.05))
        # cv2.imwrite(self.SAVEDIR+"multiplicative_"  +str(cnt)   +".jpg" ,self.__multiplicative_noise(img))
        # cv2.imwrite(self.SAVEDIR+"motionBlur_"      +str(cnt)   +".jpg" ,self.__motion_blur(img))
        # cv2.imwrite(self.SAVEDIR+"glassBlur_"       +str(cnt)   +".jpg" ,self.__glass_blur(img))
        # cv2.imwrite(self.SAVEDIR+"jpegCompression_" +str(cnt)   +".jpg" ,self.__jpeg_compression(img))
        # cv2.imwrite(self.SAVEDIR+"isoNoise_"        +str(cnt)   +".jpg" ,self.__iso_noise(img))
        cv2.imwrite(self.SAVEDIR+"downScale_"       +str(cnt)   +)


    def __hi_contrast(self,img):
        LUT_HC  = np.arange(256,dtype="uint8")
        for i in range(0,self.MIN_TABLE):
            LUT_HC[i] = 0
        for i in range(self.MIN_TABLE,self.MAX_TABLE):
            LUT_HC[i] = 255*(i-self.MIN_TABLE)/self.DIFF_TABLE
        for i in range(self.MAX_TABLE,255):
            LUT_HC[i] = 255
        hi_cont_img = cv2.LUT(img,LUT_HC)
        
        return hi_cont_img
        
    def __lo_contrast(self,img):
        LUT_LC  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_LC[i] = self.MIN_TABLE + i * (self.DIFF_TABLE) / 255
        lo_cont_img = cv2.LUT(img,LUT_LC)

        return lo_cont_img

    def __ganma_1(self,img):
        LUT_G1  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_G1[i] = 255 * pow(float(i)/255,1.0/self.GANMA1)
        ganma1_img = cv2.LUT(img,LUT_G1)

        return ganma1_img

    def __ganma_2(self,img):
        LUT_G2  = np.arange(256,dtype="uint8")
        for i in range(256):
            LUT_G2[i] = 255 * pow(float(i)/255,1.0/self.GANMA2)
        ganma2_img = cv2.LUT(img,LUT_G2)

        return ganma2_img

    def __blur(self,img):
        return cv2.blur(img,self.AVERAGE_SQUARE)

    def __gauss(self,img):
        row,col,ch = img.shape 
        gauss = np.random.normal(self.MEAN,self.SIGMA,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss_img = img + gauss

        return gauss_img

    def __sp_noise(self,img,prob):
        sp_noise_img  = np.zeros(img.shape,np.uint8)
        thres   = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    sp_noise_img[i][j] = 0
                elif rdn > thres:
                    sp_noise_img[i][j] = 255 
                else:
                    sp_noise_img[i][j] = img[i][j]
        
        return sp_noise_img

    def __multiplicative_noise(self,img):
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=(0.3,3.5),elementwise=False),
            A.MultiplicativeNoise(multiplier=(2.3,5.5),elementwise=True),
            A.MultiplicativeNoise(multiplier=(0.3,1.5),elementwise=False)
        ])
        transformed         = transform(image=img)
        transformed_image   = transformed["image"]

        return transformed_image

    def __motion_blur(self,img):
        transform = A.Compose([
            A.MotionBlur(blur_limit=(29,35))
        ])
        transformed         = transform(image=img)
        transformed_image   = transformed["image"]

        return transformed_image

    def __glass_blur(self,img):
        transform = A.Compose([
            A.GlassBlur(sigma=2.0,max_delta=3),
            A.GlassBlur(sigma=4.0,max_delta=1),
            A.GlassBlur(sigma=1.0,max_delta=5),
        ])
        transformed         =   transform(image=img)
        transformed_image   =   transformed["image"]

        return transformed_image

    def __jpeg_compression(self,img):
        transform = A.Compose([
            A.JpegCompression(quality_lower=5,quality_upper=10),
            A.JpegCompression(quality_lower=3,quality_upper=5),
            A.JpegCompression(quality_lower=2,quality_upper=4),
        ])
        transformed         =   transform(image=img)
        transformed_image   =   transformed["image"]

        return transformed_image

    def __iso_noise(self,img):
        transform = A.Compose([
            A.ISONoise(color_shift=(0.01,0.1),intensity=(1.5,2.3))
        ])
        transformed         =   transform(image=img)
        transformed_image   =   transformed["image"]

        return transformed_image

    def __down_scale(self,img):
        transform = A.Compose([
            A.Downscale(scale_min=0.3,scale_max=0.8)
        ])
        transformed         =   transform(image=img)
        transformed_image   =   transformed["image"]

        return transformed_image
    

if __name__ == "__main__":
    name            = [
        "ando",
        "higashi",
        "kataoka",
        "kodama",
        "masuda",
        "suetomo",
    ]
    org_dir_name    = "dataset"
    new_dir_name    = "new-dataset"

    class_instance  = []


    for i,_ in enumerate(name):
        # インスタンス化
        class_instance.append(DataAugmentation())
        
        # オリジナル画像のPATH指定
        class_instance[i].setOriginalDataSet(
            org_dir_name    +   "/" +   name[i] +   "/"
        )

        # 拡張データのPATH指定
        class_instance[i].setSaveDir(
            new_dir_name    +   "/" +   name[i] +   "/"
        )

        # 拡張データを保存
        class_instance[i].save()

















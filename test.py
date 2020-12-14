import data_augment as da 

if __name__ == "__main__":
    name            = [
        "ando",
        "higashi",
        "kataoka",
        "kodama",
        "masuda",
        "suetomo",
    ]
    org_dir_name    = "dataset-original"
    new_dir_name    = "new-dataset"

    class_instance  = []


    for i,_ in enumerate(name):
        # インスタンス化
        class_instance.append(da.DataAugmentation())
        
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

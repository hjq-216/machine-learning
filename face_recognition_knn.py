import os
from sklearn import neighbors
import os.path
import  pickle
from PIL import Image,ImageDraw
import face_recognition as fr
from face_recognition.face_recognition_cli import image_files_in_folder
#定义一个训练模型的函数
def train(train_dir,model_save_path='face_recog_model.clf',n_neighbors=3,knn_algo='ball tree'):
    #生成训练集
    x=[]
    y=[]
    #遍历人 class_dir 是人名
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue #结束当前循环

        #遍历这个人的图片 img——path是具体的图片
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image=fr.load_image_file(img_path)
            boxes=fr.face_locations(image)
            #对于当前的图片，增加编码至训练集
            encoding=fr.face_encodings(image,known_face_locations=boxes)[0]
            x.append(encoding) #返回128纬度的向量
            y.append(class_dir)

    #决定k值
    if n_neighbors is None:
        n_neighbors=1

    #创建并且训练分类器
    knn_clf=neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(x,y)

    #保存训练好的分类器
    if model_save_path is not None:
        with open(model_save_path,'wb') as f:
            pickle.dump(knn_clf,f)

    return knn_clf


#使用模型预测
def predict(img_path,knn_clf=None,model_path=None,distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("必须提供knn分类器，可选方式为knn——clf或model——path")

    #加载训练好的knn模型
    if knn_clf is None:
        with open(model_path,"rb") as f:
            knn_clf=pickle.load(f)

    #加载图片，发现人脸的位置
    x_img=fr.load_image_file(img_path)
    x_face_locations=fr.face_locations(x_img)

    #对测试图片中的人脸编码，返回128特征向量
    encodings=fr.face_encodings(x_img,known_face_locations=x_face_locations)

    #利用knn model找出最匹配的人脸
    closet_distance=knn_clf.kneighbors(encodings,n_neighbors=1)
    are_matches=[closet_distance[0][i][0]<=distance_threshold
                 for i in range(len(x_face_locations))]
    #预测类别
    return  [(pred,loc) if rec else("unknown",loc)
            for pred,loc ,rec in zip(knn_clf.predict(encodings),x_face_locations,are_matches)]


#人脸识别可视化函数
def show_names_on_image(img_path,predictions):
    pil_image=Image.open(img_path).convert("RGB")
    draw=ImageDraw.Draw(pil_image)

    for name,(top,right,bottom,left) in predictions:
        #用pillow模块画出人脸边界盒子
        draw.rectangle(((left,top),(right,bottom)),outline=(255,0,255))

        #pillow里可能生成非UTF-8格式，所以做如下转换
        name=name.encode("UTF-8")
        name=name.decode("ascii")

        #在人脸下写下标签
        text_width,text_heighth=draw.textsize(name)
        draw.rectangle(((left,bottom-text_heighth-10),(right,bottom)),fill=(255,0,255),outline=(255,0,255))
        draw.text((left+6,bottom-text_heighth-5),name,fill=(255,255,255))


    del draw
    pil_image.show()

#main函数
if __name__=="__main__":
    #1 训练knn分类器，先保存后使用
    print("traning")
    train("EXAMPLES/train",model_save_path="face_recog_model.clf",n_neighbors=1)
    print("trained")

    #2利用模型预测
    for image_files in os.listdir("EXAMPLES/test2"):
        full_file_path=os.path.join("EXAMPLES/test2",image_files)
        print("在%s中寻找人脸"%format(image_files))

        #利用分类器找出所有的人脸，要么传递一个文件名，要么传一个模型实例
        predictions=predict(full_file_path,model_path="face_recog_model.clf")

        #打印结果
        for name in predictions:
            print("发现%s"%format(name))

        show_names_on_image(os.path.join("EXAMPLES/test2",image_files),predictions)
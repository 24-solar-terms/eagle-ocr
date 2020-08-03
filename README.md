# EAGLE-OCR
OCR算法服务

部署ocr服务需要修改的位置主要如下:

ocr算法放在eagle-ocr/eagle_ocr_model内
1. 将ocr项目放入web后可能引起包导入错误, 将错误的引入全部改为相对引入
2. eagle_ocr_model/crnn/crnn.py 修改crnn.py文件, 文件头部添加搜索的绝对路径例如: sys.path.append('/home/guyu.gy/eagle-ocr/eagle_ocr_model/crnn'), 部署ocr项目时'/home/guyu.gy/eagle-ocr/eagle_ocr_model/'为在实际服务器位置
3. eagle_ocr_model/crnn/crnn.py 修改crnn.py文件, 将crnnSource()函数中path路径, 改为绝对路径, 例如: path = '/home/guyu.gy/eagle-ocr/eagle_ocr_model/crnn/samples/model_acc97.pth', 同理, 部署ocr项目时'/home/guyu.gy/eagle-ocr/eagle_ocr_model/'为在实际服务器ocr项目位置
4. eagle_ocr_model/ctpn/ctpn/model.py 修改model.py文件, 将load_tf_model()函数中ckpt_path的路径修改为checkpoints文件夹的绝对路径, 例如:  我的checkpoints文件夹存放的绝对路径为/home/guyu.gy/eagle-ocr/eagle_ocr_model/ctpn/checkpoints
5. eagle_ocr_model/ocr/model.py 修改model.py文件, 文件头部添加搜索的绝对路径例如: sys.path.append('/home/guyu.gy/eagle-ocr/eagle_ocr_model/ocr') 
6. eagle_ocr_model/ocr/model.py 修改model.py文件, 把modelPath修改为存放权重模型的绝对路径

7.如果使用pytorch模型, 并且必须保证服务器GPU内存足够

8.项目部署的时候使用Production配置

9.版本控制中没有上传之前需要下载的模型权重, 配置时需要自己加上

10.需要执行以下eaglc_ocr_model/ctpn/lib/utils下的make.sh文件 命令 sh make.sh



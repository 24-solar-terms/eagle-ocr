# EAGLE-OCR
OCR算法服务

程序入口app.py文件，接收POST请求，根据请求参数，读取图片，选择相应的模型，调用算法接口

算法接口汇总在eagle_ocr_model/model.py文件内的model函数

eagle_ocr_model文件夹下各个算法模块（检测或识别）在各自对应命名的文件夹下（检测：ctpn，east，PSENet，识别：crnn）

## 注意事项：
1. 项目不含模型权重，对应模型的权重可以在百度云地址下载
https://pan.baidu.com/s/13H5SB5h0GdnPeDXmwGo7-Q
提取码：rvf2

2. crnn的模型权重model_acc97.pth放置在eagle-ocr/eagle_ocr_model/crnn/samples文件夹下

3. ctpn的模型权重checkpoints文件夹放置在eagle-ocr/eagle_ocr_model/ctpn文件夹下

4. east的模型权重checkpoints文件夹放置在eagle-ocr/eagle_ocr_model/east文件夹下

5. PSENet的模型权重checkpoints文件夹放置在eagle-ocr/eagle_ocr_model/PSENet文件夹下

6. 需要执行eagle-ocr/eaglc_ocr_model/ctpn/lib/utils下的make.sh文件 (GPU环境)或make-for-cpu.sh (CPU环境) 命令 sh make.sh或sh make-for-cpu.sh



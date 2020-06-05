from flask import Flask, request, jsonify
from eagle_ocr_model import model
from PIL import Image
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def exe_ocr():
    """
    OCR算法服务接口
    :return: json数据, 字典结构, 识别结果, 包括文字框坐标和识别文字
             文字框坐标顺序左上, 右上, 左下, 右下
             识别文字与文字框坐标对应
             {
                'ocr_text_res': ocr_text_res,
                'ocr_region_res': ocr_region_res
             }
    """

    if request.method == 'POST':
        img_path = request.form.get('img_path')
        detection = request.form.get('detection')

        if not detection:
            detection = 'ctpn'

        # 读取图片
        im = Image.open(img_path)
        img = np.array(im.convert('RGB'))

        # 执行ocr获得结果
        result, img, angle, ocr_region_res = model.model(
            img, img_path, model='pytorch', adjust=False, detectAngle=False, detect=detection)

        ocr_text_res = []
        for i, key in enumerate(result):
            ocr_text_res.append(result[key][1])

        response_data = {
            'ocr_text_res': ocr_text_res,
            'ocr_region_res': ocr_region_res
        }

        return jsonify(response_data)


if __name__ == '__main__':
    app.run()

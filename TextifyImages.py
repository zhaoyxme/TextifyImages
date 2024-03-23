from flask import Flask, request, render_template_string
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import base64

# 初始化Flask应用和模型
app = Flask(__name__)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
<title>Image Captioning</title>
<style>
/* 隐藏原生的文件输入 */
#file-upload {
  display: none;
}

/* 自定义的上传按钮样式 */
.custom-file-upload {
  display: inline-block;
  padding: 6px 12px;
  cursor: pointer;
  background-color: #f8f9fa;
  border: 1px solid #ddd;
  border-radius: 4px;
}
</style>
</head>
<body>
<h1>Upload an image</h1>
<form method="post" enctype="multipart/form-data">
  <!-- 隐藏的原生文件输入 -->
  <input id="file-upload" type="file" name="file"/>
  <!-- 自定义的文件上传按钮 -->
  <label for="file-upload" class="custom-file-upload">
    Choose File
  </label>
  <input type="submit" value="Upload">
</form>
{% if caption %}
    <h2>Caption:</h2>
    <p>{{ caption }}</p>
    <h2>Image:</h2>
    <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded Image"/>
{% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件在请求中
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # 如果用户没有选择文件，浏览器也会提交一个空的文件部分
        if file.filename == '':
            return 'No selected file'
        if file:
            # 读取图片文件并转换为PIL Image
            image_bytes = io.BytesIO(file.read())
            image = Image.open(image_bytes).convert('RGB')

            # 使用模型生成描述
            inputs = processor(image, return_tensors="pt")
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # 将图片转换为Base64编码
            image_bytes.seek(0)  # 重置文件指针位置
            b64_image = base64.b64encode(image_bytes.read()).decode('utf-8')

            # 返回带有描述和图片的页面
            return render_template_string(HTML_TEMPLATE, caption=caption, image_data=b64_image)

    return render_template_string(HTML_TEMPLATE, caption=None)

if __name__ == '__main__':
    app.run(debug=True)
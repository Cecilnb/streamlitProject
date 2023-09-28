import streamlit as st
import json
import tempfile
import io
import os
from PIL import Image
from typing import List

# 导入背景移除相关的函数和类

from rembg.bg import remove
from rembg.session_factory import new_session
from rembg.sessions import sessions_names

# 创建Streamlit应用程序
st.title("Background Removal with Streamlit")

# 上传模特图像
uploaded_image = st.file_uploader("Upload Model Image", type=["jpg", "png", "jpeg"])

# 添加选项和参数的Streamlit用户界面元素
model = st.selectbox("Choose Model", sessions_names, index=sessions_names.index("u2net_cloth_seg"))
alpha_matting = st.checkbox("Use Alpha Matting")
foreground_threshold = st.slider("Foreground Threshold", min_value=0, max_value=255, value=240)
background_threshold = st.slider("Background Threshold", min_value=0, max_value=255, value=10)
erode_size = st.slider("Erode Size", min_value=0, max_value=50, value=10)
only_mask = st.checkbox("Output Only Mask")
post_process_mask = st.checkbox("Post Process Mask")
background_color = st.color_picker("Background Color", value="#FFFFFF")

# 添加执行按钮
if st.button("Remove Background"):
    if uploaded_image is not None:
        # 保存上传的图像到临时文件夹
        temp_folder = tempfile.TemporaryDirectory()
        uploaded_image_path = os.path.join(temp_folder.name, "uploaded_model.png")
        with open(uploaded_image_path, "wb") as f:
            f.write(uploaded_image.read())

        # 构建参数字典
        kwargs = {
            "model": model,
            "alpha_matting": alpha_matting,
            "alpha_matting_foreground_threshold": foreground_threshold,
            "alpha_matting_background_threshold": background_threshold,
            "alpha_matting_erode_size": erode_size,
            "only_mask": only_mask,
            "post_process_mask": post_process_mask,
            "background_color": background_color,
        }

        # 执行背景移除
        # 读取上传的图像数据
        uploaded_image_data = open(uploaded_image_path, "rb").read()

        # 创建新会话
        session = new_session(model, **kwargs)

        # 执行背景移除
        result_data = remove(uploaded_image_data, session=session, **kwargs)

        # 将结果显示为图像
        result_image = Image.open(io.BytesIO(result_data))
        st.image(result_image, caption="Background Removed", use_column_width=True)

        # 清理临时文件夹
        temp_folder.cleanup()



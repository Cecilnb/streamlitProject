import streamlit as st
import os
import numpy as np
from PIL import Image
from u2net_cloth_seg import Unet2ClothSession

# 创建Streamlit应用程序
st.title("Clothing Mask Extraction")

# 上传模特图像
uploaded_file = st.file_uploader("Upload Model Image", type=["jpg", "png", "jpeg"])

# 衣物类型选项
cloth_category = st.selectbox("Select Cloth Category", ["upper", "lower", "full"])

if uploaded_file is not None:
    # 保存上传的图像到临时文件
    uploaded_image = Image.open(uploaded_file)
    temp_image_path = "temp_image.png"
    uploaded_image.save(temp_image_path, "PNG")

    # 创建 Unet2ClothSession 实例
    model_session = Unet2ClothSession()

    # 执行衣物提取
    masks = model_session.predict(uploaded_image, cloth_category=cloth_category)

    # 显示提取的蒙版
    st.subheader("Clothing Mask:")
    for i, mask in enumerate(masks):
        st.image(mask, caption=f"Mask {i+1}", use_column_width=True)

    # 清理临时文件
    os.remove(temp_image_path)

import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import os

# ================= 核心配置 =================

SYSTEM_PROMPT = """
You are a world-class comic book writer and storyboard artist.
IMPORTANT: After writing the script, extract the SINGLE BEST visual description for the main panel.
Enclose this visual description strictly within triple backticks and the label 'visual_prompt' like this:
```visual_prompt
(A detailed, hyper-realistic visual description of the main scene, American comic book art style, 8k resolution, dynamic lighting...)
"""
================= 功能模块 =================
def get_api_key(): if "GOOGLE_API_KEY" in os.environ: return os.environ["GOOGLE_API_KEY"] elif "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"] else: return st.sidebar.text_input("请输入 Google API Key", type="password")

def get_available_model(api_key): """【智能核心】自动寻找当前 Key 可用的模型""" genai.configure(api_key=api_key) try: # 问 Google：我现在能用啥？ valid_models = [] for m in genai.list_models(): if 'generateContent' in m.supported_generation_methods: valid_models.append(m.name)

    # 优先寻找 1.5 Pro, 然后 Flash, 然后 Pro
    preferred_order = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    
    # 1. 先在可用列表中找我们最想要的
    for pref in preferred_order:
        for valid in valid_models:
            if pref in valid:
                return valid # 找到了！
    
    # 2. 如果都没找到，就随便返回第一个能用的 Gemini 模型
    for valid in valid_models:
        if "gemini" in valid:
            return valid
            
    return None
except Exception as e:
    st.error(f"连接 Google 服务器失败，请检查 API Key 是否正确: {e}")
    return None
def generate_script(api_key, story_idea): """模块1：让 Gemini 写剧本""" # 先自动找模型 model_name = get_available_model(api_key)

if not model_name:
    st.error("❌ 你的 API Key 在 Google 上找不到任何可用的模型。")
    return None
    
try:
    # st.info(f"正在使用模型: {model_name}") # 调试信息
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(f"Create a comic script for: {story_idea}")
    return response.text
except Exception as e:
    st.error(f"剧本生成出错 ({model_name}): {e}")
    return None
def extract_prompt(script_text): if not script_text: return None if "visual_prompt" in script_text: try: return script_text.split("visual_prompt")[1].split("```")[0].strip() except: pass return None

def generate_image_with_gemini(api_key, visual_prompt): """模块2：调用 Gemini (Imagen) 画图""" try: genai.configure(api_key=api_key) # 这里的 Imagen 模型名称通常比较固定，如果报错也可能是 Key 权限问题 imagen_model = genai.ImageGenerationModel("imagen-3.0-generate-001")

    response = imagen_model.generate_images(
        prompt=visual_prompt,
        number_of_images=1,
        aspect_ratio="1:1",
        safety_filter="block_only_high",
    )
    return response.images[0]
except Exception as e:
    st.error(f"绘图失败: {e}。可能是 Key 权限不足或地区限制。")
    return None
def remove_watermark(pil_image): """模块3：自动去水印""" img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) h, w = img_cv.shape[:2] mask = np.zeros(img_cv.shape[:2], np.uint8) cv2.rectangle(mask, (w - 300, h - 80), (w, h), 255, -1) cleaned_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA) return Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))

================= 软件界面 =================
st.set_page_config(page_title="一键连环画神器", layout="wide") st.title("🚀 连环画自动生成器 (Web版)") st.caption("流程：输入故事 -> 自动寻找可用模型 -> 写剧本 -> 画图 -> 去水印")

api_key = get_api_key()

if not api_key: st.info("👋 请配置 API Key 开始使用。")

user_input = st.text_area("输入你的故事想法：", height=100)

if st.button("开始制作", type="primary"): if not api_key: st.warning("请先配置 API Key！") st.stop()

status = st.status("正在启动流水线...", expanded=True)

status.write("🔍 正在自动匹配最佳模型...")
script = generate_script(api_key, user_input)

if script:
    status.write("📝 剧本已生成！")
    prompt = extract_prompt(script)
    if not prompt: prompt = user_input
    status.write(f"🎨 提取绘图指令: {prompt[:50]}...")
    
    status.write("🖼️ 正在生成高清图像 (Imagen)...")
    raw_image = generate_image_with_gemini(api_key, prompt)
    
    if raw_image:
        status.write("🧼 正在去除水印...")
        final_image = remove_watermark(raw_image)
        
        status.update(label="制作完成！", state="complete", expanded=False)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("连环画成品")
            st.image(final_image, use_container_width=True)
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            st.download_button("📥 下载图片", data=buf.getvalue(), file_name="comic.png", mime="image/png", type="primary")
        with col2:
            st.subheader("剧本详情")
            st.markdown(script)
    else:
        status.update(label="绘图失败", state="error")
else:
    status.update(label="剧本生成失败", state="error")

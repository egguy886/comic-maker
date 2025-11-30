import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import os

# 配置提示词
SYSTEM_PROMPT = """
You are a world-class comic book writer and storyboard artist.
IMPORTANT: After writing the script, extract the SINGLE BEST visual description for the main panel.
Enclose this visual description strictly within triple backticks and the label 'visual_prompt' like this:
```visual_prompt
(A detailed, hyper-realistic visual description of the main scene, American comic book art style, 8k resolution, dynamic lighting...)
"""
 
def get_api_key(): 
    # 优先读取环境变量或 Secrets 
    if "GOOGLE_API_KEY" in os.environ: 
        return os.environ["GOOGLE_API_KEY"] 
    elif "GOOGLE_API_KEY" in st.secrets: 
        return st.secrets["GOOGLE_API_KEY"] 
    else: 
        return st.sidebar.text_input("请输入 Google API Key", type="password")

def get_available_model(api_key): 
    # 自动寻找当前可用的 Gemini 模型 
    genai.configure(api_key=api_key) 
    try: 
        valid_models = [] 
        for m in genai.list_models(): 
            if 'generateContent' in m.supported_generation_methods: 
                valid_models.append(m.name)

        # 优先顺序：1.5 Pro -> 1.5 Flash -> Pro
        preferred = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        
        for p in preferred:
            for v in valid_models:
                if p in v:
                    return v
        
        # 如果都没找到，返回第一个带 gemini 的
        for v in valid_models:
            if "gemini" in v:
                return v
        return None
    except Exception as e:
        st.error(f"无法连接 Google 服务器: {e}")
        return None

def generate_script(api_key, story_idea): 
    # 第一步：写剧本 
    model_name = get_available_model(api_key) 
    if not model_name: 
        st.error("没有找到可用的模型，请检查 API Key。") 
        return None

    try:
        model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(f"Create a comic script for: {story_idea}")
        return response.text
    except Exception as e:
        st.error(f"剧本生成失败: {e}")
        return None

def extract_prompt(script_text): 
    # 辅助：提取提示词 
    if not script_text: 
        return None 
    if "visual_prompt" in script_text: 
        try: 
            return script_text.split("visual_prompt")[1].split("```")[0].strip() 
        except: 
            pass 
    return None

def generate_image_with_gemini(api_key, visual_prompt): 
    # 第二步：画图 (Imagen 3) 
    try: 
        genai.configure(api_key=api_key) 
        imagen_model = genai.ImageGenerationModel("imagen-3.0-generate-001") 
        response = imagen_model.generate_images(
            prompt=visual_prompt, 
            number_of_images=1, 
            aspect_ratio="1:1", 
            safety_filter="block_only_high", 
        ) 
        return response.images[0] 
    except Exception as e: 
        st.error(f"绘图失败: {e}") 
        return None

def remove_watermark(pil_image): 
    # 第三步：去水印 
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
    h, w = img_cv.shape[:2] 
    mask = np.zeros(img_cv.shape[:2], np.uint8) 
    # 设定右下角去水印区域 (300x80) 
    cv2.rectangle(mask, (w - 300, h - 80), (w, h), 255, -1) 
    cleaned_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA) 
    return Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))

# 界面主程序
st.set_page_config(page_title="一键连环画", layout="wide") 
st.title("🚀 连环画自动生成器")

api_key = get_api_key()

if not api_key: 
    st.info("请配置 API Key。")

user_input = st.text_area("输入故事想法：", height=100)

if st.button("开始制作", type="primary"): 
    if not api_key: 
        st.warning("请先配置 Key！") 
        st.stop()

    status = st.status("正在制作中...", expanded=True)

    status.write("1. 正在写剧本...")
    script = generate_script(api_key, user_input)

    if script:
        prompt = extract_prompt(script)
        if not prompt: 
            prompt = user_input
        status.write("2. 正在绘图 (Imagen 3)...")

        raw_image = generate_image_with_gemini(api_key, prompt)

        if raw_image:
            status.write("3. 正在去水印...")
            final_image = remove_watermark(raw_image)
            status.update(label="完成！", state="complete", expanded=False)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(final_image, use_container_width=True)
                buf = io.BytesIO()
                final_image.save(buf, format="PNG")
                st.download_button("下载图片", data=buf.getvalue(), file_name="comic.png", mime="image/png")
            with col2:
                st.markdown(script)
        else:
            status.write("绘图失败，请查看上方报错。")
    else:
        status.write("剧本生成失败。")

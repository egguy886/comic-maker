import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import os

# ================= 核心配置 =================

# 你的御用编剧提示词
SYSTEM_PROMPT = """
You are a world-class comic book writer and storyboard artist. You create visually stunning American-style full-color comics.
Your core mission is to create a detailed script and visual description for a comic book based on the user's story.
IMPORTANT: After writing the script, extract the SINGLE BEST visual description for the main panel.
Enclose this visual description strictly within triple backticks like this:
```visual_prompt
(A detailed, hyper-realistic visual description of the main scene, American comic book art style, 8k resolution, dynamic lighting...)
""

================= 功能模块 =================
def get_api_key(): """尝试从服务器机密或侧边栏获取 API Key""" # 优先读取 HuggingFace/Streamlit 的 Secrets if "GOOGLE_API_KEY" in os.environ: return os.environ["GOOGLE_API_KEY"] elif "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"] else: return st.sidebar.text_input("请输入 Google API Key", type="password")

def generate_script(api_key, story_idea): """模块1：让 Gemini 写剧本""" # 注意：这里所有的标点都必须是英文 try: genai.configure(api_key=api_key) # 使用 Gemini 1.5 Pro model = genai.GenerativeModel("gemini-1.5-pro-latest", system_instruction=SYSTEM_PROMPT) response = model.generate_content(f"Create a comic script for: {story_idea}") return response.text except Exception as e: st.error(f"剧本生成出错: {e}") return None

def extract_prompt(script_text): """辅助：从剧本里抠出画画用的提示词""" if "visual_prompt" in script_text: try: return script_text.split("visual_prompt")[1].split("```")[0].strip() except: pass return None

def generate_image_with_gemini(api_key, visual_prompt): """模块2：调用 Gemini (Imagen) 画图""" try: genai.configure(api_key=api_key) # 调用 Imagen 3 模型 imagen_model = genai.ImageGenerationModel("imagen-3.0-generate-001")

    response = imagen_model.generate_images(
        prompt=visual_prompt,
        number_of_images=1,
        aspect_ratio="1:1",
        safety_filter="block_only_high",
    )
    return response.images[0] # 返回 PIL Image 对象
except Exception as e:
    st.error(f"绘图失败: {e}。可能是 Key 权限不足或地区限制。")
    return None
def remove_watermark(pil_image): """模块3：自动去水印 (右下角强力修复)""" # 转为 OpenCV 格式 img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) h, w = img_cv.shape[:2]

# === 定义要去水印的区域 (右下角) ===
mask = np.zeros(img_cv.shape[:2], np.uint8)
# y: h-80 到 h, x: w-300 到 w
cv2.rectangle(mask, (w - 300, h - 80), (w, h), 255, -1)

# 智能修复 (Inpainting)
cleaned_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)

# 转回 PIL 格式以便显示
return Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
================= 软件界面 =================
st.set_page_config(page_title="一键连环画神器", layout="wide") st.title("🚀 连环画自动生成器 (Web版)") st.caption("流程：输入故事 -> Gemini写剧本 -> Gemini画图 -> 自动去水印")

获取 API Key
api_key = get_api_key()

if not api_key: st.info("👋 欢迎！请在代码配置中设置 Secrets，或在左侧侧边栏输入 Key 开始使用。")

输入框
user_input = st.text_area("在这个框里输入你的画面/故事想法：", height=100)

if st.button("开始制作", type="primary"): if not api_key: st.warning("请先配置 API Key！") st.stop()

status = st.status("正在启动流水线...", expanded=True)

# 1. 写剧本
status.write("✍️ 正在构思剧本...")
script = generate_script(api_key, user_input)

if script:
    # 2. 提取提示词
    prompt = extract_prompt(script)
    if not prompt: 
        prompt = user_input
    status.write(f"🎨 提取绘图指令: {prompt[:50]}...")
    
    # 3. 画图
    status.write("🖼️ 正在生成高清图像 (调用 Imagen)...")
    raw_image = generate_image_with_gemini(api_key, prompt)
    
    if raw_image:
        # 4. 去水印
        status.write("🧼 正在执行去水印修复...")
        final_image = remove_watermark(raw_image)
        
        status.update(label="制作完成！", state="complete", expanded=False)
        
        # 结果展示
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("连环画成品")
            st.image(final_image, use_container_width=True)
            
            # 下载按钮
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            st.download_button("📥 下载图片", data=buf.getvalue(), file_name="comic_card.png", mime="image/png", type="primary")
            
        with col2:
            st.subheader("剧本详情")
            st.markdown(script)
    else:
        status.update(label="绘图失败", state="error")
else:
    status.update(label="剧本生成失败", state="error")

### 修正要点（供你参考）：
1.  **英文冒号**：第 22 行 `def generate_script(...):` 后面必须是英文冒号 `:`。
2.  **换行缩进**：Python 对缩进非常敏感，`try:` 必须另起一行，并向右缩进。
3.  **Secrets 读取**：我增加了一行 `os.environ` 的判断，这样无论你在 Hugging Face 还是 Streamlit Cloud，都能更稳地读取到 API Key。

保存后，Hugging Face 或 Streamlit 会自动重新部署（Building），稍等 1-2 分钟即可恢复正常。

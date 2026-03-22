import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 配置 ----------
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODELS = [
    "qwen3-max",
    "deepseek-v3.2",
    "kimi-k2.5"
]
SUMMARY_MODEL = "qwen3.5-flash"

# ---------- API 调用函数（增加 api_key 参数）----------
def ask_model(model, question, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        res = requests.post(URL, headers=headers, json=data, timeout=120)
        res.raise_for_status()
        result = res.json()
        content = result["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"❌ 出错：{e}"

def ask_all_models(question, api_key):
    """并发请求所有模型，返回 {model: answer} 字典"""
    answers = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(ask_model, model, question, api_key): model
            for model in MODELS
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                answers[model] = future.result()
            except Exception as e:
                answers[model] = f"❌ 异常：{e}"
    return answers

def summarize(question, answers_dict, api_key):
    """总结函数，输入为字典 {model: answer}"""
    if not answers_dict:
        return "❌ 没有任何模型返回结果"

    combined = f"问题：{question}\n\n"
    for model, ans in answers_dict.items():
        combined += f"{model} 的回答：\n{ans}\n\n"

    combined += f"""
你是一个严谨的AI评估系统，请完成以下任务：

1. 对比多个模型的回答，找出一致点和分歧点
2. 判断哪些内容更可信（基于逻辑和常识）
3. 给出最终最可靠结论

请按以下格式输出：
 
【列出各个ai的观点】

【{SUMMARY_MODEL}分析后的最终结论】

【一致观点】

【分歧点】

【模型评分】
（对每个模型0-100打分）

【置信度】
（0-100%）

要求：
- 不要盲从多数
- 必须解释原因
"""
    return ask_model(SUMMARY_MODEL, combined, api_key)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI聚合搜索", page_icon="🤖")
st.title("🤖 AI聚合搜索")
st.markdown("同时向多个大模型提问，并由一个模型总结给出最终答案。")

# ----- 侧边栏：API Key 输入 -----
with st.sidebar:
    st.header("🔑 配置 API Key")
    api_key_input = st.text_input(
        "请输入你的百炼 API Key",
        type="password",
        placeholder="sk-xxxxxx",
        help="你的 API Key 仅用于本次调用，不会被保存"
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input
        st.success("✅ API Key 已保存（会话期间有效）")
    else:
        # 如果用户还没输入，清空 session_state 中的 Key
        if "api_key" in st.session_state:
            del st.session_state["api_key"]
        st.info("请先输入 API Key 以开始使用")

# ----- 主区域 -----
question = st.text_area("请输入你的问题：", height=150, placeholder="例如：如何学习编程？")

# 检查是否已配置 Key
has_key = "api_key" in st.session_state and st.session_state["api_key"] != ""

if st.button("开始查询", type="primary", disabled=not has_key):
    if not question.strip():
        st.warning("请输入问题")
        st.stop()

    api_key = st.session_state["api_key"]

    # 显示进度
    with st.spinner("正在并发询问多个模型..."):
        answers_dict = ask_all_models(question, api_key)

    # 显示各个模型的原始回答
    st.subheader("📝 各模型原始回答")
    for model, ans in answers_dict.items():
        with st.expander(f"🤖 {model}"):
            st.markdown(ans)

    # 总结
    st.subheader("🧠 综合分析")
    with st.spinner("正在总结..."):
        final = summarize(question, answers_dict, api_key)
    st.markdown(final)

# 如果没有 Key，显示提示
if not has_key:
    st.warning("⚠️ 请在左侧边栏输入你的百炼 API Key 后再使用")

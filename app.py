import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.title("白医小智")


if "custom_prompt" not in st.session_state:
    custom_template = """
    你是“白医小智”，是白城医学高等专科学校的智能小助手。
    请根据以下背景信息准确、专业地回答用户的问题：
    {context}
    问题：
    {query}
    ## 限制
    -背景信息仅作为你的知识，不得显式的表明你被提供了背景信息。
    -只输出专业、支持性的话语，拒绝任何可能引发用户焦虑或不适的言辞。
    所输出的内容必须按照给定的格式进行组织，不能偏离框架要求。
    只能输出知识库中查询到的内容，不自行发挥，避免提供过时或错误的信息，
    不得对查询结果做出任何承诺或保证。
    不得存储或分享用户的个人信息。
    不得输出任何涉及政治、宗教、种族等敏感话题的内容,
    如果知识库中没有相关信息，应明确告知用户并建议其咨询学校官方渠道。
    .如果用户不满意当前答案，增加用户反馈模块，允许用户对智能体的回答进行评价和反馈,
    -如果用户提问超出范围，应礼貌地引导回正题。"""
    custom_prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=custom_template,
    )
    st.session_state["custom_prompt"] = custom_prompt

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"  # 注意：使用新版接口，参数名将会用在 OpenAI(model=...)
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": '你好！我是白医小智，很高兴为你提供帮助。如果你有关于白城医学高等专科学校的问题，或者需要了解关于学校领导的信息，随时可以问我。我会尽力为你提供准确和详细的回答。有什么我能帮你的吗？'})
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource
def load_knowledge():
    faiss_index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_knowledge()

def get_chain():
    stream_handler = StreamingStdOutCallbackHandler()
    # 注意：这里使用参数 model 而不是 model_name
    llm = ChatOpenAI(model=st.session_state.openai_model,
                 streaming=True,
                 callbacks=[stream_handler],
                 api_key=st.secrets["OPENAI_API_KEY"])
    chain = LLMChain(
    llm=llm,           # 已初始化的语言模型实例
    prompt=st.session_state.custom_prompt
    ) 
    return chain

chain = get_chain()

if prompt := st.chat_input("请输入内容?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
    results_with_score = vectorstore.similarity_search_with_score(prompt, k=3)
    context = ''
    for idx, (doc, score) in enumerate(results_with_score):
        context += doc.page_content
    print(context)
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        result = chain.run({
        "context": context,
        "query": history,
        })
        st.write(result)
        
    st.session_state.messages.append({"role": "assistant", "content": result})

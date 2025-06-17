import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- CÀI ĐẶT CHUNG CHO TRANG ---
st.set_page_config(page_title="ChatPDF Cục Bộ", layout="wide")

# --- HÀM XỬ LÝ ---

@st.cache_resource
def create_vector_store(file_path, _embeddings):
    """Tải, phân đoạn và tạo vector store từ file PDF."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(texts, _embeddings)
    return vectorstore

@st.cache_data
def get_file_name(uploaded_file):
    """Lấy tên file đã tải lên."""
    return uploaded_file.name if uploaded_file else "Chưa có file nào"

# --- GIAO DIỆN THANH BÊN (SIDEBAR) ---

with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    st.subheader("1. Tải lên tài liệu của bạn")
    uploaded_file = st.file_uploader("Chọn một file PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        # Lưu file tạm thời để LangChain có thể đọc
        temp_dir = "temp_docs"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success(f"Đã tải lên: {uploaded_file.name}")

    st.subheader("2. Chọn mô hình")
    # Giả lập việc chọn mô hình. Bạn có thể mở rộng để thực sự thay đổi mô hình
    # Đảm bảo bạn đã chạy `ollama pull <model_name>` cho các mô hình này
    model_name = st.selectbox(
        "Chọn mô hình LLM (chạy bằng Ollama):",
        ("llama3", "mistral", "gemma"),
        key="llm_model"
    )

    if st.button("Bắt đầu cuộc trò chuyện mới", type="primary"):
        # Xóa lịch sử chat và vectorstore để bắt đầu lại
        st.session_state.messages = []
        st.session_state.vectorstore = None
        # Xóa file tạm nếu có
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             os.remove(temp_file_path)
        st.success("Đã sẵn sàng cho cuộc trò chuyện mới!")


# --- KHỞI TẠO STATE VÀ CÁC BIẾN CẦN THIẾT ---

# Khởi tạo session state để lưu tin nhắn
if "messages" not in st.session_state:
    st.session_state.messages = []

# Khởi tạo vector store trong session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Khởi tạo các mô hình từ Ollama
try:
    embeddings = OllamaEmbeddings(model=st.session_state.get('llm_model', 'llama3'))
    llm = Ollama(model=st.session_state.get('llm_model', 'llama3'))
except Exception as e:
    st.error(f"Không thể kết nối tới Ollama. Hãy đảm bảo Ollama đang chạy.\nLỗi: {e}")
    st.stop()


# Xử lý file khi được tải lên và vectorstore chưa được tạo
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Đang xử lý tài liệu... Việc này có thể mất một lúc."):
        st.session_state.vectorstore = create_vector_store(temp_file_path, embeddings)


# --- GIAO DIỆN CHÍNH (MAIN CHAT AREA) ---

st.title("💬 ChatPDF Cục bộ")
st.caption(f"Đang sử dụng mô hình: {st.session_state.get('llm_model', 'llama3')}")


# Hiển thị lịch sử tin nhắn
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if prompt := st.chat_input("Hỏi bất cứ điều gì về tài liệu của bạn..."):
    # Thêm tin nhắn của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo và hiển thị câu trả lời của AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("AI đang suy nghĩ..."):
            if st.session_state.vectorstore is not None:
                # Tạo chuỗi truy vấn nếu có tài liệu
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                response = qa_chain.invoke(prompt)
                full_response = response.get('result', "Xin lỗi, tôi không tìm thấy câu trả lời trong tài liệu.")
            else:
                # Trả lời chung nếu chưa có tài liệu
                full_response = "Vui lòng tải lên một file PDF ở thanh bên trái để tôi có thể trả lời câu hỏi của bạn."

        message_placeholder.markdown(full_response)

    # Thêm câu trả lời của AI vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})

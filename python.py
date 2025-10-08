import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# **********************************************************************
# KHỞI TẠO STATE CHO KHUNG CHAT (Mới)
# **********************************************************************
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini (Dùng cho Nhận xét nhanh - Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tóm tắt."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# **********************************************************************
# HÀM MỚI: KHỞI TẠO VÀ HIỂN THỊ KHUNG CHAT TRÊN SIDEBAR
# **********************************************************************

def initialize_gemini_chat(data_for_ai, api_key):
    """Khởi tạo phiên chat Gemini với ngữ cảnh là dữ liệu tài chính."""
    if st.session_state.chat_session is None:
        try:
            # 1. Khởi tạo Client
            client = genai.Client(api_key=api_key)
            model_name = 'gemini-2.5-flash'
            
            # 2. Định nghĩa System Instruction (Context)
            system_instruction = f"""
            Bạn là một chuyên gia phân tích tài chính chuyên nghiệp, có tên là Hatyx AI. 
            Nhiệm vụ của bạn là giúp người dùng phân tích sâu hơn về Báo cáo Tài chính (BCTC) mà họ đã tải lên.
            
            Dữ liệu BCTC đã được xử lý (Tốc độ Tăng trưởng và Tỷ trọng) như sau:
            {data_for_ai}
            
            Hãy trả lời các câu hỏi của người dùng về dữ liệu này, duy trì ngữ cảnh tài chính và đưa ra những nhận xét chuyên sâu, khách quan. Không cần lặp lại dữ liệu trên trong câu trả lời, chỉ cần trả lời câu hỏi.
            """
            
            # 3. Tạo Chat Session
            chat = client.chats.create(
                model=model_name,
                system_instruction=system_instruction
            )
            
            st.session_state.chat_session = chat
            
            # 4. Tin nhắn chào mừng
            st.session_state.messages = [] # Reset messages
            st.session_state.messages.append({"role": "assistant", "content": "Chào bạn! Tôi là Hatyx AI. Dữ liệu tài chính của bạn đã sẵn sàng. Hãy hỏi tôi về **tốc độ tăng trưởng**, **cơ cấu tài sản**, hoặc bất kỳ chỉ số nào trong báo cáo này."})
            st.session_state.chat_ready = True
        
        except APIError:
            st.sidebar.error("Lỗi API: Vui lòng kiểm tra Khóa API Gemini.")
            st.session_state.chat_ready = False
        except Exception as e:
            st.sidebar.error(f"Lỗi không xác định khi khởi tạo chat: {e}")
            st.session_state.chat_ready = False

def render_chat_sidebar(data_for_ai):
    """Vẽ giao diện khung chat trên sidebar."""
    st.sidebar.subheader("💬 Hatyx AI Chat - Phân tích chuyên sâu")
    
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.sidebar.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng Chat.")
        return

    # Nút Bắt đầu Chat
    if st.session_state.chat_session is None:
        if st.sidebar.button("Bắt đầu Chat Phân Tích"):
            initialize_gemini_chat(data_for_ai, api_key)
        else:
            st.sidebar.info("Nhấn nút để khởi động phiên chat với ngữ cảnh là dữ liệu BCTC đã tải.")
    
    # Nút Reset Chat
    if st.session_state.chat_ready:
        if st.sidebar.button("Reset Chat", type="secondary"):
            st.session_state.chat_session = None
            st.session_state.messages = []
            st.session_state.chat_ready = False
            # Dùng st.experimental_rerun() để làm mới giao diện
            st.experimental_rerun()
            return
            
    # Hiển thị lịch sử chat
    chat_container = st.sidebar.container(height=400) # Giới hạn chiều cao cho thanh cuộn
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Xử lý input mới
    if st.session_state.chat_ready and st.session_state.chat_session:
        # Thẻ chat_input phải được gọi sau khi lịch sử đã được vẽ
        if prompt := st.sidebar.chat_input("Hỏi Hatyx AI về báo cáo này..."):
            
            # 1. Thêm tin nhắn người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Hiển thị tin nhắn người dùng
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # 2. Gửi tin nhắn đến Gemini
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Hatyx AI đang phân tích..."):
                        try:
                            response = st.session_state.chat_session.send_message(prompt)
                            ai_response = response.text
                            st.markdown(ai_response)
                            
                            # 3. Thêm tin nhắn AI vào lịch sử
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                        except APIError as e:
                            error_message = f"Lỗi gọi Gemini API: {e}. Vui lòng kiểm tra Khóa API."
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                        except Exception as e:
                            error_message = f"Lỗi không xác định: {e}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# --- Logic chính ---
data_for_ai_markdown = None # Khởi tạo biến này để truyền context vào chat

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định để tránh lỗi khi tính toán không thành công
            thanh_toan_hien_hanh_N = 'N/A'
            thanh_toan_hien_hanh_N_1 = 'N/A'
            tsnh_tang_truong = 'N/A'
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n_val = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1_val = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                tsnh_tang_truong_val = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N_val = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1_val = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                if no_ngan_han_N_val != 0:
                    thanh_toan_hien_hanh_N = tsnh_n_val / no_ngan_han_N_val
                if no_ngan_han_N_1_val != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1_val / no_ngan_han_N_1_val

                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1
                    )
                with col2:
                    delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N, (int, float)) and isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else None
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N,
                        delta=f"{delta_value:.2f}" if delta_value is not None else None
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                st.warning("Nợ Ngắn Hạn bằng 0. Không thể tính Chỉ số Thanh toán Hiện hành.")
            
            # --- Chuẩn bị dữ liệu để gửi cho AI (cho cả Chức năng 5 và Chat) ---
            data_for_ai_markdown = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{tsnh_tang_truong_val:.2f}%" if isinstance(tsnh_tang_truong_val, (int, float)) else tsnh_tang_truong, 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1, 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N
                ]
            }).to_markdown(index=False) 

            # **********************************************************************
            # GỌI KHUNG CHAT TRÊN SIDEBAR (Mới)
            # **********************************************************************
            render_chat_sidebar(data_for_ai_markdown)


            # --- Chức năng 5: Nhận xét AI (Button gốc) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI - Tóm tắt nhanh)")
            
            if st.button("Yêu cầu AI Phân tích (Tóm tắt nhanh)"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai_markdown, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.chat_session = None
        st.session_state.messages = []
        st.session_state.chat_ready = False
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
        st.session_state.chat_session = None
        st.session_state.messages = []
        st.session_state.chat_ready = False

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    st.sidebar.info("Sau khi tải file, khung chat sẽ xuất hiện ở đây để bạn có thể hỏi sâu hơn về báo cáo.")

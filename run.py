import streamlit as st
from datetime import datetime
from PIL import Image
from app_streaming import run_emotion_analysis
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ——— Page config & title ———
st.set_page_config(layout='wide', page_title='ethicapp')
st.title('감정을 읽는 기계')

# ——— Sidebar navigation menu ———
st.sidebar.subheader('Menu …')
page = st.sidebar.radio(
    '',
    ['Home', 'Teachable Machine','Emotion Analysis', 'Student Data','Help']
)

# ——— Main layout: two columns (4:1) ———
left_col, right_col = st.columns([4, 1])

if page == 'Home':
    with left_col:
        st.subheader('Content')
        st.video('https://www.youtube.com/watch?v=lkT6qg55kpE') #https://youtu.be/CShXWACuGp8?si=ANvHKLLaTQq6jU00


        # 폰트 크기를 키워서 안내 문구 출력
        st.markdown(
            """
            <p style='font-size:20px; font-weight:bold;'>기계가 감정을 읽을 수 있다고 생각하나요?</p>
            """,
            unsafe_allow_html=True
        )
        thoughts = st.text_area('학생 개인 생각을 기록하세요:', height=150)
        if st.button('제출'):
            if thoughts.strip():
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = f'[{timestamp}] {thoughts}\n'
                with open('data.txt', 'a', encoding='utf-8') as f:
                    f.write(entry)
                st.success('생각이 성공적으로 제출되었습니다!')
            else:
                st.warning('생각을 입력한 후 제출해주세요.')


        # thoughts = st.text_area('기계가 감정을 읽을 수 있을까?', height=150)
        # if st.button('제출'):
        #     if thoughts.strip():
        #         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #         entry = f'[{timestamp}] {thoughts}\n'
        #         try:
        #             with open('data.txt', 'a', encoding='utf-8') as f:
        #                 f.write(entry)
        #             st.success('생각이 성공적으로 제출되었습니다!')
        #         except Exception as e:
        #             st.error(f'제출 중 오류 발생: {e}')
        #     else:
        #         st.warning('생각을 입력한 후 제출해주세요.')

    with right_col:
        st.subheader('Tips & Help')
        st.markdown(
            '''
- 💡 **Tip 1:** 윤리적 딜레마가 발생할 수 있는 상황을 미리 상상해 보세요.  
- 💡 **Tip 2:** AI가 내린 판단을 그대로 믿기보다, 항상 비판적으로 검토하세요.  
- ❓ **Help:** 문제가 있을 때 사이드바의 ‘문의하기’ 버튼을 눌러주세요.
            '''
        )

elif page == 'Teachable Machine':
    # 외부 Teachable Machine 페이지로 이동
    components.html(
        """
        <script>
            window.open('https://teachablemachine.withgoogle.com/train', '_blank')
        </script>
        """
    )
    st.write('Teachable Machine 페이지로 이동 중입니다...')


elif page == 'Emotion Analysis':
    st.subheader("Real‑Time Emotion Analysis")

    # 1) 트랜스포머 클래스 정의
    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.label_map, self.model = load_emotion_model()
            self.detector = load_face_detector()

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb)
            if detections:
                x, y, w, h = detections[0]['box']
                x, y = max(0, x), max(0, y)
                face = rgb[y:y+h, x:x+w]
                if face.size and w >= 20 and h >= 20:
                    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    resized = cv2.resize(gray, (48, 48))
                    x_input = np.expand_dims(resized.astype("float32")/255.0, axis=(0, -1))
                    proba = self.model.predict(x_input)[0]
                    idx = int(np.argmax(proba))
                    label = self.label_map[idx]
                    status = (
                        "Positive" if label=="Happy"
                        else "Negative" if label in ["Sad","Angry","Disgust","Fear"]
                        else "Neutral"
                    )
                    cv2.putText(img, f"{status} ({label})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # 2) 스트리밍 위젯 호출
    webrtc_streamer(
        key="emotion",
        mode="SENDRECV",
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )














    # 왼쪽에서 분석 및 피드백 폼 표시
    # with left_col:
    #     # 실시간 감정 분석 시작 버튼
    #     if st.button('Start Emotion Analysis'):
    #         run_emotion_analysis()
    #     st.subheader('학생 피드백 기록')
    #     student_name = st.text_input('Student')
    #     incorrect = st.text_area('Incorrect Analysis', height=100)
    #     reason = st.text_area('Reasons for Missing', height=100)
    #     if st.button('Submit Feedback'):
    #         if student_name.strip() and incorrect.strip() and reason.strip():
    #             ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             entry = f'[{ts}] Student: {student_name} | Incorrect Analysis: {incorrect} | Reason: {reason}\n'
    #             try:
    #                 with open('analyze.txt', 'a', encoding='utf-8') as f:
    #                     f.write(entry)
    #                 st.success('Feedback submitted!')
    #             except Exception as e:
    #                 st.error(f'Error saving feedback: {e}')
    #         else:
    #             st.warning('모든 필드를 입력한 후 제출해주세요.')



    # 왼쪽에서 감정 분석과 피드백 폼을 렌더링합니다.
    # with left_col:
    #     run_emotion_analysis()
    #     # 학생 피드백 기록 폼
    #     st.subheader('학생 피드백 기록')
    #     student_name = st.text_input('Student')
    #     incorrect = st.text_area('Incorrect Analysis', height=100)
    #     reason = st.text_area('Reasons for Missing', height=100)
    #     if st.button('Submit Feedback'):
    #         if student_name.strip() and incorrect.strip() and reason.strip():
    #             ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             entry = f'[{ts}] Student: {student_name} | Incorrect Analysis: {incorrect} | Reason: {reason}\n'
    #             try:
    #                 with open('analyze.txt', 'a', encoding='utf-8') as f:
    #                     f.write(entry)
    #                 st.success('Feedback submitted!')
    #             except Exception as e:
    #                 st.error(f'Error saving feedback: {e}')
    #         else:
    #             st.warning('모든 필드를 입력한 후 제출해주세요.')



elif page == 'Student Data':
    with left_col:
        st.subheader('Stored Student Data')
        # data.txt 내용 표시
        try:
            with open('data.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            st.text_area('', content, height=300)
        except FileNotFoundError:
            st.error('data.txt 파일이 없습니다.')
        except Exception as e:
            st.error(f'data.txt 불러오기 중 오류 발생: {e}')

        # analyze.txt 데이터를 테이블로 표시
        st.subheader('Emotional Analysis Results')
        try:
            import pandas as pd
            rows = []
            with open('analyze.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    # Expected format: [timestamp] Student: name | Incorrect Analysis: incorrect | Reason: reason
                    try:
                        parts = line.strip().split('|')
                        name = parts[0].split('Student:')[1].strip()
                        incorrect = parts[1].split('Incorrect Analysis:')[1].strip()
                        reason = parts[2].split('Reason:')[1].strip()
                        rows.append({'학번': name, '잘못 인식된 감정': incorrect, '이유': reason})
                    except Exception:
                        continue
            if rows:
                df = pd.DataFrame(rows)
                st.table(df)
            else:
                st.info('analyze.txt에 기록된 데이터가 없습니다.')
        except FileNotFoundError:
            st.warning('analyze.txt 파일이 없습니다.')
        except Exception as e:
            st.error(f'analyze.txt 불러오기 중 오류 발생: {e}')

    with right_col:
        st.write('')








# elif page == '학생 데이터':
#     with left_col:
#         st.subheader('저장된 학생 데이터')
#         try:
#             with open('data.txt', 'r', encoding='utf-8') as f:
#                 content = f.read()
#             st.text_area('', content, height=300)
#         except FileNotFoundError:
#             st.error('data.txt 파일이 없습니다.')
#         except Exception as e:
#             st.error(f'데이터 불러오기 중 오류 발생: {e}')
#     #(코드 개선 요구):학생이 작성한 analyze.txt저장된 데이터가 "student_name","incorrect","reason"을 컬럼명을 갖는 표로 출력된다. 표의 제목은 "감정 분석 결과"이다.
#     with right_col:
#         st.write('')  # 비어 있는 영역

elif page == 'Help':
    with left_col:
        st.subheader('Tips & Help')
        st.markdown(
            '''
- 💡 **Tip 1:** 윤리적 딜레마가 발생할 수 있는 상황을 미리 상상해 보세요.  
- 💡 **Tip 2:** AI가 내린 판단을 그대로 믿기보다, 항상 비판적으로 검토하세요.  
- ❓ **Help:** 문제가 있을 땈 사이드바의 ‘문의하기’ 버튼을 눌러주세요.
            '''
        )
    with right_col:
        st.write('')  # 비어 있는 영역

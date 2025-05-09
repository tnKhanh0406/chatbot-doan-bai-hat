import spacy
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import numpy as np
try:
    import whisper
except ImportError as e:
    print("⚠️ Không thể import whisper. MP3 sẽ chỉ dùng ACRCloud.")
    whisper = None
import hmac
import hashlib
import base64
import time
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import tempfile

# Cấu hình
MUSIXMATCH_API_KEY = "cba762d1c41f6650cb552d364182211f"
ACRCLOUD_ACCESS_KEY = "17e0737bb685313d72e4bec1e4746d49"
ACRCLOUD_SECRET_KEY = "zOTTYblA5dzbe9QU5tp6IB52gXcJaDyyyn0PF1Hh"
ACRCLOUD_HOST = "identify-ap-southeast-1.acrcloud.com"
TELEGRAM_TOKEN = "7792993578:AAGtBxd-WziWATzNhIw9Uq-WY_rzGIlQV64"
SONG_DATABASE_PATH = "songs_database.json"

# Load mô hình Intent Detection + NER
try:
    nlp_intent_ner = spacy.load("model_intent_ner")
except Exception as e:
    print(f"⚠️ Lỗi khi load model_intent_ner: {e}")
    exit(1)

# Load cơ sở dữ liệu bài hát cho TF-IDF
def load_song_database(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File {file_path} không tồn tại. TF-IDF sẽ không hoạt động.")
        return []

SONG_DATABASE = load_song_database(SONG_DATABASE_PATH)

# Hàm Musixmatch API
def search_song_musixmatch(lyrics, singer=None, top_n=3):
    url = "https://api.musixmatch.com/ws/1.1/track.search"
    params = {
        "q_lyrics": lyrics,
        "f_has_lyrics": 1,
        "s_track_rating": "desc",
        "apikey": MUSIXMATCH_API_KEY,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        track_list = data["message"]["body"]["track_list"]
        results = []
        for track_item in track_list[:top_n]:
            track = track_item["track"]
            results.append({
                "song": fix_encoding(track["track_name"]),
                "artist": fix_encoding(track["artist_name"]),
                "album": fix_encoding(track["album_name"]),
            })
        print(f"Musixmatch trả về {len(results)} bài hát")
        return results if results else None
    except requests.RequestException as e:
        print(f"⚠️ Lỗi khi gọi Musixmatch API: {e}")
        return None

# Hàm TF-IDF
def search_song_tfidf(lyrics, song_database, threshold=0.4, top_n=3):
    if not song_database:
        print("⚠️ Cơ sở dữ liệu bài hát trống.")
        return None
    documents = [song["lyrics"] for song in song_database]
    documents.append(lyrics)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Lấy chỉ số của top_n bài hát có độ tương đồng cao nhất
    top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    results = []
    
    for idx in top_indices:
        if cosine_similarities[idx] >= threshold:
            results.append({
                "song": song_database[idx]["song"],
                "artist": song_database[idx]["artist"],
                "similarity": cosine_similarities[idx],
            })
    
    return results if results else None

def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

# Hàm ACRCloud
def sign_request(method, uri, access_key, secret_key, timestamp, data_type="audio", signature_version="1"):
    string_to_sign = f"{method}\n{uri}\n{access_key}\n{data_type}\n{signature_version}\n{timestamp}"
    signature = hmac.new(secret_key.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1).digest()
    return base64.b64encode(signature).decode()

def recognize_song_mp3(file_path, top_n=3):
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại.")
        return None

    endpoint = f"http://{ACRCLOUD_HOST}/v1/identify"
    timestamp = str(int(time.time()))
    uri = "/v1/identify"

    signature = sign_request("POST", uri, ACRCLOUD_ACCESS_KEY, ACRCLOUD_SECRET_KEY, timestamp)

    with open(file_path, "rb") as f:
        file_data = f.read()

    data = {
        "access_key": ACRCLOUD_ACCESS_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "data_type": "audio",
        "signature_version": "1",
        "sample_bytes": str(len(file_data)),
    }

    files = {
        "sample": ("audio.mp3", file_data, "audio/mpeg")
    }

    try:
        print(f"Đang gửi yêu cầu tới ACRCloud với data: {data}")
        response = requests.post(endpoint, data=data, files=files)
        response.raise_for_status()
        result = response.json()
        print(f"Phản hồi từ ACRCloud: {result}")

        if result["status"]["msg"] == "Success":
            metadata = result["metadata"].get("music", [])
            results = []
            for track in metadata[:top_n]:
                results.append({
                    "song": fix_encoding(track["title"]),
                    "artist": fix_encoding(track["artists"][0]["name"]),
                    "album": fix_encoding(track.get("album", {}).get("name", "")),
                })
            return results if results else None
        else:
            return None
    except requests.RequestException as e:
        print(f"⚠️ Lỗi khi gọi ACRCloud API: {e}")
        return None

# Hàm trích xuất lời bài hát từ MP3
def extract_lyrics_from_mp3(file_path):
    if whisper is None:
        print("⚠️ Whisper không khả dụng. Bỏ qua trích xuất lời bài hát.")
        return None
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} không tồn tại.")
            return None
        print(f"Đang xử lý file MP3 với Whisper: {file_path}")
        model = whisper.load_model("base")
        result = model.transcribe(file_path, language="vi", verbose=True)
        lyrics = result["text"].strip()
        print(f"Kết quả Whisper: '{lyrics}'")
        if not lyrics:
            print("⚠️ Whisper không trích xuất được lời bài hát.")
            return None
        return lyrics
    except Exception as e:
        print(f"⚠️ Lỗi khi trích xuất lời bài hát: {e}")
        return None

# Hàm xử lý đầu vào
def process_input(lyrics=None, singer=None, mp3_file=None):
    if mp3_file:
        # Thử ACRCloud 
        results = recognize_song_mp3(mp3_file)
        if results:
            response = "🎵 Các bài hát từ ACRCloud:\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['song']} - {result['artist']} (Album: {result['album']})\n"
            return response

        # Thử Whisper
        lyrics = extract_lyrics_from_mp3(mp3_file)
        if lyrics:
            print(f"Lời bài hát từ Whisper: {lyrics}")
        else:
            return "⚠️ Không thể trích xuất lời bài hát từ MP3. Hãy thử gửi lời bài hát."

    if not lyrics:
        return "⚠️ Không tìm thấy lời bài hát trong đầu vào."

    # Thử TF-IDF
    results = search_song_tfidf(lyrics, SONG_DATABASE, threshold=0.4, top_n=3)
    if results:
        response = "🎵 Các bài hát tương đồng nhất (TF-IDF):\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result['song']} - {result['artist']} (Độ tương đồng: {result['similarity']:.2f})\n"
        return response

    # Thử Musixmatch API
    results = search_song_musixmatch(lyrics, singer)
    if results:
        response = "🎵 Các bài hát từ Musixmatch:\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result['song']} - {result['artist']} (Album: {result['album']})\n"
        return response

    return "⚠️ Không tìm thấy bài hát."

# Hàm xử lý Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào bạn! Tôi là bot tìm bài hát. Gửi một câu lời bài hát hoặc file MP3 để tôi tìm bài hát cho bạn!"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    doc = nlp_intent_ner(user_input)
    
    lyrics = None
    singer = None
    if doc.cats.get("find_song", 0) > 0.5:
        for ent in doc.ents:
            if ent.label_ == "LYRICS":
                lyrics = ent.text
            if ent.label_ == "SINGER":
                singer = ent.text
    
    result = process_input(lyrics=lyrics, singer=singer)
    await update.message.reply_text(result)

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    audio = update.message.audio or update.message.document
    if not audio:
        await update.message.reply_text("⚠️ Vui lòng gửi file MP3.")
        return
    
    # Tải file MP3
    file = await context.bot.get_file(audio.file_id)
    temp_file = None
    try:
        # Tạo file tạm với đuôi .mp3
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", mode='wb')
        await file.download_to_drive(temp_file.name)
        
        temp_file.close()
        
        # Gọi hàm xử lý MP3
        result = process_input(mp3_file=temp_file.name)
        await update.message.reply_text(result)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi khi xử lý file MP3: {str(e)}")
    finally:
        if temp_file:
            try:
                os.unlink(temp_file.name)  # Xóa file tạm
            except Exception as e:
                print(f"⚠️ Không thể xóa file tạm {temp_file.name}: {e}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.AUDIO | filters.Document.MP3, handle_audio))

    print("Bot đang chạy...")
    application.run_polling()

if __name__ == "__main__":
    main()
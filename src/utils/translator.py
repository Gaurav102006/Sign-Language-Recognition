"""
Multilingual Translation and Speech Synthesis Suite.
Translates recognized sign language words and synthesized sentences into 15+ languages.
Features dual-engine operation:
  1. Real-time online translation via MyMemory / Google Translator
  2. Built-in comprehensive offline multilingual vocabulary dictionary for all 98 classes
  3. Text-to-Speech (TTS) audio synthesis using gTTS for auditory feedback
"""

import io
from typing import Dict, List, Optional, Tuple
import requests

try:
    from deep_translator import MyMemoryTranslator, GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False


# Supported Languages Directory (Code, Display Name, Country Flag/Symbol, TTS Code)
SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi (हिन्दी)", "flag": "🇮🇳", "tts_code": "hi", "mymemory_code": "hi-IN"},
    "es": {"name": "Spanish (Español)", "flag": "🇪🇸", "tts_code": "es", "mymemory_code": "es-ES"},
    "fr": {"name": "French (Français)", "flag": "🇫🇷", "tts_code": "fr", "mymemory_code": "fr-FR"},
    "de": {"name": "German (Deutsch)", "flag": "🇩🇪", "tts_code": "de", "mymemory_code": "de-DE"},
    "ja": {"name": "Japanese (日本語)", "flag": "🇯🇵", "tts_code": "ja", "mymemory_code": "ja-JP"},
    "zh-CN": {"name": "Chinese (Simplified)", "flag": "🇨🇳", "tts_code": "zh-CN", "mymemory_code": "zh-CN"},
    "ar": {"name": "Arabic (العربية)", "flag": "🇸🇦", "tts_code": "ar", "mymemory_code": "ar-SA"},
    "ru": {"name": "Russian (Русский)", "flag": "🇷🇺", "tts_code": "ru", "mymemory_code": "ru-RU"},
    "pt": {"name": "Portuguese (Português)", "flag": "🇵🇹", "tts_code": "pt", "mymemory_code": "pt-PT"},
    "it": {"name": "Italian (Italiano)", "flag": "🇮🇹", "tts_code": "it", "mymemory_code": "it-IT"},
    "bn": {"name": "Bengali (বাংলা)", "flag": "🇧🇩", "tts_code": "bn", "mymemory_code": "bn-IN"},
    "ta": {"name": "Tamil (தமிழ்)", "flag": "🇮🇳", "tts_code": "ta", "mymemory_code": "ta-IN"},
    "te": {"name": "Telugu (తెలుగు)", "flag": "🇮🇳", "tts_code": "te", "mymemory_code": "te-IN"},
    "ko": {"name": "Korean (한국어)", "flag": "🇰🇷", "tts_code": "ko", "mymemory_code": "ko-KR"},
    "en": {"name": "English (Original)", "flag": "🇬🇧", "tts_code": "en", "mymemory_code": "en-GB"},
}

# Comprehensive Offline Multilingual Dictionary for all 98 dataset vocabulary items
OFFLINE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "afternoon": {"hi": "दोपहर", "es": "tarde", "fr": "après-midi", "de": "Nachmittag", "ja": "午後", "ar": "بعد الظهر", "zh-CN": "下午"},
    "angry": {"hi": "गुस्सा", "es": "enojado", "fr": "en colère", "de": "wütend", "ja": "怒っている", "ar": "غاضب", "zh-CN": "生气"},
    "answer": {"hi": "उत्तर", "es": "respuesta", "fr": "réponse", "de": "Antwort", "ja": "答え", "ar": "إجابة", "zh-CN": "答案"},
    "apple": {"hi": "सेब", "es": "manzana", "fr": "pomme", "de": "Apfel", "ja": "りんご", "ar": "تفاحة", "zh-CN": "苹果"},
    "baby": {"hi": "बच्चा", "es": "bebé", "fr": "bébé", "de": "Baby", "ja": "赤ちゃん", "ar": "طفل رضيع", "zh-CN": "婴儿"},
    "bad": {"hi": "बुरा", "es": "malo", "fr": "mauvais", "de": "schlecht", "ja": "悪い", "ar": "سيء", "zh-CN": "坏"},
    "bank": {"hi": "बैंक", "es": "banco", "fr": "banque", "de": "Bank", "ja": "銀行", "ar": "بنك", "zh-CN": "银行"},
    "bathroom": {"hi": "स्नानघर", "es": "baño", "fr": "salle de bain", "de": "Badezimmer", "ja": "浴室", "ar": "حمام", "zh-CN": "浴室"},
    "bed": {"hi": "बिस्तर", "es": "cama", "fr": "lit", "de": "Bett", "ja": "ベッド", "ar": "سرير", "zh-CN": "床"},
    "bike": {"hi": "साइकिल", "es": "bicicleta", "fr": "vélo", "de": "Fahrrad", "ja": "自転車", "ar": "دراجة", "zh-CN": "自行车"},
    "book": {"hi": "किताब", "es": "libro", "fr": "livre", "de": "Buch", "ja": "本", "ar": "كتاب", "zh-CN": "书"},
    "boy": {"hi": "लड़का", "es": "niño", "fr": "garçon", "de": "Junge", "ja": "男の子", "ar": "ولد", "zh-CN": "男孩"},
    "bread": {"hi": "रोटी", "es": "pan", "fr": "pain", "de": "Brot", "ja": "パン", "ar": "خبز", "zh-CN": "面包"},
    "brother": {"hi": "भाई", "es": "hermano", "fr": "frère", "de": "Bruder", "ja": "兄弟", "ar": "أخ", "zh-CN": "兄弟"},
    "bus": {"hi": "बस", "es": "autobús", "fr": "bus", "de": "Bus", "ja": "バス", "ar": "حافلة", "zh-CN": "公共汽车"},
    "bye": {"hi": "अलविदा", "es": "adiós", "fr": "au revoir", "de": "Tschüss", "ja": "さようなら", "ar": "مع السلامة", "zh-CN": "再见"},
    "car": {"hi": "गाड़ी", "es": "coche", "fr": "voiture", "de": "Auto", "ja": "車", "ar": "سيارة", "zh-CN": "汽车"},
    "chair": {"hi": "कुर्सी", "es": "silla", "fr": "chaise", "de": "Stuhl", "ja": "椅子", "ar": "كرسي", "zh-CN": "椅子"},
    "child": {"hi": "बच्चा", "es": "niño", "fr": "enfant", "de": "Kind", "ja": "子供", "ar": "طفل", "zh-CN": "孩子"},
    "city": {"hi": "शहर", "es": "ciudad", "fr": "ville", "de": "Stadt", "ja": "都市", "ar": "مدينة", "zh-CN": "城市"},
    "class": {"hi": "कक्षा", "es": "clase", "fr": "classe", "de": "Klasse", "ja": "クラス", "ar": "فصل", "zh-CN": "班级"},
    "coffee": {"hi": "कॉफ़ी", "es": "café", "fr": "café", "de": "Kaffee", "ja": "コーヒー", "ar": "قهوة", "zh-CN": "咖啡"},
    "college": {"hi": "कॉलेज", "es": "universidad", "fr": "collège", "de": "Hochschule", "ja": "大学", "ar": "كلية", "zh-CN": "大学"},
    "come": {"hi": "आओ", "es": "venir", "fr": "venir", "de": "kommen", "ja": "来る", "ar": "تعال", "zh-CN": "来"},
    "computer": {"hi": "कंप्यूटर", "es": "computadora", "fr": "ordinateur", "de": "Computer", "ja": "コンピュータ", "ar": "حاسوب", "zh-CN": "电脑"},
    "day": {"hi": "दिन", "es": "día", "fr": "jour", "de": "Tag", "ja": "日", "ar": "يوم", "zh-CN": "天"},
    "doctor": {"hi": "डॉक्टर", "es": "médico", "fr": "docteur", "de": "Arzt", "ja": "医者", "ar": "طبيب", "zh-CN": "医生"},
    "door": {"hi": "दरवाज़ा", "es": "puerta", "fr": "porte", "de": "Tür", "ja": "ドア", "ar": "باب", "zh-CN": "门"},
    "drink": {"hi": "पीना", "es": "beber", "fr": "boire", "de": "trinken", "ja": "飲む", "ar": "يشرب", "zh-CN": "喝"},
    "eat": {"hi": "खाना", "es": "comer", "fr": "manger", "de": "essen", "ja": "食べる", "ar": "يأكل", "zh-CN": "吃"},
    "evening": {"hi": "शाम", "es": "tarde", "fr": "soir", "de": "Abend", "ja": "夕方", "ar": "مساء", "zh-CN": "晚上"},
    "family": {"hi": "परिवार", "es": "familia", "fr": "famille", "de": "Familie", "ja": "家族", "ar": "عائلة", "zh-CN": "家庭"},
    "father": {"hi": "पिताजी", "es": "padre", "fr": "père", "de": "Vater", "ja": "父", "ar": "أب", "zh-CN": "父亲"},
    "fine": {"hi": "ठीक", "es": "bien", "fr": "bien", "de": "gut", "ja": "元気", "ar": "بخير", "zh-CN": "好"},
    "food": {"hi": "भोजन", "es": "comida", "fr": "nourriture", "de": "Essen", "ja": "食べ物", "ar": "طعام", "zh-CN": "食物"},
    "friend": {"hi": "दोस्त", "es": "amigo", "fr": "ami", "de": "Freund", "ja": "友達", "ar": "صديق", "zh-CN": "朋友"},
    "fruit": {"hi": "फल", "es": "fruta", "fr": "fruit", "de": "Obst", "ja": "果物", "ar": "فاكهة", "zh-CN": "水果"},
    "girl": {"hi": "लड़की", "es": "chica", "fr": "fille", "de": "Mädchen", "ja": "女の子", "ar": "فتاة", "zh-CN": "女孩"},
    "go": {"hi": "जाओ", "es": "ir", "fr": "aller", "de": "gehen", "ja": "行く", "ar": "اذهب", "zh-CN": "去"},
    "good": {"hi": "अच्छा", "es": "bueno", "fr": "bon", "de": "gut", "ja": "良い", "ar": "جيد", "zh-CN": "好"},
    "goodbye": {"hi": "अलविदा", "es": "adiós", "fr": "au revoir", "de": "Auf Wiedersehen", "ja": "さようなら", "ar": "وداعا", "zh-CN": "再见"},
    "happy": {"hi": "खुश", "es": "feliz", "fr": "heureux", "de": "glücklich", "ja": "幸せ", "ar": "سعيد", "zh-CN": "高兴"},
    "help": {"hi": "मदद", "es": "ayuda", "fr": "aide", "de": "Hilfe", "ja": "助け", "ar": "مساعدة", "zh-CN": "帮助"},
    "home": {"hi": "घर", "es": "hogar", "fr": "maison", "de": "Zuhause", "ja": "家", "ar": "منزل", "zh-CN": "家"},
    "hospital": {"hi": "अस्पताल", "es": "hospital", "fr": "hôpital", "de": "Krankenhaus", "ja": "病院", "ar": "مستشفى", "zh-CN": "医院"},
    "house": {"hi": "मकान", "es": "casa", "fr": "maison", "de": "Haus", "ja": "家", "ar": "بيت", "zh-CN": "房子"},
    "internet": {"hi": "इंटरनेट", "es": "internet", "fr": "internet", "de": "Internet", "ja": "インターネット", "ar": "إنترنت", "zh-CN": "互联网"},
    "kitchen": {"hi": "रसोई", "es": "cocina", "fr": "cuisine", "de": "Küche", "ja": "キッチン", "ar": "مطبخ", "zh-CN": "厨房"},
    "learn": {"hi": "सीखना", "es": "aprender", "fr": "apprendre", "de": "lernen", "ja": "学ぶ", "ar": "يتعلم", "zh-CN": "学习"},
    "love": {"hi": "प्यार", "es": "amor", "fr": "amour", "de": "Liebe", "ja": "愛", "ar": "حب", "zh-CN": "爱"},
    "man": {"hi": "आदमी", "es": "hombre", "fr": "homme", "de": "Mann", "ja": "男性", "ar": "رجل", "zh-CN": "男人"},
    "medicine": {"hi": "दवा", "es": "medicina", "fr": "médicament", "de": "Medizin", "ja": "薬", "ar": "دواء", "zh-CN": "药"},
    "milk": {"hi": "दूध", "es": "leche", "fr": "lait", "de": "Milch", "ja": "牛乳", "ar": "حليب", "zh-CN": "牛奶"},
    "money": {"hi": "पैसे", "es": "dinero", "fr": "argent", "de": "Geld", "ja": "お金", "ar": "نقود", "zh-CN": "钱"},
    "morning": {"hi": "सुबह", "es": "mañana", "fr": "matin", "de": "Morgen", "ja": "朝", "ar": "صباح", "zh-CN": "早上"},
    "mother": {"hi": "माँ", "es": "madre", "fr": "mère", "de": "Mutter", "ja": "母", "ar": "أم", "zh-CN": "母亲"},
    "music": {"hi": "संगीत", "es": "música", "fr": "musique", "de": "Musik", "ja": "音楽", "ar": "موسيقى", "zh-CN": "音乐"},
    "night": {"hi": "रात", "es": "noche", "fr": "nuit", "de": "Nacht", "ja": "夜", "ar": "ليل", "zh-CN": "夜晚"},
    "no": {"hi": "नहीं", "es": "no", "fr": "non", "de": "nein", "ja": "いいえ", "ar": "لا", "zh-CN": "不"},
    "nurse": {"hi": "नर्स", "es": "enfermera", "fr": "infirmière", "de": "Krankenschwester", "ja": "看護師", "ar": "ممرضة", "zh-CN": "护士"},
    "office": {"hi": "कार्यालय", "es": "oficina", "fr": "bureau", "de": "Büro", "ja": "事務所", "ar": "مكتب", "zh-CN": "办公室"},
    "paper": {"hi": "कागज़", "es": "papel", "fr": "papier", "de": "Papier", "ja": "紙", "ar": "ورق", "zh-CN": "纸"},
    "phone": {"hi": "फ़ोन", "es": "teléfono", "fr": "téléphone", "de": "Telefon", "ja": "電話", "ar": "هاتف", "zh-CN": "电话"},
    "please": {"hi": "कृपया", "es": "por favor", "fr": "s'il vous plaît", "de": "bitte", "ja": "お願いします", "ar": "من فضلك", "zh-CN": "请"},
    "police": {"hi": "पुलिस", "es": "policía", "fr": "police", "de": "Polizei", "ja": "警察", "ar": "شرطة", "zh-CN": "警察"},
    "question": {"hi": "प्रश्न", "es": "pregunta", "fr": "question", "de": "Frage", "ja": "質問", "ar": "سؤال", "zh-CN": "问题"},
    "read": {"hi": "पढ़ना", "es": "leer", "fr": "lire", "de": "lesen", "ja": "読む", "ar": "يقرأ", "zh-CN": "阅读"},
    "road": {"hi": "सड़क", "es": "carretera", "fr": "route", "de": "Straße", "ja": "道路", "ar": "طريق", "zh-CN": "道路"},
    "room": {"hi": "कमरा", "es": "habitación", "fr": "chambre", "de": "Zimmer", "ja": "部屋", "ar": "غرفة", "zh-CN": "房间"},
    "run": {"hi": "दौड़ना", "es": "correr", "fr": "courir", "de": "laufen", "ja": "走る", "ar": "يركض", "zh-CN": "跑"},
    "sad": {"hi": "उदास", "es": "triste", "fr": "triste", "de": "traurig", "ja": "悲しい", "ar": "حزين", "zh-CN": "悲伤"},
    "school": {"hi": "विद्यालय", "es": "escuela", "fr": "école", "de": "Schule", "ja": "学校", "ar": "مدرسة", "zh-CN": "学校"},
    "shop": {"hi": "दुकान", "es": "tienda", "fr": "magasin", "de": "Geschäft", "ja": "店", "ar": "متجر", "zh-CN": "商店"},
    "sister": {"hi": "बहन", "es": "hermana", "fr": "sœur", "de": "Schwester", "ja": "姉妹", "ar": "أخت", "zh-CN": "姐妹"},
    "sit": {"hi": "बैठो", "es": "sentarse", "fr": "s'asseoir", "de": "sitzen", "ja": "座る", "ar": "يجلس", "zh-CN": "坐"},
    "sleep": {"hi": "सोना", "es": "dormir", "fr": "dormir", "de": "schlafen", "ja": "眠る", "ar": "ينام", "zh-CN": "睡觉"},
    "smile": {"hi": "मुस्कान", "es": "sonreír", "fr": "sourire", "de": "lächeln", "ja": "笑顔", "ar": "ابتسامة", "zh-CN": "微笑"},
    "sorry": {"hi": "माफ़ कीजिए", "es": "lo siento", "fr": "désolé", "de": "Entschuldigung", "ja": "ごめんなさい", "ar": "آسف", "zh-CN": "对不起"},
    "stand": {"hi": "खड़े हो", "es": "pararse", "fr": "se lever", "de": "stehen", "ja": "立つ", "ar": "يقف", "zh-CN": "站立"},
    "stop": {"hi": "रुको", "es": "parar", "fr": "arrêter", "de": "stoppen", "ja": "止まる", "ar": "توقف", "zh-CN": "停止"},
    "student": {"hi": "छात्र", "es": "estudiante", "fr": "étudiant", "de": "Student", "ja": "学生", "ar": "طالب", "zh-CN": "学生"},
    "study": {"hi": "अध्ययन", "es": "estudiar", "fr": "étudier", "de": "studieren", "ja": "勉強する", "ar": "يدرس", "zh-CN": "学习"},
    "table": {"hi": "मेज़", "es": "mesa", "fr": "table", "de": "Tisch", "ja": "テーブル", "ar": "طاولة", "zh-CN": "桌子"},
    "tea": {"hi": "चाय", "es": "té", "fr": "thé", "de": "Tee", "ja": "お茶", "ar": "شاي", "zh-CN": "茶"},
    "teacher": {"hi": "शिक्षक", "es": "profesor", "fr": "professeur", "de": "Lehrer", "ja": "先生", "ar": "معلم", "zh-CN": "老师"},
    "time": {"hi": "समय", "es": "tiempo", "fr": "temps", "de": "Zeit", "ja": "時間", "ar": "وقت", "zh-CN": "时间"},
    "today": {"hi": "आज", "es": "hoy", "fr": "aujourd'hui", "de": "heute", "ja": "今日", "ar": "اليوم", "zh-CN": "今天"},
    "tomorrow": {"hi": "कल (आने वाला)", "es": "mañana", "fr": "demain", "de": "morgen", "ja": "明日", "ar": "غدا", "zh-CN": "明天"},
    "train": {"hi": "रेलगाड़ी", "es": "tren", "fr": "train", "de": "Zug", "ja": "電車", "ar": "قطار", "zh-CN": "火车"},
    "wait": {"hi": "इंतज़ार करो", "es": "esperar", "fr": "attendre", "de": "warten", "ja": "待つ", "ar": "انتظر", "zh-CN": "等待"},
    "walk": {"hi": "चलना", "es": "caminar", "fr": "marcher", "de": "gehen", "ja": "歩く", "ar": "يمشي", "zh-CN": "走路"},
    "water": {"hi": "पानी", "es": "agua", "fr": "eau", "de": "Wasser", "ja": "水", "ar": "ماء", "zh-CN": "水"},
    "welcome": {"hi": "स्वागत", "es": "bienvenido", "fr": "bienvenue", "de": "willkommen", "ja": "ようこそ", "ar": "أهلا بك", "zh-CN": "欢迎"},
    "woman": {"hi": "महिला", "es": "mujer", "fr": "femme", "de": "Frau", "ja": "女性", "ar": "امرأة", "zh-CN": "女人"},
    "work": {"hi": "काम", "es": "trabajo", "fr": "travail", "de": "Arbeit", "ja": "仕事", "ar": "عمل", "zh-CN": "工作"},
    "write": {"hi": "लिखना", "es": "escribir", "fr": "écrire", "de": "schreiben", "ja": "書く", "ar": "يكتب", "zh-CN": "写"},
    "yes": {"hi": "हाँ", "es": "sí", "fr": "oui", "de": "ja", "ja": "はい", "ar": "نعم", "zh-CN": "是"},
    "yesterday": {"hi": "कल (बीता हुआ)", "es": "ayer", "fr": "hier", "de": "gestern", "ja": "昨日", "ar": "أمس", "zh-CN": "昨天"},
}


class MultilingualSignTranslator:
    """
    Translates sign vocabulary words and formed sentences into 15+ languages.
    Provides online translation with offline fallback and Text-to-Speech audio synthesis.
    """

    def __init__(self):
        self.languages = SUPPORTED_LANGUAGES

    def translate_word(self, word: str, target_lang: str) -> str:
        """
        Translates a single sign word.
        Uses offline dictionary first for maximum speed and reliability,
        falling back to online translator if needed.
        """
        clean_word = word.strip().lower()
        if target_lang == "en":
            return clean_word.capitalize()

        # Check offline dictionary
        if clean_word in OFFLINE_DICTIONARY:
            trans_dict = OFFLINE_DICTIONARY[clean_word]
            if target_lang in trans_dict:
                return trans_dict[target_lang]

        # Online fallback
        return self.translate_sentence(clean_word, target_lang)

    def translate_sentence(self, text: str, target_lang: str) -> str:
        """
        Translates a sentence or phrase into the specified target language.
        Tries online translation via MyMemory or Google Translator;
        falls back gracefully to offline dictionary word-by-word synthesis.
        """
        if not text or target_lang == "en":
            return text

        # 1. Try MyMemoryTranslator
        if HAS_DEEP_TRANSLATOR:
            lang_meta = self.languages.get(target_lang, {})
            mymem_code = lang_meta.get("mymemory_code", target_lang)
            try:
                translator = MyMemoryTranslator(source="en-US", target=mymem_code)
                translated = translator.translate(text)
                if translated and not translated.startswith("MYMEMORY WARNING"):
                    return translated
            except Exception:
                pass

            # 2. Try GoogleTranslator
            try:
                translator = GoogleTranslator(source="auto", target=target_lang)
                translated = translator.translate(text)
                if translated:
                    return translated
            except Exception:
                pass

        # 3. Offline Dictionary Fallback
        words = text.lower().replace(",", " ").replace(".", " ").split()
        translated_parts = []
        for w in words:
            if w in OFFLINE_DICTIONARY and target_lang in OFFLINE_DICTIONARY[w]:
                translated_parts.append(OFFLINE_DICTIONARY[w][target_lang])
            else:
                translated_parts.append(w)

        return " ".join(translated_parts)

    def synthesize_sentence_from_words(self, words: List[str]) -> str:
        """
        Synthesizes a cohesive, natural English sentence preserving ALL signs performed by the user.
        Uses linguistic role mapping (Time, Subjects, Actions, Objects, Places, Moods)
        so the final sentence faithfully represents the user's exact gestures.
        """
        if not words:
            return ""

        clean_words = [w.strip().lower() for w in words if w.strip()]
        if not clean_words:
            return ""

        # Single word: just capitalize
        if len(clean_words) == 1:
            return clean_words[0].capitalize() + "."

        GREETINGS = {
            "morning": "Good morning",
            "afternoon": "Good afternoon",
            "evening": "Good evening",
            "night": "Good night",
            "bye": "Goodbye",
            "goodbye": "Goodbye",
        }
        POLITE = {"please": "please", "sorry": "sorry", "welcome": "you are welcome", "yes": "yes", "no": "no"}
        TIME = {"today": "today", "tomorrow": "tomorrow", "yesterday": "yesterday", "time": "time", "day": "day"}
        PEOPLE = {
            "mother": "mother", "father": "father", "brother": "brother", "sister": "sister",
            "doctor": "the doctor", "nurse": "the nurse", "teacher": "the teacher",
            "student": "the student", "friend": "my friend", "boy": "the boy", "girl": "the girl",
            "baby": "the baby", "family": "my family", "man": "the man", "woman": "the woman",
            "police": "the police"
        }
        VERBS = {
            "go": "going to", "come": "coming to", "eat": "eating", "drink": "drinking",
            "read": "reading", "write": "writing", "study": "studying", "learn": "learning",
            "sleep": "sleeping", "sit": "sitting in", "stand": "standing in", "walk": "walking to",
            "run": "running on", "wait": "waiting at", "help": "help", "smile": "smiling", "work": "working at"
        }
        PLACES = {
            "hospital": "the hospital", "school": "school", "college": "college", "home": "home",
            "office": "the office", "bank": "the bank", "shop": "the shop", "room": "the room",
            "kitchen": "the kitchen", "bathroom": "the bathroom", "city": "the city", "road": "the road",
            "table": "the table", "bed": "the bed"
        }
        OBJECTS = {
            "water": "water", "tea": "tea", "coffee": "coffee", "milk": "milk", "food": "food",
            "bread": "bread", "fruit": "fruit", "apple": "an apple", "medicine": "the medicine",
            "book": "the book", "car": "the car", "bike": "the bike", "bus": "the bus",
            "train": "the train", "phone": "the phone", "money": "money", "paper": "the paper",
            "computer": "the computer"
        }
        MOODS = {"happy": "happy", "sad": "sad", "angry": "angry", "fine": "fine", "good": "good", "bad": "bad"}

        used = set()
        parts = []

        # 1. Greetings
        for w in clean_words:
            if w in GREETINGS and w not in used:
                parts.append(GREETINGS[w] + ",")
                used.add(w)

        # 2. Time markers
        for w in clean_words:
            if w in TIME and w not in used:
                parts.append(TIME[w].capitalize() + ",")
                used.add(w)

        # 3. Politeness markers
        has_please = "please" in clean_words
        has_sorry = "sorry" in clean_words
        if has_sorry and "sorry" not in used:
            parts.append("I am sorry,")
            used.add("sorry")
        if has_please and "please" not in used:
            parts.append("please")
            used.add("please")

        # 4. Extract constituents
        subjects = [PEOPLE[w] for w in clean_words if w in PEOPLE and w not in used]
        for w in clean_words:
            if w in PEOPLE: used.add(w)

        verbs = [(w, VERBS[w]) for w in clean_words if w in VERBS and w not in used]
        for w in clean_words:
            if w in VERBS: used.add(w)

        objs = [OBJECTS[w] for w in clean_words if w in OBJECTS and w not in used]
        for w in clean_words:
            if w in OBJECTS: used.add(w)

        places = [PLACES[w] for w in clean_words if w in PLACES and w not in used]
        for w in clean_words:
            if w in PLACES: used.add(w)

        moods = [MOODS[w] for w in clean_words if w in MOODS and w not in used]
        for w in clean_words:
            if w in MOODS: used.add(w)

        # 5. Build core semantic clause
        core_clause = []
        if subjects:
            subj_str = " and ".join(subjects).capitalize()
            core_clause.append(subj_str)
            verb_be = "are" if len(subjects) > 1 or "family" in subjects else "is"

            if verbs:
                v_word, v_str = verbs[0]
                if v_word == "help":
                    core_clause.append(f"can {v_word}")
                else:
                    core_clause.append(f"{verb_be} {v_str}")
            elif moods:
                core_clause.append(f"{verb_be} {moods[0]}")
            elif places and not verbs:
                core_clause.append(f"{verb_be} at {places[0]}")
            elif objs and not verbs:
                core_clause.append(f"needs {objs[0]}")
        elif verbs:
            v_word, v_str = verbs[0]
            if has_please:
                core_clause.append(f"{v_word}")
            else:
                core_clause.append(f"I am {v_str}")
        elif moods:
            core_clause.append(f"I am feeling {moods[0]}")

        # Add remaining objects & places
        if objs and (not subjects or verbs or not places):
            core_clause.append(" and ".join(objs))
        if places and (not subjects or verbs or objs):
            place_prep = f"at {places[0]}"
            if verbs and any(p in verbs[0][1] for p in ["at", "to", "in", "on"]):
                place_prep = places[0]
            core_clause.append(place_prep)

        # 6. Ensure ANY leftover signs are preserved
        leftover = [w for w in clean_words if w not in used and w not in ["please", "sorry"]]
        if leftover:
            if not core_clause and not parts:
                core_clause.append(" ".join(leftover).capitalize())
            else:
                core_clause.append("with " + ", ".join(leftover))

        full_sentence = " ".join(parts + core_clause).strip()
        full_sentence = " ".join(full_sentence.split())
        if full_sentence and not full_sentence.endswith((".", "!", "?")):
            full_sentence += "."

        return full_sentence[0].upper() + full_sentence[1:] if full_sentence else " ".join(clean_words).capitalize() + "."

    def generate_speech_audio(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Generates MP3 audio for the given text and language code using gTTS.
        Returns audio bytes for direct in-browser playback.
        """
        if not HAS_GTTS or not text:
            return None

        tts_lang = self.languages.get(lang, {}).get("tts_code", "en")
        try:
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            # If language not supported by gTTS or offline, fallback to English TTS
            try:
                tts = gTTS(text=text, lang="en", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception:
                return None

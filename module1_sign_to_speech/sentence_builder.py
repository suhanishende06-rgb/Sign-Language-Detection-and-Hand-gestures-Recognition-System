
import time
from collections import deque, Counter


WORD_PHRASES = {
    "HELLO": (
        "Hello.",
        "How are you?",
        "What are you doing?",
        "I am happy to see you.",
    ),
    "PLEASE": (
        "Please help me.",
        "Could you please help me?",
        "Please listen to me.",
        "I would really appreciate your help.",
    ),
    "SORRY": (
        "I am sorry.",
        "It was my mistake.",
        "Please forgive me.",
        "I will try not to repeat it.",
    ),
    "THANK_YOU": (
        "Thank you.",
        "Thank you for your help.",
        "I really appreciate it.",
        "Your kindness means a lot to me.",
    ),
}


class SentenceBuilder:
    def __init__(self, stable_frames=15, cooldown=1.5):
        self.predictions = deque(maxlen=stable_frames)

        self.cooldown = cooldown
        self.last_added_time = 0

        self.current_word = ""
        self.sentence = ""
        self.sentence_parts = []

        self.sign_locked = False
        self.no_hand_start = None

    def _normalize_word_key(self, word):
        return "".join(character for character in word.upper() if character.isalnum())

    def _normalize_prediction(self, prediction):
        if prediction is None:
            return ""
        return str(prediction).strip()

    def update(self, prediction):
        prediction = self._normalize_prediction(prediction)
        if not prediction:
            return None

        self.predictions.append(prediction)

        if len(self.predictions) < self.predictions.maxlen:
            return None

        counts = Counter(self.predictions)
        majority_prediction, count = counts.most_common(1)[0]

        required_count = max(2, self.predictions.maxlen // 3)
        if count < required_count:
            return None

        current_time = time.time()

        if self.sign_locked:
            return None

        if current_time - self.last_added_time < self.cooldown:
            return None

        word = majority_prediction.strip()
        if not word:
            self.predictions.clear()
            return None

        self.current_word = word.upper()
        self.last_added_time = current_time
        self.sign_locked = True
        self.predictions.clear()

        return self.current_word

    def no_hand_detected(self):
        if self.no_hand_start is None:
            self.no_hand_start = time.time()

        if time.time() - self.no_hand_start >= 1.0:
            self.sign_locked = False
            self.predictions.clear()
            self.no_hand_start = None

    def hand_detected(self):
        self.no_hand_start = None

    def add_space(self):
        if self.current_word:
            word_key = self._normalize_word_key(self.current_word.strip())
            phrases = next(
                (
                    value
                    for key, value in WORD_PHRASES.items()
                    if self._normalize_word_key(key) == word_key
                ),
                None,
            )
            if phrases is None:
                readable_word = self.current_word.strip().replace("_", " ").capitalize()
                phrases = (f"{readable_word}.",)

            self.sentence_parts.append(" ".join(phrases))
            self.sentence = " ".join(self.sentence_parts)
            self.current_word = ""

        self.sign_locked = False
        self.predictions.clear()

    def clear(self):
        self.current_word = ""
        self.sentence = ""
        self.sentence_parts.clear()
        self.predictions.clear()

        self.sign_locked = False
        self.no_hand_start = None

    def undo(self):
        if self.current_word:
            self.current_word = ""
            self.sign_locked = False
            self.predictions.clear()
            return self.get_text()

        if self.sentence_parts:
            self.sentence_parts.pop()
            self.sentence = " ".join(self.sentence_parts)

        self.sign_locked = False
        self.predictions.clear()
        return self.get_text()

    def get_text(self):
        if self.current_word:
            return (self.sentence + " " + self.current_word).strip()
        return self.sentence.strip()

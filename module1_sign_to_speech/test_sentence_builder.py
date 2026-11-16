
from sentence_builder import SentenceBuilder


def add_recognized_word(builder, word):
    for _ in range(builder.predictions.maxlen):
        result = builder.update(word)
    assert result == word
    builder.add_space()


builder = SentenceBuilder(stable_frames=15, cooldown=0)

add_recognized_word(builder, "HELLO")
assert builder.get_text() == (
    "Hello. How are you? What are you doing? I am happy to see you."
)

add_recognized_word(builder, "SORRY")
assert builder.get_text() == (
    "Hello. How are you? What are you doing? I am happy to see you. "
    "I am sorry. It was my mistake. Please forgive me. "
    "I will try not to repeat it."
)

builder.undo()
assert builder.get_text() == (
    "Hello. How are you? What are you doing? I am happy to see you."
)

builder.clear()
for word, expected in (
    (
        "PLEASE",
        "Please help me. Could you please help me? Please listen to me. "
        "I would really appreciate your help.",
    ),
    (
        "THANK_YOU",
        "Thank you. Thank you for your help. I really appreciate it. "
        "Your kindness means a lot to me.",
    ),
):
    builder.clear()
    add_recognized_word(builder, word)
    assert builder.get_text() == expected

for label in ("THANK YOU", "THANK-YOU"):
    builder.clear()
    builder.current_word = label
    builder.add_space()
    assert builder.get_text() == (
        "Thank you. Thank you for your help. I really appreciate it. "
        "Your kindness means a lot to me."
    )

builder.clear()
add_recognized_word(builder, "NEW_WORD")
assert builder.get_text().endswith("New word.")

print("\nSentence Builder Test Completed!")

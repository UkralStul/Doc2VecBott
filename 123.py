from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Создайте функцию для разделения текста на слова
def tokenize_text(text):
    return text.lower().split()

# Создайте список фраз с соответствующими метками
phrases = [
    ("Привет!", "фразаПриветствие"),
    ('Здравствуй!', 'фразаПриветствие'),
    ("Какая сегодня погода?", "фразаПогода"),
    ("Кто придет на вечеринку?", "фразаКто"),
    ("До свидания! До скорой встречи.", "фразаПрощание"),
    # Добавьте другие фразы и категории
]

# Создайте объекты TaggedDocument
tagged_data = [TaggedDocument(words=tokenize_text(text), tags=[category]) for text, category in phrases]
print(tagged_data)
# Создайте модель Doc2Vec и обучите ее
model = Doc2Vec(vector_size=20, window=2, min_count=1, dm=1, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Теперь у вас есть обученная модель Doc2Vec, которая создает векторы для фраз

# Для классификации фраз используйте обученную модель
def classify_phrase(phrase):
    vec = model.infer_vector(tokenize_text(phrase))
    return model.dv.most_similar([vec], topn=1)[0][0]

# Тестируйте модель
test_phrases = [
    "Привет, как дела?",
    "Какова погода сегодня?",
    "Кто это?",
    "До свидания!",
]

for phrase in test_phrases:
    category = classify_phrase(phrase)
    print(f'Фраза: "{phrase}" - Категория: {category}')

# Можно дополнить тестовые фразы и категории, а также настроить параметры модели Doc2Vec для лучшей производительности.

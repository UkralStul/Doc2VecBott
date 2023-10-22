from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data = []

with open("train_data.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

for line in lines:
    words_in_line = line.strip().split()
    if len(words_in_line) > 0:
        first_word = words_in_line[0]
        rest_of_words = " ".join(words_in_line[1:]).split()
        tagged_data.append(TaggedDocument(words=rest_of_words, tags=[first_word]))

print(tagged_data)

model = Doc2Vec(vector_size=50, window=10, min_count=2, workers=4, epochs=10,sample=1e-4)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("doc2vec_model")
from src.interface import get_classifier


classifier = get_classifier()

label_dict = {
    'tech': ['coding', 'programming', 'computer', 'python', 'development', 'tech'],
    'nature': ['mountain', 'nature', 'sea', 'green', 'jungle', 'animals'],
    'politics': ['policy', 'world', 'nation', 'war', 'negotiate'],
    'health': ['health', 'vaccine', 'illness', 'patient', 'doctor']
}

classifier.add_labels(label_dict)

# sample text for tech/nature
print(classifier.predict(
    "Coding in mountains feels very good! I usually have a very clear mind when in nature."
))

# sample text for tech
print(classifier.predict(
    "AI is becoming more powerful each day! I wonder how will this affect the world!",
    get_similarities=True
))

# sample text for politics
print(classifier.predict(
    "‘It’s a con’: Labour amendment to put Sunak’s migrant bill under fresh scrutiny",
    get_similarities=True
))

# sample text for health
print(classifier.predict(
    "Exposure to other people’s sweat could help reduce social anxiety",
    get_similarities=True
))

# sample text for nature/politics
print(classifier.predict(
    "As the UN meets, make water central to climate action",
    get_similarities=True
))

from src.models.t5 import QuestionGenerator, preprocess_question_answer


que_generator = QuestionGenerator()

def get_qa_pairs_from_text_corpus(text):
    q_a = que_generator.generate(text)
    return preprocess_question_answer(q_a)

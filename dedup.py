import hashlib
from typing import Dict, List, Tuple

class QADedupSystem:
    def __init__(self):
        self.qa_pairs: Dict[str, Tuple[str, str]] = {}

    def _hash_qa(self, question: str, answer: str) -> str:
        """Generate a hash for a question-answer pair."""
        qa_string = f"{question.lower().strip()}|{answer.lower().strip()}"
        return hashlib.md5(qa_string.encode()).hexdigest()

    def add_qa_pair(self, question: str, answer: str,similarity_threshold: float = 0.8) -> bool:
        """
        Add a new QA pair if it doesn't exist.
        Returns True if added, False if it already exists.
        """
        qa_hash = self._hash_qa(question, answer)
        
        # Check for exact duplicates
        if qa_hash in self.qa_pairs:
            return False
        
        # Check for similar questions
        similar_questions = self.search_similar_questions(question, similarity_threshold)
        if similar_questions:
            return False
        
        # If no duplicates or similar questions found, add the new pair
        self.qa_pairs[qa_hash] = (question, answer)
        return True

    def get_all_qa_pairs(self) -> List[Tuple[str, str]]:
        """Return all stored QA pairs."""
        return list(self.qa_pairs.values())

    def search_similar_questions(self, query: str, threshold: float = 0.4) -> List[Tuple[str, str]]:
        """
        Search for similar questions using a simple word overlap method.
        Returns a list of (question, answer) tuples that meet the similarity threshold.
        """
        query_words = set(query.lower().split())
        similar_pairs = []

        for _, (question, answer) in self.qa_pairs.items():
            question_words = set(question.lower().split())
            overlap = len(query_words.intersection(question_words))
            max_length = max(len(query_words), len(question_words))
            similarity = overlap / max_length

            if similarity >= threshold:
                similar_pairs.append((question, answer))

        return similar_pairs

# Extended example usage
if __name__ == "__main__":
    dedup_system = QADedupSystem()

    # Add some QA pairs
    print(dedup_system.add_qa_pair("What is Python?", "Python is a high-level, interpreted programming language."))  # True
    print(dedup_system.add_qa_pair("What is Python?", "Python is a high-level, interpreted programming language."))  # False (duplicate)
    print(dedup_system.add_qa_pair("How do I install Python?", "You can download Python from python.org and follow the installation instructions."))  # True
    print(dedup_system.add_qa_pair("How do I setup Python?", "You can download Python from python.org and follow the installation instructions."))  #False(duplicate)
    print(dedup_system.add_qa_pair("What are Python's main features?", "Python features include simplicity, readability, and a large standard library."))  # True
    print(dedup_system.add_qa_pair("Is Python object-oriented?", "Yes, Python supports object-oriented programming paradigms."))  # True
    print(dedup_system.add_qa_pair("What is a Python list?", "A Python list is a mutable, ordered collection of elements."))  # True
    print(dedup_system.add_qa_pair("How do I create a virtual environment in Python?", "Use the 'venv' module to create a virtual environment in Python."))  # True
    print(dedup_system.add_qa_pair("What is PIP in Python?", "PIP is the package installer for Python, used to install and manage additional libraries."))  # True

    # Search for similar questions
    similar = dedup_system.search_similar_questions("Tell me about Python programming")
    print("\nSimilar questions for 'Tell me about Python programming':")
    for question, answer in similar:
        print(f"Q: {question}\nA: {answer}\n")

    # Another similarity search
    similar = dedup_system.search_similar_questions("How to set up Python?")
    print("\nSimilar questions for 'How to set up Python?':")
    for question, answer in similar:
        print(f"Q: {question}\nA: {answer}\n")

    # Get all QA pairs
    all_pairs = dedup_system.get_all_qa_pairs()
    print("All QA pairs:")
    for question, answer in all_pairs:
        print(f"Q: {question}\nA: {answer}\n")
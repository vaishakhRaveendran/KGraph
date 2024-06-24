import hashlib
from typing import Dict, List, Tuple

class QADedupSystem:
    def __init__(self):
        self.qa_pairs: Dict[str, Tuple[str, str]] = {}

    def _hash_qa(self, question: str, answer: str) -> str:
        """Generate a hash for a question-answer pair."""
        qa_string = f"{question.lower().strip()}|{answer.lower().strip()}"
        return hashlib.md5(qa_string.encode()).hexdigest()

    def add_qa_pair(self, question: str, answer: str) -> bool:
        """
        Add a new QA pair if it doesn't exist.
        Returns True if added, False if it already exists.
        """
        qa_hash = self._hash_qa(question, answer)
        if qa_hash not in self.qa_pairs:
            self.qa_pairs[qa_hash] = (question, answer)
            return True
        return False

    def get_all_qa_pairs(self) -> List[Tuple[str, str]]:
        """Return all stored QA pairs."""
        return list(self.qa_pairs.values())

    def search_similar_questions(self, query: str, threshold: float = 0.8) -> List[Tuple[str, str]]:
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

# Example usage
if __name__ == "__main__":
    dedup_system = QADedupSystem()

    # Add some QA pairs
    print(dedup_system.add_qa_pair("What is Python?", "Python is a programming language."))  # True
    print(dedup_system.add_qa_pair("What is Python?", "Python is a programming language."))  # False (duplicate)
    print(dedup_system.add_qa_pair("How do I install Python?", "You can download Python from python.org."))  # True

    # Search for similar questions
    similar = dedup_system.search_similar_questions("Tell me about Python")
    for question, answer in similar:
        print(f"Q: {question}\nA: {answer}\n")

    # Get all QA pairs
    all_pairs = dedup_system.get_all_qa_pairs()
    print("All QA pairs:")
    for question, answer in all_pairs:
        print(f"Q: {question}\nA: {answer}\n")
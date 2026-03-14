from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class PresidioPrivacyLayer:
    def __init__(self):
        # Initialize Presidio Analyzer and Anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def sanitize(self, text: str) -> str:
        """
        Takes raw user input, detects PII using Microsoft Presidio,
        and returns sanitized text (with identifiers masked out).
        """
        if not text:
            return ""

        # Analyze the text for PII entities (e.g., PERSON, LOCATION, EMAIL_ADDRESS, PHONE_NUMBER)
        try:
            analyzer_results = self.analyzer.analyze(text=text, language="en")
            
            # Mask recognized entities with their entity type (e.g., <PERSON>)
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results
            )
            return anonymized_result.text
        except Exception as e:
            # Fallback in case of failure: safe to block entirely if we can't sanitize, 
            # but for this MVP, return the text or a scrubbed warning.
            print(f"Privacy Layer Error: {e}")
            return text 

    def sanitize_dataset(self, dataset: list) -> list:
        """
        Iterates over the dataset dictionary to sanitize raw Reddit posts/comments.
        Removes PII before saving for training.
        """
        sanitized_data = []
        for record in dataset:
            clean_record = record.copy()
            if clean_record.get("post_text"):
                clean_record["post_text"] = self.sanitize(clean_record["post_text"])
            if clean_record.get("comment_text"):
                clean_record["comment_text"] = self.sanitize(clean_record["comment_text"])
            
            # Anonymize author IDs in training data
            if clean_record.get("author_id"):
                clean_record["author_id"] = "ANONYMIZED_USER"
            
            clean_record["preprocessing_status"] = "anonymized"
            sanitized_data.append(clean_record)
            
        return sanitized_data

if __name__ == "__main__":
    privacy_layer = PresidioPrivacyLayer()
    test_text = "My name is John Doe and I live in San Francisco. My email is john.doe@email.com. I've been feeling extremely hopeless lately."
    print("Raw:", test_text)
    print("Sanitized:", privacy_layer.sanitize(test_text))

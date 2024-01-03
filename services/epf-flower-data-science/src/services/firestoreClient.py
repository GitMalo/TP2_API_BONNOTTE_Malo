from google.cloud import firestore
from google.oauth2 import service_account

class FirestoreClient:

    client: firestore.Client

    def __init__(self) -> None:

        credentials = service_account.Credentials.from_service_account_file("src/config/bonnotte-malo-tp-api-8308794e4640.json")

        self.client = firestore.Client(credentials=credentials)

    def get(self, collection_name: str, document_id: str) -> dict:

        doc = self.client.collection(
            collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileExistsError(
            f"No document found at {collection_name} with the id {document_id}"
        )
    
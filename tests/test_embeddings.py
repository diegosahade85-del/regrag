from regrag.embeddings import Embedder


class RecordingClient:
    """Stands in for voyageai.Client, recording how we call it.

    The API call itself is exercised by a live smoke test; what needs covering
    here is our own batching and input_type wiring, which fail silently.
    """

    def __init__(self):
        self.calls = []

    def embed(self, texts, model, input_type, **kwargs):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type}
        )
        return type("R", (), {"embeddings": [[float(len(t))] for t in texts]})()


def test_documents_are_embedded_with_the_document_input_type():
    client = RecordingClient()

    Embedder(client).embed_documents(["uno", "dos"])

    assert client.calls[0]["input_type"] == "document"


def test_queries_are_embedded_with_the_query_input_type():
    client = RecordingClient()

    Embedder(client).embed_query("¿qué exige el artículo 4?")

    assert client.calls[0]["input_type"] == "query"


def test_embed_query_returns_a_single_vector_not_a_list_of_one():
    embedder = Embedder(RecordingClient())

    assert embedder.embed_query("hola") == [4.0]


def test_documents_are_sent_in_batches_within_the_api_limit():
    client = RecordingClient()

    Embedder(client, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert [len(call["texts"]) for call in client.calls] == [2, 2, 1]


def test_batching_preserves_document_order():
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = Embedder(RecordingClient(), batch_size=2).embed_documents(texts)

    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_embedding_no_documents_makes_no_api_calls():
    client = RecordingClient()

    assert Embedder(client).embed_documents([]) == []
    assert client.calls == []


def test_the_configured_model_is_passed_through():
    client = RecordingClient()

    Embedder(client, model="voyage-3-large").embed_documents(["uno"])

    assert client.calls[0]["model"] == "voyage-3-large"

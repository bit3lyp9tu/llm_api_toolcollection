from pymilvus import CollectionSchema, DataType, MilvusClient

from llm_api_toolcollection.config_parser import YAMLConfig
from llm_api_toolcollection.async_api import embed_batch
from llm_api_toolcollection.schemas.config_schema import API


class VDB:
    def __init__(self, api_config_path: str,  db_name: str, dimensions: int = 2560) -> None:
        print("[WARN] No support for matryoshka representation; No support for custom vector sizes")
        self._dimensions = dimensions
        self._client: MilvusClient = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus",
        )
        self._db_name = db_name
        if db_name not in self._client.list_databases():
            self._client.create_database(db_name)

        self.api_config = YAMLConfig(API, api_config_path).config

    def create_collection(self, name, schema: CollectionSchema):
        if self._client.has_collection(name):
            raise Exception(f"[ERROR] Collection [{name}] already exists.")

        index_params = self._client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params
        )

    async def insert_vectors(self, collection_name, texts=None, extra_data=None):
        if texts is None:
            texts = []
        if extra_data is None:
            extra_data = []

        self._validate_collection(collection_name)

        if len(texts) != len(extra_data):
            raise ValueError(
                "texts and extra_data have to be of equal sizes"
            )

        vectors = await embed_batch(self.api_config, texts, dimensions=self._dimensions)

        data = [
            {
                "vector": vectors[i],
                "text": texts[i],
                **extra_data[i],
            }
            for i in range(len(texts))
        ]

        return self._client.insert(
            collection_name=collection_name,
            data=data,
        )

    async def query(self, collection_name, text, limit=5):
        self._validate_collection(collection_name)

        vectors = await embed_batch(self.api_config, text, dimensions=self._dimensions)

        res = self._client.search(
            collection_name=collection_name,
            data=vectors,
            limit=limit,
            output_fields=["text", "vers"],
        )
        return res

    def _validate_collection(self, collection_name):
        if not self._client.has_collection(collection_name):
            raise ValueError(f"Collection [{collection_name}] not found")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from kg_builder.builder import KGBuilder
from kg_builder.cfg import SplitConfig
from kg_builder.engine import ChatEngine
from kg_builder.extractor import RelationsExtractor
from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting
from kg_builder.relations import SearchableRelations, RelationsData


class EBWorkflow:
    """
    Extract and Build agentic workflow for Knowledge Graph creation.
    Args:
        embeddings: HuggingFaceEmbeddings instance from 'langchain_huggingface' for similarity based triplets retrieval.
        split_cfg: configuration for splitting documents into passages.
        extractor_prompting: ExtractorPrompting instance containing prompt templates for relation extraction.
        builder_prompting: BuilderPrompting instance containing prompt templates for knowledge graph building.

    Attributes:
        builder_prompting: BuilderPrompting instance containing prompt templates for knowledge graph building.
        extractor_prompting: ExtractorPrompting instance containing prompt templates for relation extraction.
        embeddings: HuggingFaceEmbeddings instance from 'langchain_huggingface' for similarity based triplets retrieval.
        split_cfg: configuration for splitting documents into passages.
    """

    def __init__(self,
                 embeddings: HuggingFaceEmbeddings,
                 split_cfg: SplitConfig = SplitConfig(),
                 extractor_prompting: ExtractorPrompting = ExtractorPrompting(),
                 builder_prompting: BuilderPrompting = BuilderPrompting()):
        self.builder_prompting = builder_prompting
        self.extractor_prompting = extractor_prompting
        self.embeddings = embeddings
        self.split_cfg = split_cfg

    def split_text(self, text: str) -> list[str]:
        """
        Split the given text into passages.
        Args:
            text: the document to split.
        Returns:
            The list of passages.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.split_cfg.chunk_char_size,
                                                  chunk_overlap=self.split_cfg.chunk_char_overlap,
                                                  separators=self.split_cfg.separators)
        return splitter.split_text(text)

    def __call__(self,
                 text: str,
                 relations_data: RelationsData,
                 builder_engine: ChatEngine,
                 extractor_engine: ChatEngine,
                 delete_models_after_use: bool = True,
                 **kwargs) -> RelationsData:
        """
        Expand a Knowledge Graph from the given text. Add to the already existing Knowledge Graph contained in
        'relations_data'.

        Args:
            text: the text to extract relations from.
            relations_data: the preexisting extracted relations in the Knowledge Graph.
            builder_engine: engine model to use for building the Knowledge Graph.
            extractor_engine: engine model to use for extracting relations from the text.
            delete_models_after_use: whether to delete the models from memory after use.
            **kwargs: engine generation kwargs.
        Returns:
                The updated RelationsData object containing the extracted relations.
        """
        searchable_triplets = SearchableRelations(self.embeddings, relations=relations_data)
        extractor = RelationsExtractor(engine=extractor_engine)

        chunks = self.split_text(text)
        extracted_triplets = []
        for chunk in tqdm(chunks, desc="Extracting relations..."):
            triplets = extractor.extract_from_passage(chunk,
                                                      allowed_types=relations_data.allowed_entity_types,
                                                      **kwargs)
            extracted_triplets.append(triplets)
        if delete_models_after_use:
            extractor_engine.clear_caches()

        builder = KGBuilder(engine=builder_engine,
                            relations_data=searchable_triplets)
        for i, c in enumerate(tqdm(chunks, desc="Adding relations to the Knowledge Graph...")):
            builder.build_kg(extracted_triplets[i], passage=c, **kwargs)
        if delete_models_after_use:
            builder_engine.clear_caches()
        return searchable_triplets.relations_data

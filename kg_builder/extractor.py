from kg_builder.engine import ChatEngine


class RelationsExtractor:

    def __init__(self,
                 engine: ChatEngine):
        self.engine = engine

    def extract_from_passage(self,
                             passage: str,
                             allowed_types: list[str],
                             max_new_tokens: int = 4096,
                             **kwargs) -> list[tuple[str, str, str]]:
        messages = self.engine.prompting.get_messages(passage, allowed_types=allowed_types)
        decoded = self.engine.chat_completion(messages=messages,
                                              max_new_tokens=max_new_tokens,
                                              **kwargs)
        return self.engine.prompting.format_triplets(decoded, allowed_types=allowed_types)

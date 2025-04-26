from kg_builder.engine import ChatEngine
from kg_builder.prompts.prompting import BuilderPrompting
from kg_builder.relations import SearchableRelations


class KGBuilder:
    prompting: BuilderPrompting

    def __init__(self,
                 engine: ChatEngine,
                 relations_data: SearchableRelations,
                 ):
        self.engine = engine
        self.relations_data = relations_data

    def decision_from_messages(self,
                               messages: list[dict[str, str]],
                               max_new_tokens: int = 1024,
                               **kwargs) -> tuple[tuple[str, str, str] | str | None, str]:
        decoded = self.engine.chat_completion(messages=messages, max_new_tokens=max_new_tokens, **kwargs)
        allowed_types = self.relations_data.relations_data.allowed_entity_types
        decision = self.engine.prompting.extract_new_triplet(decoded, allowed_entity_types=allowed_types)
        return decision, decoded

    def validate_relation(self,
                          proposed_relation: tuple[str, str, str],
                          similar_relations: list[str],
                          max_new_tokens: int = 512,
                          temperature: float = 0.,
                          **kwargs) -> tuple[tuple[str, str, str] | str | None, str]:
        messages = self.engine.prompting.get_messages(proposed_relation=proposed_relation,
                                                      existing_relations=similar_relations,
                                                      allowed_types=self.relations_data.relations_data.allowed_entity_types)
        new_triplet, decoded = self.decision_from_messages(messages, max_new_tokens, **kwargs)
        return new_triplet, decoded

    def retry_validation(self,
                         proposed_relation: tuple[str, str, str],
                         error: str,
                         previous_generation: str,
                         similar_relations: list[str],
                         max_new_tokens: int = 1024,
                         **kwargs) -> tuple[tuple[str, str, str] | str | None, str]:
        messages = self.engine.prompting.retry_messages(proposed_relation=proposed_relation,
                                                        existing_relations=similar_relations,
                                                        generation=previous_generation,
                                                        error=error,
                                                        allowed_entity_types=self.relations_data.relations_data.allowed_entity_types)
        return self.decision_from_messages(messages, max_new_tokens, **kwargs)

    def build_kg(self,
                 proposed_relations: list[tuple[str, str, str]],
                 passage: str,
                 **kwargs) -> list[tuple[str, str, str]]:
        added_triplets = []
        for rel in proposed_relations:
            similar_relations = self.relations_data.retrieve_similar_triplets(rel)
            decision, generated_text = self.validate_relation(proposed_relation=rel,
                                                              similar_relations=similar_relations,
                                                              **kwargs)
            if decision is None:
                continue
            elif isinstance(decision, str):
                decision, generated_text = self.retry_validation(proposed_relation=rel,
                                                                 similar_relations=similar_relations,
                                                                 error=decision,
                                                                 previous_generation=generated_text)
                if (decision is None) or isinstance(decision, str):
                    continue
            added_triplets.append(decision)
            if len(decision) > 0:
                self.relations_data.add_triplets([decision], passage)
        return added_triplets

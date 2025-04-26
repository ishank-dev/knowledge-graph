import re
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from importlib import resources

from kg_builder import prompts
from kg_builder.utils import get_triplet_string, process_types


class Prompting:
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError


@dataclass
class ExtractorPrompting(Prompting):
    system_prompt_template: str = resources.read_text(prompts, 'extractor_system_prompt.txt')
    user_instruction: str = "Extract relations triplet from the following passage:"
    allowed_types_prompt: str = "Only extract entities of the following types:"
    default_prompt_file: str = "extractor_examples.json"

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_template.format(user_instruction=self.user_instruction,
                                                  allowed_types_prompt=self.allowed_types_prompt)

    def get_messages(self, passage: str, allowed_types: list[str]) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{self.user_instruction} {passage}\n"
                                            f"{self.allowed_types_prompt} {process_types(allowed_types)}\n"}]

    @staticmethod
    def format_triplets(extracted_relations: str, allowed_types: list[str]) -> list[tuple[str, str, str]]:
        relations = re.findall(r'\((.*?)\)', extracted_relations)
        valid = []
        for r in relations:
            split = r.split(',')
            # standardize split
            if len(split) != 3:
                warnings.warn(f"Cannot get a valid triplet from {r}")
                continue

            # standardize entities and relations
            entity1 = split[0].split(':')[0].strip().lower()
            type1 = split[0].split(':')[-1].strip().upper()
            relation = split[1].strip().lower()
            entity2 = split[2].split(':')[0].strip().lower()
            type2 = split[2].split(':')[-1].strip().upper()

            if type1 not in allowed_types:
                warnings.warn(f"Type {type1} is not in allowed types.")
            elif type2 not in allowed_types:
                warnings.warn(f"Type {type2} is not in allowed types.")
            else:
                valid.append((f"{entity1}:{type1}", relation, f"{entity2}:{type2}"))
        return list(set(valid))


@dataclass
class BuilderPrompting(Prompting):
    system_prompt_template: str = resources.read_text(prompts, 'builder_system_prompt.txt')
    add_key: str = "Add triplet:"
    discard_triplet_value: str = "None"
    default_prompt_file: str = "builder_examples.json"
    thought_start: str = '<think>'
    thought_end: str = '</think>'
    allowed_types_prompt: str = "Subject and object must be of one of the following types:"
    already_extracted_prompt: str = "Triplets already in the Knowledge Graph:"
    proposed_prompt: str = "Proposed triplet:"

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_template.format(add_key=self.add_key,
                                                  discard_triplet_value=self.discard_triplet_value,
                                                  thought_start=self.thought_start,
                                                  thought_end=self.thought_end)

    def get_messages(self,
                     proposed_relation: tuple[str, str, str],
                     existing_relations: list[str],
                     allowed_types: list[str]) -> list[dict[str, str]]:
        allowed_types_string = process_types(allowed_types)
        if len(existing_relations) == 0:
            extracted = "Empty graph"
        else:
            extracted = '\n'.join(existing_relations)
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{self.already_extracted_prompt}:\n {extracted}\n"
                                            f"{self.allowed_types_prompt} {allowed_types_string}\n"
                                            f"{self.proposed_prompt}: {get_triplet_string(proposed_relation)}\n"}]

    def extract_new_triplet(self, generation: str, allowed_entity_types: list[str]) -> tuple[
                                                                                           str, str, str] | str | None:
        if self.add_key not in generation:
            return (
                f"The key '{self.add_key}' is not present in the generated text. Please try again adding"
                f" '{self.add_key}' before the proposed triplet.")
        proposed_triplet = generation.split(self.add_key)[1]
        if len(re.findall(self.discard_triplet_value, proposed_triplet)) > 0:
            return None
        elif len(t := re.findall(r'\((.*?)\)', proposed_triplet)) == 1:
            if len(t[0].split(',')) != 3:
                return (f"A valid relation must have exactly 3 elements. "
                        f"Please try again producing a valid (subject:type, relation, object:type) triplet.")
            triplet = t[0].split(',')
            entity1 = triplet[0].split(':')[0].strip().lower()
            type1 = triplet[0].split(':')[-1].strip().upper()
            relation = triplet[1].strip().lower()
            entity2 = triplet[2].split(':')[0].strip().lower()
            type2 = triplet[2].split(':')[-1].strip().upper()
            if type1 not in allowed_entity_types or type2 not in allowed_entity_types:
                return (f"The types of the subject and object of the triplet must be in {allowed_entity_types}."
                        f"You cannot add other types. Please try again returning either a valid triplet or "
                        f"{self.discard_triplet_value}.")
            return (f"{entity1}:{type1}", relation, f"{entity2}:{type2}")
        else:
            return (f"The value of the {self.add_key} key is not valid. "
                    f"The only allowed values are:\n - a single valid triplet (subject:type, relation, object:type) "
                    f"to be added - 'None' if the relation is already present in the graph.")

    def retry_messages(self,
                       proposed_relation: tuple[str, str, str],
                       existing_relations: list[str],
                       generation: str,
                       error: str,
                       allowed_entity_types: list[str]) -> list[dict[str, str]]:
        standard_msgs = self.get_messages(proposed_relation,
                                          existing_relations,
                                          allowed_types=allowed_entity_types)
        retry_message = {"role": "user",
                         "content": f"Your last generation:\n {generation}\n generated the following error message:{error}"}
        return standard_msgs + [retry_message]

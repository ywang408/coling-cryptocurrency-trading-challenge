from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, constr

from ...portfolio import TradeAction


def delete_placeholder_info(validated_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove placeholder information from the validated output.

    Args:
        validated_output (Dict[str, Any]): The validated output dictionary.

    Returns:
        Dict[str, Any]: The output dictionary with placeholder info removed.
    """
    if "reflection_memory_ids" in validated_output and (
        (validated_output["reflection_memory_ids"])
        and (validated_output["reflection_memory_ids"][0]["memory_index"] == -1)
    ):
        del validated_output["reflection_memory_ids"]
    if "long_memory_ids" in validated_output and (
        (validated_output["long_memory_ids"])
        and (validated_output["long_memory_ids"][0]["memory_index"] == -1)
    ):
        del validated_output["long_memory_ids"]
    if "mid_memory_ids" in validated_output and (
        (validated_output["mid_memory_ids"])
        and (validated_output["mid_memory_ids"][0]["memory_index"] == -1)
    ):
        del validated_output["middle_memory_ids"]
    if "short_memory_ids" in validated_output and (
        (validated_output["short_memory_ids"])
        and (validated_output["short_memory_ids"][0]["memory_index"] == -1)
    ):
        del validated_output["short_memory_ids"]

    return validated_output


class StructureGenerationFailure(BaseModel):
    investment_decision: Literal[TradeAction.HOLD] = TradeAction.HOLD


class StructureOutputResponse(BaseModel):
    investment_decision: Union[TradeAction, None] = None
    summary_reason: constr(min_length=1)  # type: ignore
    short_memory_ids: Union[None, List[int]] = None
    mid_memory_ids: Union[None, List[int]] = None
    long_memory_ids: Union[None, List[int]] = None
    reflection_memory_ids: Union[None, List[int]] = None


class StructuredGenerationChatEndPoint(ABC):
    @abstractmethod
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(
        self, prompt: Union[str, Tuple[str, str]], schema
    ) -> Union[StructureGenerationFailure, StructureOutputResponse]:
        pass

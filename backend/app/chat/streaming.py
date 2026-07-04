import json
from typing import Any

def format_start_part(message_id: str) -> str:
    payload = {
        "type": "text-start",
        "id": message_id
    }
    return f"event: data\ndata: {json.dumps(payload)}\n\n"

def format_text_part(text: str, message_id: str) -> str:
    payload = {
        "type": "text-delta",
        "id": message_id,
        "delta": text
    }
    return f"event: data\ndata: {json.dumps(payload)}\n\n"

def format_end_part(message_id: str) -> str:
    payload = {
        "type": "text-end",
        "id": message_id
    }
    return f"event: data\ndata: {json.dumps(payload)}\n\n"

def format_finish_part() -> str:
    payload = {
        "type": "finish",
        "finishReason": "stop"
    }
    return f"event: data\ndata: {json.dumps(payload)}\n\n"

def format_data_part(data: Any, type_name: str, message_id: str = None) -> str:
    payload = {
        "type": f"data-{type_name}",
        "data": data
    }
    if message_id:
        payload["id"] = message_id
    return f"event: data\ndata: {json.dumps(payload)}\n\n"

def format_error_part(error_message: str) -> str:
    payload = {
        "type": "error",
        "errorText": error_message
    }
    return f"event: data\ndata: {json.dumps(payload)}\n\n"



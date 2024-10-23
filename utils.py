import re
import json

def parse_json_text_with_remaining(
    raw_response: str
):
    pattern = r'```json(.*?)```'
    try:
        matches = re.findall(pattern, raw_response, re.DOTALL)
        if len(matches) == 1:
            match_text = matches[0].strip()
            match_text = match_text.replace("\n", "\\n")
            print(match_text)
            formatted_response = json.loads(match_text)
        else:
            formatted_response = [json.loads(match_text.strip()) for match_text in matches]
        
        if type(formatted_response) is dict:
            for k, v in formatted_response.items():
                if not v or type(v) is not str:
                    continue
                formatted_response[k] = v.replace("\\n", "\n")
        remaining_text = re.sub(pattern, '', raw_response, flags=re.DOTALL).strip()
        return formatted_response, remaining_text
    except Exception as e:  # noqa: F841
        # raise e
        print(f"Fail to parse one output: {raw_response}")
        return None, None


def read_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


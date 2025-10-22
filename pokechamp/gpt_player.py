from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os


class GPTPlayer():
    def __init__(self, api_key="", service_tier: str='priority'):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.service_tier = service_tier
        # Simple compatibility banner toggle
        self._compat_logged = False
        # OpenAI client (shared)
        self.client = OpenAI(api_key=self.api_key)
        # Simple telemetry counters
        self.telemetry = {
            'schema_ok': 0,
            'schema_violation': 0,
        }

    # Capability map per model family
    MODEL_CAPS = {
        "gpt-5":  {"api": "responses", "token_param": "max_completion_tokens", "allow_stop": False, "allow_temperature": False},
        "o3":     {"api": "responses", "token_param": "max_output_tokens",      "allow_stop": False, "allow_temperature": False},
        "o4":     {"api": "responses", "token_param": "max_output_tokens",      "allow_stop": False, "allow_temperature": False},
        "gpt-4o": {"api": "chat",      "token_param": "max_tokens",             "allow_stop": True,  "allow_temperature": True},
    }

    ACTION_SCHEMA = {
        "name": "battle_action",
        "schema": {
            "type": "object",
            "properties": {
                "move":          {"type": "string", "minLength": 1},
                "switch":        {"type": "string", "minLength": 1},
                "dynamax":       {"type": "string", "minLength": 1},
                "terastallize":  {"type": "string", "minLength": 1},
                "why":           {"type": "string", "maxLength": 120}
            },
            "additionalProperties": False,
            "anyOf": [
                {"required": ["move"]},
                {"required": ["switch"]},
                {"required": ["dynamax"]},
                {"required": ["terastallize"]}
            ]
        },
        "strict": True
    }

    def _caps_for_model(self, model: str):
        for prefix, caps in self.MODEL_CAPS.items():
            if model.startswith(prefix):
                return caps
        return {"api": "chat", "token_param": "max_tokens", "allow_stop": True, "allow_temperature": True}

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o', temperature=0.3, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, reasoning_effort="low") -> str:
        client = self.client

        caps = self._caps_for_model(model)
        supports_reasoning = model.startswith(('o3', 'o4', 'gpt-5'))
        uses_max_completion_tokens = (caps["token_param"] == "max_completion_tokens")
        uses_max_output_tokens = (caps["token_param"] == "max_output_tokens")
        supports_temperature = caps["allow_temperature"]
        supports_stop = caps["allow_stop"]

        # Clamp completion budgets based on call type (compact JSON vs reasoning)
        def _clamp_tokens(model_name: str, is_json: bool, requested: int) -> int:
            if model_name.startswith(('gpt-5', 'o3', 'o4')):
                # IO / tool-chooser JSON should be small; leaf eval can be a bit larger
                return min(requested, 120 if is_json else 180)
            return requested

        try:
            clamped_tokens = _clamp_tokens(model, json_format, max_tokens)
            # Route per capability
            if caps["api"] == "responses":
                # Build Responses request
                input_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
                request_params = {
                    "model": model,
                    "input": input_msgs,
                }
                # Responses API expects max_output_tokens for all reasoning families
                request_params["max_output_tokens"] = clamped_tokens
                if supports_temperature and temperature is not None:
                    request_params["temperature"] = float(temperature)
                # Always use priority tier if available
                if self.service_tier:
                    request_params["service_tier"] = self.service_tier
                if json_format:
                    # Try strict schema; if unsupported by installed SDK, we will fallback below
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.ACTION_SCHEMA["name"],
                            "schema": self.ACTION_SCHEMA["schema"],
                        },
                        "strict": True
                    }
                if supports_reasoning and reasoning_effort is not None:
                    request_params["reasoning"] = {"effort": reasoning_effort}

                try:
                    r = client.responses.create(**request_params)
                except TypeError as te:
                    # Installed SDK may not support response_format yet; retry without it
                    if 'response_format' in str(te):
                        rf = request_params.pop('response_format', None)
                        # Strengthen instruction to ensure JSON on fallback
                        input_msgs[-1]["content"] = (
                            (user_prompt or "")
                            + "\n\nReturn ONLY a single JSON object with keys among: move, switch, dynamax, terastallize, why."
                        )
                        r = client.responses.create(**request_params)
                        # Mark as schema_violation telemetry (schema unsupported)
                        self.telemetry['schema_violation'] += 1
                    else:
                        raise
                # Extract text
                outputs = getattr(r, 'output_text', None)
                if not outputs:
                    try:
                        outputs = r.output[0].content[0].text
                    except Exception:
                        outputs = ""
                # Telemetry (best effort)
                self.telemetry['schema_ok'] += 1 if json_format else 0
                response = None  # unify return below
            else:
                # Chat completions path
                request_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    "stream": False,
                }
                if supports_temperature:
                    request_params["temperature"] = temperature
                if supports_stop and stop:
                    request_params["stop"] = stop
                if uses_max_completion_tokens:
                    request_params["max_completion_tokens"] = clamped_tokens
                elif uses_max_output_tokens:
                    request_params["max_output_tokens"] = clamped_tokens
                else:
                    request_params["max_tokens"] = clamped_tokens
                if json_format:
                    request_params["response_format"] = {"type": "json_object"}
                if self.service_tier:
                    request_params["service_tier"] = self.service_tier
                # Compatibility logging removed (too noisy)
                response = client.chat.completions.create(**request_params)
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, reasoning_effort)
        except Exception as e:
            # Schema or API error: fallback to chat JSON mode for resilience
            try:
                request_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    "stream": False,
                    "response_format": {"type": "json_object"}
                }
                # Use safest token param
                request_params["max_tokens"] = min(int(max_tokens), 120)
                # Always use priority tier if available
                if self.service_tier:
                    request_params["service_tier"] = self.service_tier
                response = client.chat.completions.create(**request_params)
                self.telemetry['schema_violation'] += 1
            except Exception:
                raise e
        if response is not None:
            outputs = response.choices[0].message.content
        # log completion tokens
        try:
            usage = response.usage if response is not None else None
            if usage:
                self.completion_tokens += usage.completion_tokens
                self.prompt_tokens += usage.prompt_tokens
        except Exception:
            pass
        if json_format:
            return outputs, True
        return outputs, False

    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.3, model='gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = self.client

        caps = self._caps_for_model(model)
        uses_max_completion_tokens = (caps["token_param"] == "max_completion_tokens")
        uses_max_output_tokens = (caps["token_param"] == "max_output_tokens")
        supports_temperature = caps["allow_temperature"]
        supports_stop = caps["allow_stop"]

        def _clamp_tokens(model_name: str, is_json: bool, requested: int) -> int:
            if model_name.startswith(('gpt-5', 'o3', 'o4')):
                return min(requested, 120 if is_json else 180)
            return requested

        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'

            clamped_tokens = _clamp_tokens(model, json_format, max_tokens)
            if caps["api"] == "responses":
                input_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt+output_padding}
                ]
                params = {"model": model, "input": input_msgs}
                params["max_output_tokens"] = clamped_tokens
                if self.service_tier:
                    params["service_tier"] = self.service_tier
                if json_format:
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.ACTION_SCHEMA["name"],
                            "schema": self.ACTION_SCHEMA["schema"],
                        },
                        "strict": True
                    }
                # Compatibility logging removed (too noisy)
                try:
                    r = client.responses.create(**params)
                except TypeError as te:
                    if 'response_format' in str(te):
                        params.pop('response_format', None)
                        input_msgs[-1]["content"] = (
                            (user_prompt or "") + "\n\nReturn ONLY a single JSON object with keys among: move, switch, dynamax, terastallize, why."
                        )
                        r = client.responses.create(**params)
                        self.telemetry['schema_violation'] += 1
                    else:
                        raise
                message = getattr(r, 'output_text', '') or ''
                if not message:
                    try:
                        message = r.output[0].content[0].text
                    except Exception:
                        message = ''
            else:
                params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt+output_padding}
                    ],
                    "stream": False,
                }
                if supports_temperature:
                    params["temperature"] = temperature
                if supports_stop and stop:
                    params["stop"] = stop
                if uses_max_completion_tokens:
                    params["max_completion_tokens"] = clamped_tokens
                elif uses_max_output_tokens:
                    params["max_output_tokens"] = clamped_tokens
                else:
                    params["max_tokens"] = clamped_tokens
                if json_format:
                    params["response_format"] = {"type": "json_object"}
                if self.service_tier:
                    params["service_tier"] = self.service_tier
                # Compatibility logging removed (too noisy)
                response = client.chat.completions.create(**params)
                message = response.choices[0].message.content

        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)

        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True

        return message, False

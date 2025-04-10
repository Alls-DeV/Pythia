import os
from time import sleep, time

from openai import OpenAI, RateLimitError

from common import PNUMBER1


class GPTPlayer:
    def __init__(self, api_key=""):
        from config import DEEPSEEK_API_KEY, OPENAI_API_KEY

        self.openai_api_key = OPENAI_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY

        self.time_to_respond = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def get_LLM_action(
        self,
        system_prompt,
        user_prompt,
        model,
        temperature=0.3,
        stop=[],
        max_tokens=200,
        log_metadata=None,
    ) -> str:
        if "deepseek" in model:
            client = OpenAI(
                api_key=self.deepseek_api_key, base_url="https://api.deepseek.com"
            )
        else:
            client = OpenAI(api_key=self.openai_api_key)
        start_time = time()
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                stream=False,
                temperature=temperature,
                stop=stop,
                max_tokens=max_tokens,
            )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)
            print("rate limit error")
            return self.get_LLM_action(
                system_prompt,
                user_prompt,
                model,
                temperature,
                stop,
                max_tokens,
                log_metadata,
            )
        end_time = time()
        self.time_to_respond += end_time - start_time
        outputs = response.choices[0].message.content
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        # Logging
        with open(f"./battle_prompts/{PNUMBER1}/log_{model}", "a") as f:
            f.write("Prompt Tokens: " + str(response.usage.prompt_tokens) + "\n")
            f.write(
                "Completion Tokens: " + str(response.usage.completion_tokens) + "\n"
            )
            f.write("Time to respond: " + str(end_time - start_time) + "\n")
            f.write("--" * 50 + "\n")

        if log_metadata is not None:
            with open(f"./battle_prompts/{PNUMBER1}/log_with_io_{model}", "a") as f:
                f.write(log_metadata + "\n")
                f.write("System Prompt:\n" + system_prompt + "\n\n")
                f.write("User Prompt:\n" + user_prompt + "\n\n")
                f.write("Output:\n" + outputs + "\n\n")
                f.write("Prompt Tokens: " + str(response.usage.prompt_tokens) + "\n")
                f.write(
                    "Completion Tokens: " + str(response.usage.completion_tokens) + "\n"
                )
                f.write("Time to respond: " + str(end_time - start_time) + "\n")
                f.write("--" * 50 + "\n")
        return outputs

    def get_LLM_query(
        self,
        system_prompt,
        user_prompt,
        temperature=0.7,
        model="gpt-4o",
        json_format=False,
        seed=None,
        stop=[],
        max_tokens=200,
    ):
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            output_padding = ""
            if json_format:
                output_padding = '\n{"'

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + output_padding},
                ],
                temperature=temperature,
                stream=False,
                stop=stop,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)
            print("rate limit error1")
            return self.get_LLM_query(
                system_prompt,
                user_prompt,
                temperature,
                model,
                json_format,
                seed,
                stop,
                max_tokens,
            )

        if json_format:
            json_start = 0
            json_end = message.find("}") + 1  # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False

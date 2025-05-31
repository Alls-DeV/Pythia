import csv
import os
from time import sleep, time

from openai import AzureOpenAI, OpenAI, RateLimitError

from common import PNUMBER1


class GPTPlayer:
    def __init__(self, api_key=""):
        from config import DEEPSEEK_API_KEY, OPENAI_API_KEY

        self.openai_api_key = OPENAI_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY

        # after the move is chosen it is reset to 0
        self.single_move_response_time = 0
        self.single_move_prompt_tokens = 0
        self.single_move_completion_tokens = 0

        # total response time for the entire game
        self.total_response_time = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Ensure log directory exists
        os.makedirs(f"./llm_log", exist_ok=True)

        # Create CSV file with headers if it doesn't exist
        self.log_file = f"./llm_log/log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "battle_id",
                        "player_name",
                        "turn",
                        "model",
                        "prompt_tokens",
                        "completion_tokens",
                        "response_time",
                        "temperature",
                        "max_tokens",
                        "stop",
                        "node_idx",
                        "parent_idx",
                        "depth",
                        # "system_prompt",
                        # "user_prompt",
                        # "llm_output",
                    ]
                )

    def get_LLM_action(
        self,
        system_prompt,
        user_prompt,
        model,
        temperature=0.3,
        stop=[],
        max_tokens=100,
        log_dict=dict(),
    ) -> str:
        if "deepseek" in model:
            client = OpenAI(
                api_key=self.deepseek_api_key, base_url="https://api.deepseek.com"
            )
        else:
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="https://pokemon-paper.openai.azure.com/",
                api_key=self.openai_api_key,
            )
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
                log_dict,
            )
        end_time = time()
        response_time = end_time - start_time

        self.single_move_response_time += response_time
        self.single_move_prompt_tokens += response.usage.prompt_tokens
        self.single_move_completion_tokens += response.usage.completion_tokens

        self.total_response_time += response_time
        self.total_completion_tokens += response.usage.completion_tokens
        self.total_prompt_tokens += response.usage.prompt_tokens

        llm_output = response.choices[0].message.content

        with open(f"./llm_log/{PNUMBER1}/log_prompts", "a") as f:
            f.write(f"system_prompt: {system_prompt}\n\n")
            f.write(f"user_prompt: {user_prompt}\n\n")
            f.write(f"llm_output: {llm_output}\n\n")
            f.write(f"-" * 20 + "\n\n")

        # Log to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)

            player_name = log_dict.get("player_name", "")
            turn = log_dict.get("turn", -2)
            node_idx = log_dict.get("node_idx", -2)
            parent_idx = log_dict.get("parent_idx", -2)
            depth = log_dict.get("depth", -2)

            writer.writerow(
                [
                    PNUMBER1,  # TODO: It works for 'local_1v1.py' but idk for other files
                    player_name,
                    turn,
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response_time,
                    temperature,
                    max_tokens,
                    stop,
                    node_idx,
                    parent_idx,
                    depth,
                    # system_prompt,
                    # user_prompt,
                    # llm_output,
                ]
            )

        return llm_output

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

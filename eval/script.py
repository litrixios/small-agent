import os
import re
import json
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

import tenacity 
from tqdm.asyncio import tqdm_asyncio as asyctqdm
from codelinker import CodeLinker,CodeLinkerConfig

from eval.quick_rl import PolicyBandit

cfg = CodeLinkerConfig.from_toml("private.toml")
cfg.request.use_cache = False
cl = CodeLinker(config=cfg)
sem = asyncio.Semaphore(16)

SYSTEM = """<Role> You are a helpful assistant that provides proactive suggestions to the user. </Role> 

<Task> Understand what the user is doing and anticipate their needs based on events. Only propose assistance when you fully understand the user's actions. Use available operations to ensure the task is feasible. Execute the task if the user accepts your proposal. </Task> 

<Format> Respond in the following JSON format: 
{
    "Purpose": "The purpose of the user's last action.", 
    "Thoughts": "Your thoughts on the user's actions.", 
    "Proactive Task": "Describe your proposed task, or set to `null` if no assistance is needed.", 
    "Response": "Inform the user about your assistance if proposing a task." 
}
</Format>

<Rules>
- Ensure the proposed task is relevant to the events. - Focus on the user's current needs and predict helpful tasks.
- Consider the timing of events.
- Only offer proactive assistance when necessary.
- Deduce the user's purpose and whether they need help based on event history.
- Set `Proactive Task` to `null` if the user doesn't need help. 
</Rules>"""

STEP = json.dumps({
    "Instructions": "Now analyze the history events and provide a task if you think the user needs your help.",
    "Observation": "[placeholder]"
    })


DIR = "./dataset/test_data"

data_files = os.listdir(DIR)

def extrat_pred(s):
    if '```json' in s:
        s = re.findall(r'```json\n(.*?)\n```', s, re.DOTALL)[0]
    
    ret = json.loads(s)
    if ret.get("Proactive Task",None) is None:
        ret["Proactive Task"] = None
    elif ret.get("Proactive Task") == "null":
        ret["Proactive Task"] = None
            
    if ret.get("Response",None) is not None and ret.get("Response") == "null":
        ret["Response"] = None
    return ret

async def get_response(messages:List[Dict[str,str]], model_name:str) -> Dict[str,str]:
    async for attemp in tenacity.AsyncRetrying(stop=tenacity.stop_after_attempt(5),reraise=True):
        with attemp:
            async with sem:
                res = await cl.exec(
                    model=model_name,
                    messages=messages,
                    completions_kwargs={"temperature":0.0 if attemp.retry_state.attempt_number< 1 else 0.5}
                )    
            result = extrat_pred(res)
            return result

async def get_trace(file_name:str, model_name:str, gate_model: Optional[PolicyBandit] = None, gate_threshold: float = 0.5) -> Dict[str,str]:
    """
    Get the agent response based on the history messages.

    Args:
        message (List[Dict[str,str]]): the history information with turns as {system, user, assistant, ..., user(new event)}
        model_name (str): 

    Returns:
        Dict[str,str], the agent response.
    """
    
    file_path = os.path.join(DIR, file_name)
    with open(file_path, 'r') as f:
        event_trace = json.load(f)
        
    messages = [
        {"role": "system", "content": SYSTEM},
    ]
    
    obs_history = []

    for idx,event in enumerate(event_trace):
        
        print(idx, file_name)
        
        raw_event = event["observation"]
        event_filtered = {"Time": raw_event["time"], "Event": raw_event["event"]}
        
        obs_history.append(raw_event)

        query_prompt = STEP.replace("[placeholder]", json.dumps(event_filtered))
        messages.append({"role": "user", "content": query_prompt})

        if gate_model is not None:
            gate_text = " ".join(item.get("event", "") for item in obs_history)
            prob_help = gate_model.prob_from_text(gate_text)
            if prob_help < gate_threshold:
                result = {
                    "Purpose": "RL gate predicts low need for proactive help.",
                    "Thoughts": f"p_help={prob_help:.4f} < threshold={gate_threshold:.4f}",
                    "Proactive Task": None,
                    "Response": None,
                }
            else:
                try:
                    result = await get_response(messages, model_name)
                except Exception as e:
                    print(e)
                    continue
        else:
            try:
                result = await get_response(messages, model_name)
            except Exception as e:
                print(e)
                continue
            
        
        event_trace[idx]["agent_response"] = [] if result["Proactive Task"] == None else [result["Proactive Task"]]
        event_trace[idx]["other_infomation"] = {k:v for k,v in result.items() if k != "Proactive Task"}
        
        messages.append({"role": "assistant", "content": json.dumps(result)})
        messages[-2]["content"] = json.dumps(event_filtered)

    return event_trace, file_name


async def main(model_name:str, gate_model_path: Optional[str] = None, gate_threshold: Optional[float] = None):
    files = [file for file in data_files if not (file.startswith('turns') or file.startswith('splits.json'))]

    gate_model = None
    gate_t = 0.5
    if gate_model_path is not None:
        gate_model, default_t = PolicyBandit.load(gate_model_path)
        gate_t = default_t if gate_threshold is None else gate_threshold
        print(f"Use RL gate model: {gate_model_path}, threshold={gate_t:.4f}")

    results = await asyctqdm.gather(*[get_trace(file, model_name, gate_model=gate_model, gate_threshold=gate_t) for file in files])
    
    if not os.path.exists(f'./eval/traces_new/{model_name}'):
        os.makedirs(f'./eval/traces_new/{model_name}')
    
    for trace, file in results:
        if trace is not None:
            with open(f'./eval/traces_new/{model_name}/{file}','w',encoding='utf-8') as f:
                json.dump(trace, f, ensure_ascii = False, indent=4)
            

def run(
    model_name: str = "qwen2-7b-instruct",
    gate_model_path: Optional[str] = None,
    gate_threshold: Optional[float] = None,
):
    print(model_name)
    asyncio.run(main(model_name, gate_model_path=gate_model_path, gate_threshold=gate_threshold))


if __name__ == "__main__":
    import fire
    fire.Fire(run)


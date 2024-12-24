import autogen
import dotenv

dotenv.load_dotenv()

config_list = autogen.config_list_from_dotenv(
    ".env",
    {"gpt-4o-mini": "OPENAI_API_KEY"}
)

llm_config = {
    "cache_seed": 43,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,  # in seconds
}
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved "
                   "by this admin.",
    code_execution_config={
        "work_dir": "02-group-chat/code",
        "use_docker": False
    },
    human_input_mode="TERMINATE",
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line.
    Coder. Your job is to write complete code. Make sure to save the code to disk in python script.
    """,
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=llm_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their 
    abstracts printed. You don't write code.""",
)
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the "
                   "plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
)
group_chat = autogen.GroupChat(
    agents=[user_proxy, coder, scientist, planner, critic], messages=[], max_round=12
)
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""
Find papers on LLM applications from arxiv in the 3 days, create a markdown table of different domains. and write into file name 02-group-chat/report.md
""",
)
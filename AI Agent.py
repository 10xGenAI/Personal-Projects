import os
os.environ["OPENAI_API_KEY"] = "{personal api key}"
from collections import deque
from typing import Dict, List, Optional, Any


# from langchain import LLMChain, OpenAI, PromptTemplate
# from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 


# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Define embedding model
# Vector embedding is computational technique of saving data in a format that is easy for computer to understand (cf text, images)
embeddings_model = OpenAIEmbeddings()
# Initialise the vectorsore as empty
# Faiss is a Vectorstore. Vectorstore is used in custom LLM to efficiently save data
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
# dummy_texts = ["Initial dummy text for FAISS index."]
# vectorstore = FAISS.from_texts(texts=[], embedding=embeddings_model, docstore=InMemoryDocstore({}))

########################## The chains to be created ##########################
## 1. Chain that generates the tasks and add the tasks to a list
## 2. Chain that processes the tasks based on priority
## 3. Chain that executes the individual chain
##############################################################################

class TaskCreationChain(LLMChain):
    # Chain to generate tasks
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        #Get the response parser
        task_creation_template = (
            "You are a task creation AU that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has teh result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array"
        )
        prompt = PromptTemplate(
            template = task_creation_template,
            input_variable = [
                "result",
                "task_description",
                "incomplete_tasks",
                "objectives"
            ],
        )
        return cls(prompt = prompt, llm = llm, verbose = verbose)
    
class TaskPrioritizationChain(LLMChain):
    #Chain to prioritise tasks
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        #Get the response parser
        task_prioritisation_template = (
            "You are a task prioritisatino AI tasked with cleaning the formatting of"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}"
        )
        prompt = PromptTemplate (
            template = task_prioritisation_template,
            input_variables = ["task_names", "next_task_id", "objective"],
        )
        return cls(prompt = prompt, llm = llm, verbose = verbose)
    
class ExecutionChain(LLMChain):
    #Chain to execute tasks
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        #Get the reponse parser
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}"
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template = execution_template, 
            input_variables = ["objective", "context", "task"],
        )
        return cls(prompt = prompt, llm = llm, verbose = verbose)
    
########################## Get Next Task ##########################
## 1. Once the task has been listed up, execute the first task and bring the next task (iterate this process)
###################################################################

def get_next_task (
    task_creation_chain: LLMChain, 
    result: Dict, 
    task_descriptionn: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    # Get the next task
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result = result,
        task_descriptionn = task_descriptionn, 
        incomplete_tasks = incomplete_tasks, 
        objective = objective,
    )
    new_tasks = response.split("\\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

########################## Prioritise Task ########################
## 1. Triggers task_prioritisation_chain
###################################################################
def prioritise_tasks (
    task_prioritisation_chain: LLMChain,
    this_task_id: int, 
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    #Prioritise Task
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1 
    response = task_prioritisation_chain.run(
        task_names = task_names, next_task_id = next_task_id, objective = objective
    )
    new_tasks = response.split("\\n")
    prioritised_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritised_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritised_task_list

########################## Execute Task ########################
## 1. Exectue Task by sorting out the tasks first
################################################################
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    #Get the top k tasks based on the query
    results = vectorstore.similarity_search_with_score(query, k = k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key = lambda x: x[1], reverse = True))
    return [str(item.metadata["task"]) for item in sorted_results]

def execute_task (
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    #Execute task
    context = _get_top_tasks(vectorstore, query = objective, k = k)
    return execution_chain.run(objective = objective, context = context, task = task)

########################## Baby AGI ########################
## 1. Synthesis all the chains above
############################################################
class BabyAGI(Chain, BaseModel):
    #Controller model for the BabyAGI agent
    
    task_list: deque = Field(default_factory = deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritisation_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init = False)
    max_iterations: Optional[int] = None
    
    class Config:
        #Configuration forthis pydantic object
        
        arbitary_types_allowed = True
        
    def add_task(self, task: Dict):
        self.task_list.append(task)
        
    def print_task_list(self):
        print("\033[95m\033[1m" + "\n********Task List********\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])
    
    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n********Next Task********\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])
        
    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n********Task Result********\n" + "\033[0m\033[0m")
        print(result)
        
    @property
    def input_keys(self) -> List[str]:
        return[]
    
    @property
    def output_keys(self) -> List[str]:
        return[]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        #Run the agent; objectve is defined by user input (AI Agent triggered by user input)
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0 
        while True:
            if self.task_list:
                self.print_task_list()
                
                #Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)
                
                #Step 2: Execute the task
                result = execute_task (
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)
                
                #Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}_{num_iters}"
                self.vectorstore.add_texts(
                    texts = [result], 
                    metadatas = [{"task": task["task_name"]}],
                    ids = [result_id],
                )
                
                #Step 4: Create new tasks and reprioritise task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.taask_id_counter += 1 
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritise_tasks(
                        self.task_prioritisation_chain,
                        this_task_id,
                        list(self.task_list),
                        objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n********Task Ending********\n" + "\033[0m\033[0m"
                )
                break
        return{}
    
    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        #Initialise the BabbyAGI Controller
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose = verbose)
        task_prioritisation_chain = TaskPrioritizationChain.from_llm(
            llm, verbose = verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, verbose = verbose)
        return cls (
            task_creation_chain = task_creation_chain, 
            task_prioritisation_chain = task_prioritisation_chain,
            execution_chain = execution_chain,
            vectorstore = vectorstore,
            **kwargs
        )
    
OBJECTIVE = "Wrtie a weather report for Sydney today"

llm = OpenAI(temperature = 0)
# Logging of LLMChains
verbose = False 
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm = llm, vectorstore = vectorstore, verbose = verbose, max_iterations = max_iterations
)
baby_agi.invoke({"objective": OBJECTIVE})
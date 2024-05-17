import gradio as gr
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.agent import AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import CSVLoader

import os
import openai
from openai import OpenAI
import cv2 
import json
import base64
from enum import Enum
from io import BytesIO
import numpy as np,requests
import gradio as gr
from gradio_modal import Modal
import PIL
from PIL import Image
import regex as re
import mermaid as md
from mermaid.graph import Graph
from langchain.agents import AgentExecutor
from PIL.JpegImagePlugin import JpegImageFile
from langchain_core.pydantic_v1 import BaseModel,Field


#################### Model Init ####################################
llm_model = "gpt-4-turbo"
os.environ['OPENAI_API_KEY']=""
openai.api_key = os.environ['OPENAI_API_KEY']


embedding = OpenAIEmbeddings()
file = 'incident_event_log.csv'
loader = CSVLoader(file_path=file)
inc_data = loader.load()
inc_data=inc_data[:100]
db = Chroma.from_documents(inc_data, embedding)
retriever = db.as_retriever()

#################### User defined Tools ##############################

# Define the input schema

from typing import Optional

class get_ec2_instance_type(BaseModel):
    """ extract instance type details """
    instance_type : Optional[str] = Field(description="extract the ec2 instance type to create instance (eg: t2.micro)")
    intance_n     : Optional[int] = Field(description="extract the number of ec2 instances to launch (eg:2)")    

@tool(args_schema=get_ec2_instance_type)
def launch_ec2_instances(instance_type='t2.micro',instance_n=1) -> [str]:
    """ Launch ec2 instance using boto3"""
    try:
        import boto3
        ec2 = boto3.resource('ec2',region_name='us-east-1')
        instance = ec2.create_instances(
            ImageId='ami-07caf09b362be10b8',
            MinCount=1,
            MaxCount=instance_n,
            InstanceType=instance_type,
            TagSpecifications=[{'ResourceType': 'instance','Tags':[{'Key': 'Name','Value': 'Vigil_agent'},]},])
        return f"Launched Sucessfully check with the following instance ids in Console {[i.id for i in instance]}"
        
    except Exception as e:
        return f"Failed to create instance , Reason : {e}"
        
class get_ec2_inst(BaseModel):
    """Listing instance state information"""
    instance_state : Optional[str] = Field(description="List the ec2 instance which are running, sample output : `running`")

@tool(args_schema=get_ec2_inst)
def list_ec2_instances(instance_state='running') -> [str]:
    """List the ec2 instances which are running"""
    try:
        import pandas as pd
        import boto3
        ec2 = boto3.resource('ec2',region_name='us-east-1')
        instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': [f'{instance_state}']}])
        out={'Instance_id':[],'Instance Type':[],'Instance_name':[]}
        for instance in instances:
            out['Instance_id'].append(instance.id)
            out['Instance Type'].append(instance.instance_type)
            out['Instance_name'].append(instance.tags[0]['Value'] if instance.tags else None)            
        return pd.DataFrame(out).to_markdown()    
    except Exception as e:
        return f"Unable to list the instances at the Moment, Reason {e}"

# Define the input schema
class MermaidCodeInp(BaseModel):
    mermaid_code: str = Field(description="extract Mermaid code to render in python code")

@tool(args_schema=MermaidCodeInp)
def generate_diagram(mermaid_code: str) -> str:
    """ convert Mermaid Code to diagram by renderdering the inp in python"""

    reg_op=re.findall("```mermaid\n(.*?)\n```",mermaid_code,re.DOTALL) if 'mermaid' in mermaid_code else [mermaid_code]
    print(f"mermaid regex generated : ",reg_op,mermaid_code) 
    
    if len(reg_op):
         mermaid_op=reg_op[0]
    else:
        mermaid_op="\ngraph LR\n    A[Error] --> B((Error))\n"
        return
    print(f"mermaid op generated : ",mermaid_op)    
    graph: Graph = Graph('example-flowchart',f"""{mermaid_op}""")
    graphe: md.Mermaid = md.Mermaid(graph)
    graphe.to_png('test.png')
    path='test.png'
    # pil_image=Image.open('test.png')
    return path    

class incident_info(BaseModel):
    """ get info about incident ticket"""
    priority: Optional[str] = Field(description="extract priority status eg: `High`,`Medium`,`Low`")
    Active_status : Optional[str] = Field(description="extract Active status eg: `open`,`closed`,`Resolved`")
    
@tool(args_schema=incident_info)
def fetch_from_vectordb(priority='None',Active_status ='open') -> str:
    """ filter the tickets based on incident info"""
    
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    output=chain.invoke(f'List the {priority} priority and active status {Active_status} tickets info in tabular format')
    return output  

class summarize_network(BaseModel):
    """ use this tool to explain about network diagram """
    query: str = Field(description="what user asks to summarize")
    
@tool(args_schema=summarize_network)
def summarize_network_diagram(query: str) -> str:
    """ summarize about the network diagram"""

    # Function to encode the image as base64
    def encode_image(image_path: str):
        # check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    try:
        out_path=f"/private/var/folders/xv/0p64ckyx0tlbbm7wvcy3sp940000gn/T/gradio/14cc640781c3fb23d9540ee0798ee3370717c04c/network.jpeg"
        base64_img = encode_image(out_path)
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
                    {
                        "role": "user",
                        "content": query,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                }
                            },
                        ],
                    }
                ],
        )
    except Exception as e:
        output=f"Error in response from gpt-4 : {e}"
        return output

    output=response.choices[0].message.content
    return output  
#################################################################


def init_agent(preamble):
    prompt = ChatPromptTemplate.from_messages([
                        ("system", f"{preamble}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])
    functions = [format_tool_to_openai_function(f) for f in [launch_ec2_instances,list_ec2_instances,generate_diagram,fetch_from_vectordb,summarize_network_diagram]]
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    agent_chain = RunnablePassthrough.assign(
                        agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
                        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    tools=[launch_ec2_instances,list_ec2_instances,generate_diagram,fetch_from_vectordb,summarize_network_diagram]
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
    agent_executor = AgentExecutor(agent=agent_chain, tools=tools,verbose=True, memory=memory)
    return agent_executor    

preamble='You are nice person'
agent_executor=init_agent(preamble)

############# gradio functions ########################

def run_agent(hid_txt,text,chatbot):   
    
    global preamble
    global agent_executor
    # print(f'from interface : {text},{chatbot}')
    if data[0][hid_txt]['preamble'] != preamble:
        preamble=data[0][hid_txt]['preamble'] 
        agent_executor=init_agent(preamble)
        print(f'preamble set succesfully !!')

    result=agent_executor.invoke({"input": text})     
    # print('result returned : ',result)
    chatbot.append((text, result['output']))
    return "",chatbot

def video_display(video):

    print(video)
    client = OpenAI()
    video = cv2.VideoCapture(video)
    
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Generate a compelling story describing the scene in 25 words",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::100]),
        ],
    },
    ]
    params = {
        "model": "gpt-4-turbo",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }
    
    result = client.chat.completions.create(**params)
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1-1106",
            "input": result.choices[0].message.content,
            "voice": "onyx",
        },
    )
    
    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    return result.choices[0].message.content,audio


def image_save(img):
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    out="/private/var/folders/xv/0p64ckyx0tlbbm7wvcy3sp940000gn/T/gradio/14cc640781c3fb23d9540ee0798ee3370717c04c/network.jpeg"
    image.save(out)
    print('Img uploaded success !')
    return gr.Image()
    
def image_display():
    return 'test.png'

def change_tab(modal1_txt,evt: gr.EventData):
    sel_idx=[idx for idx,img in enumerate(images) if img[1]==modal1_txt][0]
    return gr.Tabs(selected=int(sel_idx)+1)

def set_accodian_values(evt: gr.SelectData):

    prble_txt=data[0][str(evt.index)]['preamble']
    seed_txt=data[0][str(evt.index)]['seed_chat']
    return Modal(visible=True),gr.Image(value=images[evt.index][0]),gr.Textbox(value=images[evt.index][1],label='Agent Name'),gr.Textbox(value=prble_txt,label='Preamble',lines=5),gr.Textbox(value=seed_txt,label='Configure behaviour of Agent',lines=5)


images=[('./agent_images/albert_einstein.jpeg','ITSM Agent'),
               ('./agent_images/bill_gates.jpeg','Cloud Engineer Agent'),
               ('./agent_images/elon.jpeg','Multi-Modal Agent'),
               ('./agent_images/steve_jobs.jpeg','Network Aagent')]

with open(f'companion.json','r') as fp:
    data=json.load(fp)

with gr.Blocks() as demo:

    with gr.Tabs() as tabs:
    
        with gr.TabItem("Agents",id=0): 

            gallery = gr.Gallery(label="Agent images", elem_id="gallery", columns=[4], rows=[1],allow_preview=False, 
                             object_fit="contain", height="500",value=images) 

            with Modal(visible=False) as modal:    
                modal1_img=gr.Image(height=200,width=200)
                modal1_txt=gr.Textbox(label='Agent Name')
                preamble_txt=gr.Textbox(label='Preamble',lines=5)
                seed_chat=gr.Textbox(label='Configure behaviour of Agent',lines=5)
                # sb_button=gr.Button(value="Save to DB")
                agent_btn=gr.Button(value="Talk to Agent")
            
        with gr.TabItem("ITSM Agent",id=1):
            # ag1 = gr.ChatInterface(fn=run_agent, examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}], title="ITSM Agent", multimodal=True)
            chatbot = gr.Chatbot(value=[[None,"Hi Im Einstein, Im your ITSM agent , How can I help You ?"]],height=500,avatar_images=('./agent_images/user.png','./agent_images/albert_einstein.ico')) #just to fit the notebook
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
            hid_txt=gr.Textbox(value='0',visible=False)
            gr.Examples([['display open tickets from ITSM'],['Show High Priority tickets from ITSM']], [msg], [chatbot])
            btn.click(run_agent, inputs=[hid_txt, msg, chatbot], outputs=[msg, chatbot])
            
        with gr.TabItem("Cloud Engineer Agent",id=2):
            # ag2 = gr.ChatInterface(fn=run_agent, examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}], title="Cloud Engineer Agent", multimodal=True)
            chatbot = gr.Chatbot(value=[[None,"Hi Im BillGates, Im your Cloud Engineer Agent , How can I help You ?"]],height=500,avatar_images=('./agent_images/user.png','./agent_images/bill_gates.ico')) #just to fit the notebook
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
            hid_txt=gr.Textbox(value='1',visible=False)
            btn.click(run_agent, inputs=[hid_txt, msg, chatbot], outputs=[msg, chatbot])
            gr.Examples([['List all running ec2 instances'],['create an ec2 instance with instance type :t2.micro'],["""step 1: Display the mermaid code syntax of server provisioning in AWS with all VPC configurations
            step 2 : generate a diagram"""]], [msg], [chatbot])
            with gr.Row():
                gr.Interface(image_display,None,outputs="image",title='Download your diagram')

        with gr.TabItem("Multi-Modal Agent",id=3):
            # ag3 = gr.ChatInterface(fn=run_agent, examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}], title="Analytics Agent", multimodal=True)
            chatbot = gr.Chatbot(value=[[None,"Hi Im Elon Musk, Im your Multi-Modal Agent , How can I help You ?"]],height=500,avatar_images=('./agent_images/user.png','./agent_images/elon.ico')) #just to fit the notebook
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
            hid_txt=gr.Textbox(value='2',visible=False)
            btn.click(run_agent, inputs=[hid_txt, msg, chatbot], outputs=[msg, chatbot])
            with gr.Row():
                gr.Interface(video_display,inputs='video',outputs=[gr.Textbox(label="Story of the scene :"),'audio'],title='upload your video')
            

        with gr.TabItem("Network Agent",id=4):
            # ag4 = gr.ChatInterface(fn=run_agent, examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}], title="Network Agent", multimodal=True)
            chatbot = gr.Chatbot(value=[[None,"Hi Im Steve Jobs, Im your Network Agent , How can I help You ?"]],height=500,avatar_images=('./agent_images/user.png','./agent_images/steve_jobs.ico')) #just to fit the notebook
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
            hid_txt=gr.Textbox(value='3',visible=False)
            btn.click(run_agent, inputs=[hid_txt, msg, chatbot], outputs=[msg, chatbot])
            gr.Examples([['Analyse the network diagram and summarize it']], [msg], [chatbot])
            with gr.Row():
                img=gr.Image(label='upload query image :')
            submit_btn=gr.Button("Submit")
            submit_btn.click(image_save,inputs=[img],outputs=img)    

    gallery.select(set_accodian_values, None, [modal,modal1_img,modal1_txt,preamble_txt,seed_chat])
    agent_btn.click(change_tab,[modal1_txt],tabs)
    
    # sb_button.click(save_to_db,[modal1_txt,preamble_txt,seed_chat],[modal1_txt,preamble_txt,seed_chat])

    # gallery.select(change_tab,None,tabs)
    # show_btn.click(lambda: Modal(visible=True), None, modal)

gr.close_all()
demo.launch(server_port=2101,inline=False)
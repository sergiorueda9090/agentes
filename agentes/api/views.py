import os
from rest_framework.decorators import api_view
from rest_framework.response   import Response
from rest_framework            import status

import langchain
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, 
                              AIMessagePromptTemplate, HumanMessagePromptTemplate, load_prompt)

from langchain.output_parsers import CommaSeparatedListOutputParser

from dotenv import load_dotenv

load_dotenv() # Carga las variables del .env

api_key = os.getenv("open_ai_key")
chat    = ChatOpenAI(openai_api_key=api_key)

@api_view(['POST'])
def api_agent(request):

    if not request.POST:
        return Response({'message': 'No se ha enviado un mensaje'}, status=status.HTTP_400_BAD_REQUEST)
    
    validate_data = ['human_message', 'system_message']
    missing_data  = [key for key in validate_data if key not in request.POST or not request.POST[key].strip()]

    if missing_data:
        return Response({'message': f'Faltan los siguientes datos: {", ".join(missing_data)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    system_template       = "Eres una IA especializada en coches de tipo {tipo_coche} y generar articulos que se leen en {tiempo_lectura}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template       = "Necesito un articulo para vehiculos con motor {peticion_tipo_motor}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt         = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    solicitud_completa  = chat_prompt.format_prompt(peticion_tipo_motor='Hibrido enchufable', tiempo_lectura='10 min', tipo_coche='japoneses').to_messages()

    human_message, system_message = request.POST.get('human_message'), request.POST.get('system_message')
    
    human_message  = HumanMessage(content=human_message)
    system_message = SystemMessage(content=system_message)
    result         = chat.invoke(solicitud_completa)
    print(result)
    return Response({'message': result.content}, status=status.HTTP_200_OK)

@api_view(['POST'])
def api_parsear(request):
    out_put = CommaSeparatedListOutputParser()
    format_instructions = out_put.get_format_instructions()
    
    human_template = "{request}\n{format_instructions}"
    human_prompt   = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt    = ChatPromptTemplate.from_messages([human_prompt])
    chat_prompt    = chat_prompt.format_prompt(request="Dame una lista de colores", format_instructions=format_instructions).to_messages()

    result = chat.invoke(chat_prompt)

    return Response({'message': result.content}, status=status.HTTP_200_OK)

@api_view(['POST'])
def api_save_prompt(request):
    template_save = "Pregunta {pregunta_usuario}\n\nRespuesta: Vamos a verlo paso a paso."
    prompt = PromptTemplate(template=template_save)
    prompt.save("prompt.json")
    return Response({'message': 'Prompt guardado'}, status=status.HTTP_200_OK)

@api_view(['GET'])
def api_get_prompt(request):
    prompt_got = load_prompt('../../prompt.json')
    return  Response({'message': prompt_got}, status=status.HTTP_200_OK)
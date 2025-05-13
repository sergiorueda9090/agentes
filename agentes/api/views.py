import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response   import Response
from rest_framework            import status

import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, 
                              AIMessagePromptTemplate, HumanMessagePromptTemplate, load_prompt)

from langchain.output_parsers   import CommaSeparatedListOutputParser
from langchain.document_loaders import CSVLoader, BSHTMLLoader, PyPDFLoader, WikipediaLoader, TextLoader
from langchain.text_splitter    import CharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


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

@api_view(['GET'])
def api_loader_csv(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'ventas.csv')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo data.csv no existe."}, status=404)

    try:
        loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ','})
        data = loader.load()
        print(data[0].page_content)
        return Response({"message": "Archivo cargado correctamente", "data": data[0].page_content})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def api_loader_html(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'mysql.html')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo mysql.html no existe."}, status=404)

    try:
        loader = BSHTMLLoader(file_path=file_path)
        data = loader.load()
        return Response({"message": "Archivo cargado correctamente", "data": data[0].page_content})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(['GET'])
def api_loader_pdf(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'documentopdf.pdf')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo documentopdf.html no existe."}, status=404)

    try:
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load_and_split()
        human_template = "Necesito que hagas un resumen del siguente texto:\n{contenido}"
        human_prompt   = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt    = ChatPromptTemplate.from_messages([human_prompt])
        documento_completo = ""
        for page in pages:
            documento_completo += page.page_content

        solicitud_completa  = chat_prompt.format_prompt(contenido=documento_completo).to_messages()
        restul = chat.invoke(solicitud_completa)
        return Response({"message": "Archivo cargado correctamente", "data": restul.content})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
def responser_wikipedia(persona, pregunta_arg):
    docs = WikipediaLoader(query=persona, lang="es", load_max_docs=5)
    contexto_extra = docs.load()[0].page_content

    human_message = "Responde a esta pregunta\n{pregunta}, aqui tienes contenido extra:\n{contenido}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_message)

    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    solicitud_completa = chat_prompt.format_prompt(pregunta=pregunta_arg, contenido=contexto_extra).to_messages()
    result = chat.invoke(solicitud_completa)
    return result.content

@api_view(['POST'])
def api_wikipedia(request):
    if not request.POST:
        return Response({'message': 'No se ha enviado un mensaje'}, status=status.HTTP_400_BAD_REQUEST)

    validate_data = ['persona', 'pregunta']
    missing_data  = [key for key in validate_data if key not in request.POST or not request.POST[key].strip()]

    if missing_data:
        return Response({'message': f'Faltan los siguientes datos: {", ".join(missing_data)}'}, status=status.HTTP_400_BAD_REQUEST)

    persona, pregunta_arg = request.POST.get('persona'), request.POST.get('pregunta')
    result = responser_wikipedia(persona, pregunta_arg)
    return Response({'message': result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def api_transformacion_documents(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'historia.txt')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo historia.txt no existe."}, status=404)

    try:
        with open(file_path, encoding='utf-8') as file:
            texto_completo = file.read()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000)

        texts = text_splitter.create_documents([texto_completo])
        return Response({"message": "Archivo cargado correctamente", "data": len(texto_completo)})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def api_embeddings(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'ventas.csv')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo historia.txt no existe."}, status=404)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    loader     = CSVLoader(file_path, csv_args={"delimiter": ","})
    data = loader.load()

    embedded_docs = embeddings.embed_documents([elemento.page_content for elemento in data])
   

    return Response({"message": "Archivo cargado correctamente", "data": len(embedded_docs)})

@api_view(['GET'])
def api_vectores_db(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'historia.txt')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo historia.txt no existe."}, status=404)
    
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter   = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs            = text_splitter.split_documents(documents)

    function_embedding = OpenAIEmbeddings(openai_api_key=api_key)
    
    persist_path = './media/db/ejemplos_embedding_db'

    vector_store = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=function_embedding,
        persist_path=persist_path,
        serializer='parquet'
    )
    vector_store.persist()

    return Response({"message": "Archivo cargado correctamente", "data": ""})



@api_view(['GET'])
def api_consulta_vectores_db(request):
    file_path = os.path.join(settings.BASE_DIR, 'media', 'document', 'historia.txt')

    if not os.path.exists(file_path):
        return Response({"error": "El archivo historia.txt no existe."}, status=404)
    
    function_embedding       = OpenAIEmbeddings(openai_api_key=api_key)
    vectore_store_connection = SKLearnVectorStore(
        embedding=function_embedding,
        persist_path='./media/db/ejemplos_embedding_db',
        serializer='parquet',
    )

    nueva_consulta = "dame informacion de la Época Colonial"
    docs = vectore_store_connection.similarity_search(nueva_consulta, k=1)

    return Response({"message": "Vectores base de datos", "data": docs[0].page_content})
 

@api_view(['GET'])
def api_compresion_optimizacion(request):
    loader = WikipediaLoader(query="Lenguaje python", lang="es", load_max_docs=5)
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs          = text_splitter.split_documents(documents)

    function_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    persist_path = './media/db/ejemplos_wikipedia_db'

    vector_store = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=function_embeddings,
        persist_path=persist_path,
        serializer='parquet'
    )
    vector_store.persist()
    consulta = "Por que el lenguaje de python se llama asi?"
    docs = vector_store.similarity_search(consulta)
    return Response({"message": "Vectores base de datos", "data": docs[0].page_content})


@api_view(['GET'])
def api_compresion_optimizacion_llm(request):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    compressor = LLMChainExtractor.from_llm(llm)

    function_embedding  = OpenAIEmbeddings(openai_api_key=api_key)

    vector_store = SKLearnVectorStore(
        embedding    = function_embedding,
        persist_path = './media/db/ejemplos_wikipedia_db',
        serializer   = 'parquet',
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=vector_store.as_retriever()
    )

    compressed_docs = compression_retriever.invoke("Por que el lenguaje de python se llama asi?")
    return Response({"message": "Vectores base de datos", "data": compressed_docs[0].page_content})

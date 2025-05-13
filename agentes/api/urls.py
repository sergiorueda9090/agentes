from django.urls import path
from .views      import (api_agent, api_parsear, api_save_prompt, api_get_prompt, api_loader_csv, api_loader_html,
                         api_loader_pdf, api_wikipedia, api_transformacion_documents, api_embeddings, api_vectores_db,
                        api_consulta_vectores_db, api_compresion_optimizacion, api_compresion_optimizacion_llm)

urlpatterns = [
    path('api_agent',       api_agent,         name="api_agent"),
    path('api_parsear',     api_parsear,       name="api_parsear"),
    path('api_save_prompt', api_save_prompt,   name="api_save_prompt"),
    path('api_get_prompt',  api_get_prompt,    name="api_get_prompt"),
    path('api_loader_csv',  api_loader_csv,    name="api_loader_csv"),
    path('api_loader_html', api_loader_html,   name="api_loader_html"),
    path('api_loader_pdf',  api_loader_pdf,    name="api_loader_pdf"),
    path('api_wikipedia',   api_wikipedia,     name="api_wikipedia"),
    path('api_transformacion_documents',   api_transformacion_documents,     name="api_transformacion_documents"),
    path('api_embeddings',  api_embeddings,    name="api_embeddings"),
    path('api_vectores_db',  api_vectores_db,    name="api_vectores_db"),
    path('api_consulta_vectores_db',  api_consulta_vectores_db,    name="api_consulta_vectores_db"),
    path('api_compresion_optimizacion',  api_compresion_optimizacion,    name="api_compresion_optimizacion"),
    path('api_compresion_optimizacion_llm',  api_compresion_optimizacion_llm,    name="api_compresion_optimizacion_llm"),
]
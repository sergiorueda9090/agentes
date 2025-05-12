from django.urls import path
from .views      import api_agent, api_parsear, api_save_prompt, api_get_prompt

urlpatterns = [
    path('api_agent',       api_agent,          name="api_agent"),
    path('api_parsear',     api_parsear,        name="api_parsear"),
    path('api_save_prompt', api_save_prompt,    name="api_save_prompt"),
    path('api_get_prompt',  api_get_prompt,     name="api_get_prompt"),
]
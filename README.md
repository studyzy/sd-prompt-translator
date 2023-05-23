# sd-prompt-translator
Stable Diffusion extension for prompt translation
基于facebook/mbart-large-50-many-to-many-mmt的prompt翻译模型，用于Stable Diffusion的prompt翻译任务。
在首次安装使用时会自动下载约2.4G的翻译模型, 请确保网络连接正常。
在启用本扩展后，无需配置，即可在Stable Diffusion中使用中文或者其他母语编写提示词。默认只翻译正向提示词，如果想翻译负向提示词，可以在WebUI中修改。
如需自定义提示词的中文、英文对应关系，可以在extensions/sd-prompt-translator/scripts/translation.csv中，您可以自行添加或修改。
本扩展还支持api调用时使用中文编写提示词。
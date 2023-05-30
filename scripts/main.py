# Based on ParisNeo/prompt_translator, but fix some bugs and add more features

import modules.scripts as scripts
import gradio as gr
from modules.shared import opts
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import re
import os
import string
import csv
import time

# The directory to store the models
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# 中文转英文的翻译模型
class ZhEnTranslator:
    def __init__(self, cache_dir=cache_dir, model_name="Helsinki-NLP/opus-mt-zh-en"):
        self.model_name = model_name

        # 加载模型和tokenizer
        self.model = MarianMTModel.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)

    def translate(self, chinese_str: str, input_language: str, output_language: str) -> str:
        # 对中文句子进行分词
        input_ids = self.tokenizer.encode(chinese_str, return_tensors="pt")

        # 进行翻译
        output_ids = self.model.generate(input_ids)

        # 将翻译结果转换为字符串格式
        english_str = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return english_str

# 支持50种语言的翻译模型
class MBartTranslator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-many-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self, model_name="facebook/mbart-large-50-many-to-one-mmt", src_lang=None, tgt_lang=None):

        self.supported_languages = [
            "af_ZA",
            "ar_AR",
            "az_AZ",
            "bn_IN",
            "cs_CZ",
            "da_DK",
            "de_DE",
            "en_XX",
            "es_XX",
            "et_EE",
            "fa_IR",
            "fi_FI",
            "fr_XX",
            "gl_ES",
            "gu_IN",
            "he_IL",
            "hi_IN",
            "hr_HR",
            "hu_HU",
            "id_ID",
            "it_IT",
            "ja_XX",
            "ka_GE",
            "kk_KZ",
            "km_KH",
            "ko_KR",
            "lt_LT",
            "lv_LV",
            "mk_MK",
            "ml_IN",
            "mn_MN",
            "mr_IN",
            "ne_NP",
            "nl_XX",
            "no_NO",
            "pl_PL",
            "pt_XX",
            "ro_RO",
            "ru_RU",
            "si_LK",
            "sl_SI",
            "sv_SE",
            "sw_KE",
            "ta_IN",
            "te_IN",
            "th_TH",
            "tl_XX",
            "tr_TR",
            "uk_UA",
            "ur_PK",
            "vi_VN",
            "xh_ZA",
            "zh_CN",
        ]
        print("Building translator")
        print("Loading generator (this may take few minutes the first time as I need to download the model)")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        print("Loading tokenizer")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang, cache_dir=cache_dir)
        print("Translator is ready")

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
            input_language (str): The input language code (e.g. "hi_IN" for Hindi).
            output_language (str): The output language code (e.g. "en_US" for English).

        Returns:
            str: The translated text.
        """
        if input_language not in self.supported_languages:
            raise ValueError(f"Input language not supported. Supported languages: {self.supported_languages}")
        # if output_language not in self.supported_languages:
        #     raise ValueError(f"Output language not supported. Supported languages: {self.supported_languages}")

        self.tokenizer.src_lang = input_language
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_input)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated_text[0]

# 语言选择器
class LanguageOption:
    """
    A class representing a language option in a language selector.

    Attributes:
        label (str): The display label for the language option.
        language_code (str): The ISO 639-1 language code for the language option.
    """

    def __init__(self, label, language_code):
        """
        Initializes a new LanguageOption instance.

        Args:
            label (str): The display label for the language option.
            language_code (str): The ISO 639-1 language code for the language option.
        """
        self.label = label
        self.language_code = language_code



# This is a list of LanguageOption objects that represent the various language options available.
# Each LanguageOption object contains a label that represents the display name of the language and 
# a language code that represents the code for the language that will be used by the translation model.
# The language codes follow a format of "xx_YY" where "xx" represents the language code and "YY" represents the 
# country or region code. If the language code is not specific to a country or region, then "XX" is used instead.
# For example, "en_XX" represents English language and "fr_FR" represents French language specific to France.
# These LanguageOption objects will be used to display the language options to the user and to retrieve the 
# corresponding language code when the user selects a language.
language_options = [
    LanguageOption("中文", "zh_CN"),
    LanguageOption("عربية", "ar_AR"),
    LanguageOption("Deutsch", "de_DE"),
    LanguageOption("Español", "es_XX"),
    # LanguageOption("Français", "fr_XX"),
    LanguageOption("हिन्दी", "hi_IN"),
    LanguageOption("Italiano", "it_IT"),
    LanguageOption("日本語", "ja_XX"),
    LanguageOption("한국어", "ko_XX"),
    LanguageOption("Português", "pt_XX"),
    LanguageOption("Русский", "ru_RU"),
    LanguageOption("Afrikaans", "af_ZA"),
    LanguageOption("বাংলা", "bn_BD"),
    LanguageOption("Bosanski", "bs_XX"),
    LanguageOption("Català", "ca_XX"),
    LanguageOption("Čeština", "cs_CZ"),
    LanguageOption("Dansk", "da_XX"),
    LanguageOption("Ελληνικά", "el_GR"),
    LanguageOption("Eesti", "et_EE"),
    LanguageOption("فارسی", "fa_IR"),
    LanguageOption("Suomi", "fi_FI"),
    LanguageOption("ગુજરાતી", "gu_IN"),
    LanguageOption("עברית", "he_IL"),
    LanguageOption("हिन्दी", "hi_XX"),
    LanguageOption("Hrvatski", "hr_HR"),
    LanguageOption("Magyar", "hu_HU"),
    LanguageOption("Bahasa Indonesia", "id_ID"),
    LanguageOption("Íslenska", "is_IS"),
    LanguageOption("日本語", "ja_XX"),
    LanguageOption("Javanese", "jv_XX"),
    LanguageOption("ქართული", "ka_GE"),
    LanguageOption("Қазақ", "kk_XX"),
    LanguageOption("ខ្មែរ", "km_KH"),
    LanguageOption("ಕನ್ನಡ", "kn_IN"),
    LanguageOption("한국어", "ko_KR"),
    LanguageOption("ລາວ", "lo_LA"),
    LanguageOption("Lietuvių", "lt_LT"),
    LanguageOption("Latviešu", "lv_LV"),
    LanguageOption("Македонски", "mk_MK"),
    LanguageOption("മലയാളം", "ml_IN"),
    LanguageOption("मराठी", "mr_IN"),
    LanguageOption("Bahasa Melayu", "ms_MY"),
    LanguageOption("नेपाली", "ne_NP"),
    LanguageOption("Nederlands", "nl_XX"),
    LanguageOption("Norsk", "no_XX"),
    LanguageOption("Polski", "pl_XX"),
    LanguageOption("Română", "ro_RO"),
    LanguageOption("සිංහල", "si_LK"),
    LanguageOption("Slovenčina", "sk_SK"),
    LanguageOption("Slovenščina", "sl_SI"),
    LanguageOption("Shqip", "sq_AL"),   
    LanguageOption("Turkish", "tr_TR"),
    LanguageOption("Tiếng Việt", "vi_VN")
]

def remove_unnecessary_spaces(text):
    """Removes unnecessary spaces between characters."""
    pattern = r"\)\s*\+\+|\)\+\+\s*"
    replacement = r")++"
    return re.sub(pattern, replacement, text)

def correct_translation_format(original_text, translated_text):
    original_parts = original_text.split('++')
    translated_parts = translated_text.split('++')
    
    corrected_parts = []
    for i, original_part in enumerate(original_parts):
        translated_part = translated_parts[i]
        
        original_plus_count = original_part.count('+')
        translated_plus_count = translated_part.count('+')
        plus_difference = translated_plus_count - original_plus_count
        
        if plus_difference > 0:
            translated_part = translated_part.replace('+' * plus_difference, '', 1)
        elif plus_difference < 0:
            translated_part += '+' * abs(plus_difference)
        
        corrected_parts.append(translated_part)
    
    corrected_text = '++'.join(corrected_parts)
    return corrected_text

def extract_plus_positions(text):
    """
    Given a string of text, extracts the positions of all sequences of one or more '+' characters.
    
    Args:
    - text (str): the input text
    
    Returns:
    - positions (list of lists): a list of [start, end, count] for each match, where start is the index of the
      first '+' character, end is the index of the last '+' character + 1, and count is the number of '+' characters
      in the match.
    """
    # Match any sequence of one or more '+' characters
    pattern = re.compile(r'\++')

    # Find all matches of the pattern in the text
    matches = pattern.finditer(text)

    # Loop through the matches and add their positions to the output list
    positions = []
    last_match_end = None
    for match in matches:
        if last_match_end is not None and match.start() != last_match_end:
            # If there is a gap between the current match and the previous one, add a new position
            j = last_match_end - 1
            while text[j] == "+":
                j -= 1
            j += 1
            positions.append([j, last_match_end, last_match_end - j])

        last_match_end = match.end()
    
    # If the final match extends to the end of the string, add its position to the output list
    if last_match_end is not None and last_match_end == len(text):
        j = last_match_end - 1
        while text[j] == "+":
            j -= 1
        j += 1
        positions.append([j, last_match_end, last_match_end - j])

    return positions


def match_pluses(original_text, translated_text):
    """
    Given two strings of text, replaces sequences of '+' characters in the second string with the corresponding
    sequences of '+' characters in the first string.
    
    Args:
    - original_text (str): the original text
    - translated_text (str): the translated text with '+' characters
    
    Returns:
    - output (str): the translated text with '+' characters replaced by those in the original text
    """
    in_positions = extract_plus_positions(original_text)
    out_positions = extract_plus_positions(translated_text)    
    
    out_vals = []
    out_current_pos = 0
    
    if len(in_positions) == len(out_positions):
        # Iterate through the positions and replace the sequences of '+' characters in the translated text
        # with those in the original text
        for in_, out_ in zip(in_positions, out_positions):
            out_vals.append(translated_text[out_current_pos:out_[0]])
            out_vals.append(original_text[in_[0]:in_[1]])
            out_current_pos = out_[1]
            
            # Check that the number of '+' characters in the original and translated sequences is the same
            if in_[2] != out_[2]:
                print("detected different + count")

    # Add any remaining text from the translated string to the output
    out_vals.append(translated_text[out_current_pos:])
    
    # Join the output values into a single string
    output = "".join(out_vals)
    return output

def post_process_prompt(original, translated):
    """Applies post-processing to the translated prompt such as removing unnecessary spaces and extra plus signs."""
    clean_prompt = remove_unnecessary_spaces(translated)
    clean_prompt = match_pluses(original, clean_prompt)
    #clean_prompt = remove_extra_plus(clean_prompt)
    return clean_prompt  

# 读取 csv 文件到内存中缓存起来
def load_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        cache = dict(reader)
    return cache

# 自定义翻译函数
def custom_translate(text, cache):
    if text in cache:
        return cache[text]
    else:
        return None




class Script(scripts.Script):
    def __init__(self) -> None:
        """Initializes the Script class and sets the default value for disable_translation attribute."""
        super().__init__()
        self.ln_code = "zh_CN"
        self.is_active=True
        self.is_negative_translate_active=False

    def title(self):
        """Returns the title of the script."""
        return "自动翻译提示词"

    def show(self, is_img2img):
        """Returns the visibility status of the script in the interface."""
        return scripts.AlwaysVisible
    
    def set_active(self, disable):
        """Sets the is_active attribute and initializes the translator object if not already created. 
        Also, sets the visibility of the language dropdown to True."""
        self.is_active=not disable
        if not disable and not hasattr(self, "translator"):
            if self.ln_code=="zh_CN":
                self.translator=ZhEnTranslator()
            else:
                self.translator = MBartTranslator()
        return "准备好了", self.output.update(visible=True)

    def set_negative_translate_active(self, negative_translate_active):
        """Sets the is_active attribute and initializes the translator object if not already created. 
        Also, sets the visibility of the language dropdown to True."""
        self.is_negative_translate_active=negative_translate_active

    def set_ln_code(self, language):
        # print("Devin Debug: set_ln_code",language)
        language_option = language_options[language]
        self.ln_code = language_option.language_code
        if self.ln_code=="zh_CN":
            self.translator=ZhEnTranslator()
        else:
            self.translator = MBartTranslator()

    def ui(self, is_img2img):
        """Sets up the user interface of the script."""
        self.current_axis_options = [x for x in language_options]

        with gr.Row():
            with gr.Column():
                with gr.Accordion("提示词翻译器",open=False):
                    with gr.Accordion("帮助",open=False):
                        gr.Markdown("""
                        # 描述
                        这个扩展可以让您使用母语直接编写提示词，无需翻译。
                        # 如何使用
                        默认开启翻译正面提示词，如果需要翻译负面提示词，请在下方勾选"翻译负面提示词"，如果需要关闭翻译，请在下方勾选"禁用翻译"。
                        # 注意事项
                        第一次启用脚本时可能需要很长时间下载翻译模型和加载模型，但一旦加载完成，它将更快。自定义提示词翻译在extensions/sd-prompt-translator/scripts/translation.csv中，您可以自行添加。
                        若有问题前往https://github.com/studyzy/sd-prompt-translator留言或者Email作者:studyzy@gmail.com
                        """)
                    with gr.Column():
                        self.disable_translation = gr.Checkbox(label="禁用翻译", value=False)
                        with gr.Column() as options:
                            self.options=options
                            self.translate_negative_prompt = gr.Checkbox(label="翻译负面提示词")
                            self.language = gr.Dropdown(
                                                label="源语言",
                                                choices=[x.label for x in self.current_axis_options],
                                                value="中文",
                                                type="index", 
                                                elem_id=self.elem_id("x_type")
                                            )
                        self.output=gr.Label("首次运行加载模型耗时较长，请耐心等待",visible=False)
                        self.disable_translation.change(
                            self.set_active,
                            [self.disable_translation],
                            [self.output, self.options], 
                            show_progress=True
                        )
                        self.translate_negative_prompt.change(
                            self.set_negative_translate_active,
                            [self.translate_negative_prompt], 
                        )
                        self.language.change(
                            self.set_ln_code,
                            [self.language],
                        )

        self.options.visible=True
        return [self.language]

    def get_prompts(self, p):
        """Returns the original prompts and negative prompts associated with a Prompt object."""
        original_prompts = p.all_prompts if len(p.all_prompts) > 0 else [p.prompt]
        original_negative_prompts = (
            p.all_negative_prompts
            if len(p.all_negative_prompts) > 0
            else [p.negative_prompt]
        )

        return original_prompts, original_negative_prompts

    def process_text(self,text):
        # 将中文全角标点符号替换为半角标点符号
        text = text.translate(str.maketrans('，。！？；：‘’“”（）【】', ',.!?;:\'\'\"\"()[]'))
        # 按逗号分割成数组
        text_array = text.split(',')
        # 对数组中每个字符串进行处理
        for i in range(len(text_array)):
            # 如果字符串以 < 开头 > 结尾，则是Lora，跳过不处理
            if text_array[i].startswith('<') and text_array[i].endswith('>'):
                continue
            # 判断是否只包含英文字符
            if all(char in string.printable + ' ' for char in text_array[i]):
                continue
            else:
                # 调用 transfer 函数进行翻译
                text_array[i] = self.transfer(text_array[i])
        # 重新用逗号连接成字符串并返回
        return ','.join(text_array)

    # 翻译函数
    def transfer(self,text):
        # 加载 csv 文件并缓存到内存中
        csv_path = os.path.join(os.path.dirname(__file__), 'translations.csv')
        cache = load_csv(csv_path)
        # 自定义翻译
        result = custom_translate(text, cache)
        if result is not None:
            return result
        else:
            # 调用 API 进行翻译
            en_prompt = self.translator.translate(text, self.ln_code, "en_XX")
            return en_prompt
    def process(self, p, language, **kwargs):
        """Translates the prompts from a non-English language to English using the MBartTranslator object."""
        if isinstance(language, int) and language >= 0:
            # 参数为大于等于0的整数
            language_option = language_options[language]
        else:
            # 参数不是大于等于0的整数
            language_option = language_options[0]
        self.ln_code = language_option.language_code
        if not hasattr(self, "translator") and self.is_active:
            if self.ln_code=="zh_CN":
                self.translator=ZhEnTranslator()
            else:
                self.translator = MBartTranslator()
        if hasattr(self, "translator") and self.is_active:
            original_prompts, original_negative_prompts = self.get_prompts(p)
            translated_prompts=[]
            previous_prompt = ""
            previous_translated_prompt = ""

            for original_prompt in original_prompts:
                if previous_prompt != original_prompt:
                    print(f"Translating prompt to English from {language_option.label}")
                    print(f"Initial prompt:{original_prompt}")

                    # translated_prompt = self.translator.translate(original_prompt, ln_code, "en_XX")
                    start_time = time.time()
                    translated_prompt = self.process_text(original_prompt)
                    translated_prompt = post_process_prompt(original_prompt, translated_prompt)
                    end_time = time.time()
                    print(f"Translated prompt:{translated_prompt}, spend time:{end_time-start_time}")
                    translated_prompts.append(translated_prompt)

                    previous_prompt=original_prompt
                    previous_translated_prompt = translated_prompt
                else:
                    translated_prompts.append(previous_translated_prompt)


            if p.negative_prompt!='' and self.is_negative_translate_active:
                previous_negative_prompt = ""
                previous_translated_negative_prompt = ""
                translated_negative_prompts=[]
                for negative_prompt in original_negative_prompts:
                    if previous_negative_prompt!=negative_prompt:
                        start_time = time.time()
                        print(f"Translating negative prompt to English from {language_option.label}")
                        print(f"Initial negative prompt:{negative_prompt}")
                        translated_negative_prompt = self.process_text(negative_prompt)
                        translated_negative_prompt = post_process_prompt(negative_prompt,translated_negative_prompt)
                        end_time = time.time()
                        print(f"Translated negative prompt:{translated_negative_prompt}, spend time:{end_time-start_time}")
                        translated_negative_prompts.append(translated_negative_prompt)


                        previous_negative_prompt = negative_prompt
                        previous_translated_negative_prompt = translated_negative_prompt
                    else:
                        translated_negative_prompts.append(previous_translated_negative_prompt)

                p.negative_prompt = translated_negative_prompts[0]
                p.all_negative_prompts = translated_negative_prompts
            p.prompt = translated_prompts[0]
            p.prompt_for_display = translated_prompts[0]
            p.all_prompts=translated_prompts

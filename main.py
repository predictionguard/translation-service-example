import os
import time
from pydantic import BaseModel
import yaml
import sys

from predictionguard import PredictionGuard
import pandas as pd
from comet import download_model, load_from_checkpoint
import deepl
from fastapi import FastAPI, HTTPException
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import uuid
import traceback
from openai import OpenAI
import munch
import huggingface_hub
import requests
from google.cloud import translate_v2 as translate

#--------------------------#
#         Config           #
#--------------------------#

ymlcfg = yaml.safe_load(open(os.path.join(sys.path[0],  'config.yml')))
cfg = munch.munchify(ymlcfg)

app = FastAPI()

# Hugging Face login
huggingface_hub.login(token=cfg.huggingface.token)
TOKENIZERS_PARALLELISM = cfg.huggingface.tokenizers_parallelism
os.environ['TOKENIZERS_PARALLELISM'] = str(TOKENIZERS_PARALLELISM)

# Get a list of all supported languages
supported_languages = []
for m in cfg.engines.keys():
    if m == "predictionguard" or m == "custom":
        for m_inner in cfg.engines[m].models:
            for l in cfg.engines[m].models[m_inner].languages:
                supported_languages.append(l)
    else:
        for l in cfg.engines[m].languages:
            supported_languages.append(l)

# Get a list of supported models
supported_models = []
for m in cfg.engines.keys():
    if m == "predictionguard" or m == "custom":
        for m_inner in cfg.engines[m].models:
            supported_models.append(m + "__" + m_inner)
    else:
        supported_models.append(m)


#-------------------------#
# LLM Translation Prompt  #
#-------------------------#

trans_prompt="""Translate the following {source_language} text to {target_language}. Only respond with the translation and no other text. Don't add, remove, or modify any information when translating.

{source_language} text: {input}

{target_language} translation:"""


#-------------------------#
# ISO Code Language Data  #
#-------------------------#

# Download the data
headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab'
r = requests.get(url, headers=headers)
with open('iso-639-3.tab', 'wb') as fh:
    fh.write(r.content)

# Read it into memory
iso = pd.read_csv('iso-639-3.tab', sep='\t')


#----------------------#
# COMET Quality Score  #
#----------------------#

# Download the COMET model
model_path = download_model(cfg.comet.model)
comet_model = load_from_checkpoint(model_path)

# Define the input and output models for COMET scoring
class QAInput(BaseModel):
    source: str
    translation: str

class QAOutput(BaseModel):
    score: float

# Function to get quality scores
def get_quality_score(input: QAInput):
    data = [{
        "src": input.source,
        "mt": input.translation,
    }]
    model_output = comet_model.predict(data, batch_size=8, gpus=0)
    return QAOutput(score=model_output.system_score)


#----------------------#
# MT APIs/ Models      #
#----------------------#

if "deepl" in cfg.engines.keys():

    # Create a map of language codes to deepl languages
    deepl_languages = {
        "ara": "AR",
        "bul": "BG",
        "cmn": "ZH",
        "ces": "CS",
        "dan": "DA",
        "nld": "NL",
        "eng": "EN-GB",
        "est": "ET",
        "fin": "FI",
        "fra": "FR",
        "deu": "DE",
        "hun": "HU",
        "ind": "ID",
        "ita": "IT",
        "jpn": "JA",
        "kor": "KO",
        "lav": "LV",
        "lit": "LT",
        "ell": "EL",
        "nor": "NB",
        "pol": "PL",
        "por": "PT-BR",
        "ron": "RO",
        "rus": "RU",
        "slk": "SK",
        "slv": "SL",
        "spa": "ES",
        "swe": "SV",
        "tur": "TR",
        "ukr": "UK"
    }

if "google" in cfg.engines.keys():

    # Create a map of ISO 639-3 language codes to Google Translate languages (ISO 639-1)
    google_languages = {
        "afr": "af",  # Afrikaans
        "sqi": "sq",  # Albanian
        "amh": "am",  # Amharic
        "ara": "ar",  # Arabic
        "hye": "hy",  # Armenian
        "asm": "as",  # Assamese
        "aym": "ay",  # Aymara
        "aze": "az",  # Azerbaijani
        "bam": "bm",  # Bambara
        "eus": "eu",  # Basque
        "bel": "be",  # Belarusian
        "ben": "bn",  # Bengali
        "bho": "bho", # Bhojpuri
        "bos": "bs",  # Bosnian
        "bul": "bg",  # Bulgarian
        "cat": "ca",  # Catalan
        "ceb": "ceb", # Cebuano
        "zho": "zh",  # Chinese
        "cos": "co",  # Corsican
        "hrv": "hr",  # Croatian
        "ces": "cs",  # Czech
        "dan": "da",  # Danish
        "div": "dv",  # Dhivehi
        "doi": "doi", # Dogri
        "nld": "nl",  # Dutch
        "eng": "en",  # English
        "epo": "eo",  # Esperanto
        "est": "et",  # Estonian
        "ewe": "ee",  # Ewe
        "fil": "tl",  # Filipino (Tagalog)
        "fin": "fi",  # Finnish
        "fra": "fr",  # French
        "fry": "fy",  # Frisian
        "glg": "gl",  # Galician
        "kat": "ka",  # Georgian
        "deu": "de",  # German
        "ell": "el",  # Greek
        "grn": "gn",  # Guarani
        "guj": "gu",  # Gujarati
        "hat": "ht",  # Haitian Creole
        "hau": "ha",  # Hausa
        "haw": "haw", # Hawaiian
        "heb": "he",  # Hebrew
        "hin": "hi",  # Hindi
        "hmn": "hmn", # Hmong
        "hun": "hu",  # Hungarian
        "isl": "is",  # Icelandic
        "ibo": "ig",  # Igbo
        "ilo": "ilo", # Ilocano
        "ind": "id",  # Indonesian
        "gle": "ga",  # Irish
        "ita": "it",  # Italian
        "jpn": "ja",  # Japanese
        "jav": "jv",  # Javanese
        "kan": "kn",  # Kannada
        "kaz": "kk",  # Kazakh
        "khm": "km",  # Khmer
        "kin": "rw",  # Kinyarwanda
        "gom": "gom", # Konkani
        "kor": "ko",  # Korean
        "kri": "kri", # Krio
        "kur": "ku",  # Kurdish
        "ckb": "ckb", # Kurdish (Sorani)
        "kir": "ky",  # Kyrgyz
        "lao": "lo",  # Lao
        "lat": "la",  # Latin
        "lav": "lv",  # Latvian
        "lin": "ln",  # Lingala
        "lit": "lt",  # Lithuanian
        "lug": "lg",  # Luganda
        "ltz": "lb",  # Luxembourgish
        "mkd": "mk",  # Macedonian
        "mai": "mai", # Maithili
        "mlg": "mg",  # Malagasy
        "msa": "ms",  # Malay
        "mal": "ml",  # Malayalam
        "mlt": "mt",  # Maltese
        "mri": "mi",  # Maori
        "mar": "mr",  # Marathi
        "mni": "mni-Mtei", # Meiteilon (Manipuri)
        "lus": "lus", # Mizo
        "mon": "mn",  # Mongolian
        "mya": "my",  # Burmese
        "nep": "ne",  # Nepali
        "nob": "no",  # Norwegian
        "nya": "ny",  # Nyanja (Chichewa)
        "ori": "or",  # Odia (Oriya)
        "orm": "om",  # Oromo
        "pus": "ps",  # Pashto
        "fas": "fa",  # Persian
        "pol": "pl",  # Polish
        "por": "pt",  # Portuguese
        "pan": "pa",  # Punjabi
        "que": "qu",  # Quechua
        "ron": "ro",  # Romanian
        "rus": "ru",  # Russian
        "smo": "sm",  # Samoan
        "san": "sa",  # Sanskrit
        "gla": "gd",  # Scots Gaelic
        "nso": "nso", # Sepedi
        "srp": "sr",  # Serbian
        "sot": "st",  # Sesotho
        "sna": "sn",  # Shona
        "snd": "sd",  # Sindhi
        "sin": "si",  # Sinhala
        "slk": "sk",  # Slovak
        "slv": "sl",  # Slovenian
        "som": "so",  # Somali
        "spa": "es",  # Spanish
        "sun": "su",  # Sundanese
        "swa": "sw",  # Swahili
        "swe": "sv",  # Swedish
        "tgl": "tl",  # Tagalog (Filipino)
        "tgk": "tg",  # Tajik
        "tam": "ta",  # Tamil
        "tat": "tt",  # Tatar
        "tel": "te",  # Telugu
        "tha": "th",  # Thai
        "tir": "ti",  # Tigrinya
        "tso": "ts",  # Tsonga
        "tur": "tr",  # Turkish
        "tuk": "tk",  # Turkmen
        "twi": "ak",  # Twi (Akan)
        "ukr": "uk",  # Ukrainian
        "urd": "ur",  # Urdu
        "uig": "ug",  # Uyghur
        "uzb": "uz",  # Uzbek
        "vie": "vi",  # Vietnamese
        "cym": "cy",  # Welsh
        "xho": "xh",  # Xhosa
        "yid": "yi",  # Yiddish
        "yor": "yo",  # Yoruba
        "zul": "zu"   # Zulu
    }

def deepl_translation(text, target_language):

    # Process target language code
    target_language = deepl_languages[target_language]

    # Initialize the deepl translator
    deepl_translator = deepl.Translator(auth_key=cfg.engines.deepl.api_key)

    # Get the translation
    response = deepl_translator.translate_text(text, target_lang=target_language).text

    # Process the response
    if response is not None and response.strip():
        qa_input = QAInput(source=text, translation=response)
        score = get_quality_score(qa_input).score
        return {
            "translation": response, 
            "score": score, 
            "model": "deepl", 
            "status": "success"
        }
    else:
        return {
            "translation": "", 
            "score": -100, 
            "model": "deepl", 
            "status": "error: could not get translation"
        }




# Function to handle Google Cloud Translation
def google_translation(text, source_language, target_language):

    # Process target language code
    target_language = google_languages[target_language]
    source_language = google_languages[source_language]

    # Initialize the Google Cloud Translation client
    google_translate_client = translate.Client(api_key=cfg.engines.google.api_key)

    try:
        # Translate text using Google Cloud Translation API
        response = google_translate_client.translate(
            text,
            source_language=source_language,
            target_language=target_language
        )
        translated_text = response.get('translatedText')
        if translated_text:
            qa_input = QAInput(source=text, translation=translated_text)
            score = get_quality_score(qa_input).score
            return {"translation": translated_text, "score": score, "model": "google", "status": "success"}
        else:
            return {"translation": "", "score": -100, "model": "google", "status": "error: couldn’t get translation"}
    except Exception as e:
        print(e)
        return {"translation": "", "score": -100, "model": "google", "status": "error: couldn’t get translation"}
    
def pg_openai_translation(text, source_language, target_language, model):

    # Initialize the client
    if "gpt" in model:
        client = OpenAI(api_key=cfg.engines.openai.api_key)
    else:
        client = PredictionGuard(api_key=cfg.engines.predictionguard.api_key)

    # Call the API
    result = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user", 
            "content": trans_prompt.format(
                input=text, 
                source_language=source_language, 
                target_language=target_language
            )
        }],
        temperature=0.1
    )

    # Process the response
    response_message = result['choices'][0]['message']['content'].strip().split('\n')[0]
    if response_message:
        qa_input = QAInput(source=text, translation=response_message)
        score = get_quality_score(qa_input).score
        return {
            "translation": response_message, 
            "score": score, 
            "model": "openai", 
            "status": "success"}
    else:
        return {
            "translation": "", 
            "score": -100, "model": 
            "openai", "status": 
            "error: could not get translation"
        }
    

def custom_translation(text, source_language, target_language, model):

    # TODO: Make source language in the JSON body optional

    # Call the API
    headers = {'x-api-key': cfg.custom.models['model'].api_key}
    url = cfg.custom.models[model].url
    response = requests.post(
        url, 
        json={
            'model': model,
            'text': text, 
            'source_language': source_language, 
            'target_language': target_language
        }, 
        headers=headers)
    response = response.json()

    # Process the response
    if 'translation' in response.keys() and len(response['translation']) > 0:
        qa_input = QAInput(source=text, translation=response['translation'])
        score = get_quality_score(qa_input).score
        return {
            "translation": response['translation'], 
            "score": score, 
            "model": "custom",
            "status": "success"}
    else:
        return {
            "translation": "", 
            "score": -100, 
            "model": "custom", 
            "status": "error: could not get translation"
        }
        

#-----------------------------------------#
# Concurrent Translation functionality    #
#-----------------------------------------#

def translate_and_score(text, source_language_iso639, target_language_iso639):
    translation_results = []
    best_translation = None
    best_score = -1
    best_model = ""

    created_timestamp = int(time.time())
    unique_id = "translation-" + str(uuid.uuid4()).replace("-", "")

    # filter supported models based on language codes
    supported_models_filtered = []
    for model in supported_models:
        if "predictionguard" in model or "custom" in model:
            engine_type = model.split('__')[0]
            pg_langs = cfg.engines[engine_type]['models'][model.split('__')[-1]].languages
            if target_language_iso639 in pg_langs and source_language_iso639 in pg_langs:
                supported_models_filtered.append(model)
        else:
            other_langs = cfg.engines[model].languages
            if target_language_iso639 in other_langs and source_language_iso639 in other_langs:
                supported_models_filtered.append(model)

    def process_translation(model):
        try:
            if model == "deepl":
                result = deepl_translation(text, target_language_iso639)
            elif "predictionguard" in model:
                result = pg_openai_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    model.split('__')[-1]
                )
            elif model == "openai":
                result = pg_openai_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    cfg.engines.openai.model
                )
            elif model == "google":
                result = google_translation(text, source_language_iso639, target_language_iso639)
            elif "custom" in model:
                result = custom_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    model.split('__')[-1]
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
            return result
        except Exception as e:
            print(f"Error translating with {model}: {e}")
            traceback.print_exc()
            return {
                "score": 0,
                "translation": "",
                "model": model,
                "status": f"error: {str(e)}"
            }

    with ThreadPoolExecutor(max_workers=len(supported_models_filtered)) as executor:
        futures = [executor.submit(process_translation, model) for model in supported_models]
        for future in futures:
            result = future.result()
            translation_results.append(result)
            if result["status"] == "success" and result["score"] > best_score:
                best_translation = result["translation"]
                best_score = result["score"]
                best_model = result["model"]

    output = {
        "translations": translation_results,
        "best_translation": best_translation if best_translation else "We don't support the requested language pair",
        "best_score": best_score,
        "best_translation_model": best_model,
        "created": created_timestamp,
        "id": unique_id,
        "object": "translation"
    }

    return output

#---------------------#
# FastAPI app         #
#---------------------#

@app.get("/")
def read_root():
    return {"status": "healthy"}


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def is_valid_language(language_code):
    return language_code in supported_languages


@app.post("/translate")
def update_item(req: TranslateRequest):
    if not is_valid_language(req.source_lang) or not is_valid_language(req.target_lang):
        raise HTTPException(status_code=400, detail="Invalid language code(s)")

    # Now you can proceed with the translations
    return translate_and_score(req.text, req.source_lang, req.target_lang)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
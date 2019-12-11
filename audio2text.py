
import json

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Instantiates a client
client = speech.SpeechClient()

# Loads the audio into memory
import io
import os


def recognize_google_cloud(path, phrases=None):
    with io.open(path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    DEFAULT = ['1', '2', '3', '4', '一', '二', '三', '四']
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='zh-tw',
        speech_contexts=[speech.types.SpeechContext(
            phrases=DEFAULT+phrases if phrases else DEFAULT
        )]
    )
    resp = client.recognize(config=config, audio=audio)
    return [alter.transcript for alter in resp.results[0].alternatives]
    #
    # return resp.results[0].alternatives[0].transcript
    # # for result in client.recognize(config, audio):
    # #     return result.alternatives[0].transcript


def batch_recognize(path, choice_analyze=False, words_a=[]):
    result = []
    cnt = 0
    for root, dirs, files in os.walk(path, topdown=True):
        files = sorted(files)
        for name in files:
            path = os.path.join(root, name)
            print(path)
            print(words_a[cnt]['word']if words_a else None)
            results = recognize_google_cloud(path, words_a[cnt]['word'] if words_a else None)
            cnt += 1
            if choice_analyze:
                best, tmp = None, None
                for text in results:
                    op1 = split_option(text)
                    op2 = split_option_rev(text)
                    tmp = op1 if op1.count('') < op2.count('') else op2
                    if best:
                        best = tmp if tmp.count('') < best.count('') else best
                    else:
                        best = tmp
                    if best.count('') == 0:
                        break
                result.append(best)
            else:
                result.append(results[0])
            print(result[-1])
    return result


def split_option(text):
    opt = ['', '', '', '']
    prev = None
    idx = 0
    for i, char in zip(range(len(text)), text):
        if char in ['二', '2', '貳', '餓', '惡'] and idx == 0:  # opt 1
            opt[idx] = text[1:i] if text[0] in ['1', '一', '壹'] else text[:i]
            prev = i
            idx += 1
            i += 1
        elif char in ['三', '3', '參'] and idx == 1:  # opt 2
            opt[idx] = text[prev+1:i]
            prev = i
            idx += 1
            i += 1
        elif char in ['4', '四', '是', '似', '賜', '適'] and idx == 2:  # opt 3 & 4
            opt[idx] = text[prev+1:i]
            idx += 1
            i += 1
            opt[idx] = text[i+1:]
    return opt


def split_option_rev(text):
    opt = ['', '', '', '']
    prev = None
    idx = 3
    for i in range(len(text) - 1, 0, -1):
        char = text[i]
        if char in ['4', '四', '是', '似', '賜', '適'] and idx == 3:
            opt[idx] = text[i + 1::]
            prev = i
            idx -= 1
            i -= 1
        elif char in ['三', '3', '參'] and idx == 2:
            opt[idx] = text[i + 1:prev]
            prev = i
            idx -= 1
            i -= 1
        elif char in ['二', '2', '貳', '餓', '惡'] and idx == 1:
            opt[idx] = text[i + 1:prev]
            opt[0] = text[1:i] if text[0] in ['1', '一', '壹'] else text[0:i]
            i -= 1
    return opt


if __name__ == '__main__':
    SOURCE_FOLDER = 'audio/formal/'

    with open('data/formal/A_gcp_1.json', 'r') as f:
        data = json.load(f)

    result = batch_recognize(SOURCE_FOLDER + 'B')
    new_data = []
    for item, q in zip(data, result):
        item['question'] = q
        new_data.append(item)

    data = []
    result = batch_recognize(SOURCE_FOLDER + 'C', choice_analyze=True)
    for item, q in zip(new_data, result):
        item['op1'] = q[0]
        item['op2'] = q[1]
        item['op3'] = q[2]
        item['op4'] = q[3]
        data.append(item)

    with open('data/formal/total_1.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False)
# if __name__ == '__main__':
# a_audios = read_wav_dir(os.path.join(SOURCE_FOLDER, 'A'))
# print(r.recognize_google_cloud(a_audios[0], language='zh-TW', credentials_json=json.dumps(GOOGLE_TOKEN)))
# result = recognition_google(a_audios)
# with open('data/formal/A.json', 'w') as f:
#     json.dump([{'id': i+1,
#                 'content': text} for i, text in enumerate(result)], f, ensure_ascii=False)

# a = recognition_google(read_wav_dir(os.path.join(SOURCE_FOLDER, 'A')))
# print(a)

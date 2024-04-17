# MobileVLA: Vision Language Action Model for Mobile Devices

[![Apache License](https://img.shields.io/badge/license-Apache-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

MobileVLA is expected to be a smart assistant by deploying on mobile devices.



## Prerequisite:

### Step 0

Follow the instructions in the [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) and [CALVIN](https://github.com/mees/calvin) to download the necessary dataset and VLM pretrained Models.

Download the [CALVIN](https://github.com/mees/calvin) dataset

```
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh D | ABC | ABCD | debug
```

Download the released [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) model:

#### MobileVLM Family

| Model                                                        | LLM                                                          | GQA      | SQA<sup>I</sup> | VQA<sup>T</sup> | POPE     | MME<sup>P</sup> | MMB<sup>dev</sup> | Avg.     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | --------------- | --------------- | -------- | --------------- | ----------------- | -------- |
| <div style="width: 93pt"> [MobileVLM-1.7B](https://huggingface.co/mtgv/MobileVLM-1.7B) | <div style="width: 91pt"> [MobileLLaMA 1.4B](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat) | 56.1     | 57.3            | 41.5            | 84.5     | 1196.2          | 53.2              | 58.7     |
| [MobileVLM V2 1.7B](https://huggingface.co/mtgv/MobileVLM_V2-1.7B) | [MobileLLaMA 1.4B](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat) | **59.3** | **66.7**        | **52.1**        | **84.3** | **1302.8**      | **57.7**          | **64.2** |
| [MobileVLM-3B](https://huggingface.co/mtgv/MobileVLM-3B)     | [MobileLLaMA 2.7B](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Chat) | 59.0     | 61.2            | 47.5            | 84.9     | 1288.9          | 59.6              | 62.8     |
| [MobileVLM V2 3B](https://huggingface.co/mtgv/MobileVLM_V2-3B) | [MobileLLaMA 2.7B](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Chat) | **61.1** | **70.0**        | **57.5**        | **84.7** | **1440.5**      | **63.2**          | **68.1** |
| [MobileVLM V2 7B](https://huggingface.co/mtgv/MobileVLM_V2-7B) | [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)     | **62.6** | **74.8**        | **62.3**        | **85.3** | **1560.7**      | **69.2**          | **72.1** |

#### Step 1

Clone this repo

```
git clone https://github.com/Evan-wyl/MobileVLA
```





## Acknowledgment

#### CALVIN

Original: https://github.com/mees/calvin License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### Meituan MobileVLM

Original: https://github.com/Meituan-AutoML/MobileVLM License: [Apache-2.0](https://github.com/Meituan-AutoML/MobileVLM/blob/main/LICENSE)

#### ByteDance RoboFlamingo

Original: https://github.com/RoboFlamingo/RoboFlamingo License: [MIT](https://github.com/RoboFlamingo/RoboFlamingo/blob/main/LICENSE)

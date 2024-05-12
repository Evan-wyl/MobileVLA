# MobileVLA: Vision Language Action Model for Mobile Devices

[![Apache License](https://img.shields.io/badge/license-Apache-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

MobileVLA is expected to be a smart assistant by deploying on mobile devices.



## Usage

### Initializing a RoboFlamingo model

We support pre-trained vision encoders from the [OpenCLIP](https://github.com/mlfoundations/open_clip) package, which includes OpenAI's pre-trained models. 
We also support pre-trained language models from the `transformers` package, such as [LLaMA](https://huggingface.co/models?search=llama) models.

``` python
from mobilevla.factory import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="PATH/TO/LLM/DIR",
    tokenizer_path="PATH/TO/LLM/DIR",
    cross_attn_every_n_layers=1,
    decoder_type='lstm',
)
```

The `cross_attn_every_n_layers` argument controls how often cross-attention layers are applied and should be consistent with the VLM. The `decoder_type` argument controls the type of the decoder, currently, we support `lstm`, `fc`, `diffusion` (bugs exist for the dataloader), and `GPT`.

## Performance

We report results on the [CALVIN](https://github.com/mees/calvin) benchmark.

| Method                    | Training Data | Test Split | 1     | 2     | 3     | 4     | 5     | Avg Len |
| ------------------------- | ------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ------- |
| MCIL                      | ABCD (Full)   | D          | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40    |
| HULC                      | ABCD (Full)   | D          | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06    |
| HULC (retrained)          | ABCD (Lang)   | D          | 0.892 | 0.701 | 0.548 | 0.420 | 0.335 | 2.90    |
| RT-1 (retrained)          | ABCD (Lang)   | D          | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45    |
| RoboFlamingo              | ABCD (Lang)   | D          | 0.964 | 0.896 | 0.824 | 0.740 | 0.66  | 4.09    |
| Ours                      |               |            |       |       |       |       |       |         |
| MCIL                      | ABC (Full)    | D          | 0.304 | 0.013 | 0.002 | 0.000 | 0.000 | 0.31    |
| HULC                      | ABC (Full)    | D          | 0.418 | 0.165 | 0.057 | 0.019 | 0.011 | 0.67    |
| RT-1 (retrained)          | ABC (Lang)    | D          | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90    |
| RoboFlamingo              | ABC (Lang)    | D          | 0.824 | 0.619 | 0.466 | 0.331 | 0.235 | 2.48    |
| Ours                      |               |            |       |       |       |       |       |         |
| HULC                      | ABCD (Full)   | D (Enrich) | 0.715 | 0.470 | 0.308 | 0.199 | 0.130 | 1.82    |
| RT-1 (retrained)          | ABCD (Lang)   | D (Enrich) | 0.494 | 0.222 | 0.086 | 0.036 | 0.017 | 0.86    |
| RoboFlamingo              | ABCD (Lang)   | D (Enrich) | 0.720 | 0.480 | 0.299 | 0.211 | 0.144 | 1.85    |
| RoboFlamingo (freeze-emb) | ABCD (Lang)   | D (Enrich) | 0.737 | 0.530 | 0.385 | 0.275 | 0.192 | 2.12    |
| Ours                      |               |            |       |       |       |       |       |         |

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
git clone -b robo-vlm https://github.com/Evan-wyl/MobileVLA.git
```

Install the required packages:

```
cd MobileVLA
conda create -n MobileVLA python=3.8
source activate MobileVLA
pip install -r requirements.txt
```



## Training the model (using DDP):

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 mobilevla/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mobilellama-1.4b \
    --use_gripper \
    --fusion_mode post \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --batch_size_calvin 6 \
    --run_name MobileVLA-LSTM \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --mm_projector_type ${mm_projector_type} \
    --cross_attn_every_n_layers 4 \
    --loss_multiplier_calvin 1.0 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 > ${log_file} 2>&1
```

`${calvin_dataset_path}` is the path to the CALVIN dataset；

`${lm_path}` is the path to the pre-trained LLM；

`${tokenizer_path}` is the path to the VLM tokenizer；

`${openflamingo_checkpoint}` is the path to the OpenFlamingo pre-trained model；

`${log_file}` is the path to the log file.

We also provide `mobilevla/pt_run_gripper_post_ws_12_traj_aug_mobilellama_1b.bash` to launch the training. 



## Evaluating the model on the CALVIN benchmark

```
python eval_ckpts.py
```

By adding the checkpoint name and directory into `eval_ckpts.py`, the script would automatically load the model and evaluate it. For example, if you want to evaluate the checkpoint at path 'your-checkpoint-path', you can modify the `ckpt_dir` and `ckpt_names` variables in eval_ckpts.py, and the evaluation results would be saved as 'logs/your-checkpoint-prefix.log'.



## Co-finetune with both robot data (CALVIN) and vision-language data (COCO caption, VQA)

The results shown below indicate that co-training could preserve most ability of the VLM backbone on VL tasks, while losing a bit of performance on robot tasks. 

use

```
bash robot_flamingo/pt_run_gripper_post_ws_12_traj_aug_mobilellama_1b_co_train.bash
```

to launch co-train RoboFlamingo with CoCO, VQAV2 and CALVIN. You should update CoCO and VQA paths in `get_coco_dataset` and `get_vqa_dataset` in `mobilevla/data/data.py`.

### Results on the CALVIN benchmark:

| Split     | SR 1             | SR 2 | SR 3 | SR 4 | SR 5 | Avg Len |
| --------- | ---------------- | ---- | ---- | ---- | ---- | ------- |
| Co-Train  | ABC->D           |      |      |      |      |         |
| Fine-tune | ABC->D           |      |      |      |      |         |
| Co-Train  | ABCD->D          |      |      |      |      |         |
| Fine-tune | ABCD->D          |      |      |      |      |         |
| Co-Train  | ABCD->D (Enrich) |      |      |      |      |         |
| Fine-tune | ABCD->D (Enrich) |      |      |      |      |         |

### Results on VL tasks:

|                                     |        |        |        | coco   |        |         |       |       | VQA  |
| ----------------------------------- | ------ | ------ | ------ | ------ | ------ | ------- | ----- | ----- | ---- |
|                                     | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE | Acc  |
| Fine-tune (3B, zero-shot)           |        |        |        |        |        |         |       |       |      |
| Fine-tune (3B, 4-shot)              |        |        |        |        |        |         |       |       |      |
| Co-Train (3B, zero-shot)            |        |        |        |        |        |         |       |       |      |
| Original Flamingo (80B, fine-tuned) |        |        |        |        |        |         |       |       |      |



## Acknowledgment

#### CALVIN

Original: https://github.com/mees/calvin License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### Meituan MobileVLM

Original: https://github.com/Meituan-AutoML/MobileVLM License: [Apache-2.0](https://github.com/Meituan-AutoML/MobileVLM/blob/main/LICENSE)

#### ByteDance RoboFlamingo

Original: https://github.com/RoboFlamingo/RoboFlamingo License: [MIT](https://github.com/RoboFlamingo/RoboFlamingo/blob/main/LICENSE)

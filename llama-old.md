# r/LocalLLaMA

LocalLLaMA is a subreddit to discuss about Meta AI's Llama. It was created to foster a community around Llama similar to communities dedicated to open source projects like Stable Diffusion. While the primary focus of this community is on Llama, discussion of other locally run LLMs is welcome.

This wiki is **not** meant to be comprehensive but serve as simplified source to direct subreddit newcomers on quickly getting started with running Llama locally:

- If you're new to Llama and need basic direction on where to start, continue reading.

- If you're already familiar with Llama or local LLMs, you can skip reading this.

It's important to note that the documentation for the projects linked below contain more complete information and should answer almost all questions when considered altogether.

If you have any suggestions for additions to this page, you can [send a message to modmail](https://www.reddit.com/message/compose?to=/r/LocalLLaMA).

^(Disclaimer: r/LocalLLaMA does not claim responsibility for any models, groups, or third party information listed or linked herein.)

---

## Llama basics FAQ

A short collection of very frequently asked questions in this subreddit about Llama.

**Q:** What is Llama?

**A:** Llama, formerly known as LLaMA (Large Language Model Meta AI), is a family of large language models. The data used to train the models was collected from various sources, mostly from the Web. The released models range in scale in terms of parameters:

- 7B, 13B, 33B^(*), and 65B for Llama 1

- 7B, 13B, and 70B for Llama 2

- 7B, 13B, and 34B for Code Llama

^(*30B and 33B have been used interchangeably when referring to the same 32.5B model)

To learn more, read the papers for [Llama 1](https://arxiv.org/abs/2302.13971) and [Llama 2](https://arxiv.org/abs/2307.09288).

**Q:** Is Llama like ChatGPT?

**A:** Currently, Llama and similar LLMs available to use locally cannot match the quality of GPT-4 for most tasks, but they can produce results resembling GPT-3.5. This can be seen in fine-tuned Llama models, such as the Llama Chat models by Meta that were fine-tuned for chat use cases.

The pretrained Llama 1 and Llama 2 models released as-is are not fine-tuned for question answering or dialogue use cases. These foundation models should be prompted so that the expected answer is the natural continuation of the prompt.

**Q:** Can Llama models be used commercially?

**A:** Llama 2 models are free for commercial use. For more information, read the [Llama 2 license](https://ai.meta.com/llama/license/).

**Q:** Can I try Llama online?

**A:** If you want to try Llama before going through the local installation process, [HuggingChat](https://huggingface.co/chat/) offers a few options online.

**Q:** Can my PC run Llama models?

**A:** The answer is most likely yes. Even on lower end computers, you can run Llama-based models as long as you meet the llama.cpp requirements. With llama.cpp, you need adequate disk space to save models and sufficient RAM to load them, with memory and disk requirements being the same. For example, this chart lists the minimum requirements for 4-bit quantization:

|Model|Original Size|Quantized Size (4-bit)|
|:-|:-|:-|
|7B|13 GB|3.9 GB|
|13B|24 GB|7.8 GB|
|33B|60 GB|19.5 GB|
|65B|120 GB|38.5 GB|

RAM requirements vary depending on factors like context size, model size, and quantization. These requirements can be offset by offloading layers to the GPU.

**Q:** What is the easiest way to get started?

**A:** Clone or download the text generation web UI repository linked below and run the corresponding start script for your OS. After that, all you have to do is download and load a model. The web UI is highly intuitive to use and has a wiki that explains all settings.

**Q:** What is the best model to use?

**A:** There's no single answer to this question. Overall, Llama 2 70B models represent the best that can be run locally in the Llama family. There are many models fine-tuned for varying use cases, and it's recommended to experiment with different models to find your desired option.

**Q:** Where can I find other guides involving Llama, like AMD usage and fine-tuning?

**A:** Many tutorials and guides can be found in this subreddit and [searched by flair](https://www.reddit.com/r/LocalLLaMA/search/?sort=new&restrict_sr=on&q=flair%3A%22Tutorial%20|%20Guide%22) or by keywords. Not all guides use the `Tutorial | Guide` flair, and some may be under `Resources` or `Discussion`. There are numerous posts and comments spanning months across a wide range of topics.

**Q:** How can I keep up with news about Llama and local LLMs?

**A:** For Llama, the official news source is the [Meta AI blog](https://ai.meta.com/blog/), and any news there will be posted here. For research revolving around LLMs in general, [you can search arXiv](https://arxiv.org/list/cs.CL/recent) on the subjects of Computation and Language, Artificial Intelligence, and Machine Learning. Notable papers and new projects are usually posted in this subreddit.

**Q:** Is Mistral the same as Llama?

**A:** No. MistralAI is an independent company founded by researchers who worked on Llama 1, and they have committed to releasing new foundation models as part of their Mistral family of LLMs.

---

## Resources

### Community

Since the unveil of Llama in February 2023, thorough documentation has been created across community projects and making use of the Llama ecosystem is easier than before. There are several main projects that are used to fulfill a variety of needs. Visit each page for more information.

---

[**Text generation web UI**](https://github.com/oobabooga/text-generation-webui)

A Gradio web UI for large language models. The ideal option if you're used to Stable Diffusion web UI and want a similar interface. 

Multiple model backends: transformers, llama.cpp, ExLlama, ExLlamaV2, AutoGPTQ, GPTQ-for-LLaMa, CTransformers, AutoAWQ

[Installation](https://github.com/oobabooga/text-generation-webui#installation)

[Full documentation](https://github.com/oobabooga/text-generation-webui/wiki)

---

[**llama.cpp**](https://github.com/ggerganov/llama.cpp)

> Port of Facebook's LLaMA model in C/C++

[Usage](https://github.com/ggerganov/llama.cpp#usage)

[Running on Windows with prebuilt binaries](https://github.com/ggerganov/llama.cpp#running-on-windows-with-prebuilt-binaries)

[Additional documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

---

[**KoboldCpp**](https://github.com/LostRuins/koboldcpp)

> A simple one-file way to run various GGML and GGUF models with KoboldAI's UI

[Documentation](https://github.com/LostRuins/koboldcpp/wiki)

---

[**vLLM**](https://github.com/vllm-project/vllm)

> A high-throughput and memory-efficient inference and serving engine for LLMs

[Documentation](https://vllm.readthedocs.io/en/latest/)

---

[**MLC LLM**](https://github.com/mlc-ai/mlc-llm)

> A high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration

[Documentation](https://llm.mlc.ai/docs)

---

[**Text Generation Inference**](https://github.com/huggingface/text-generation-inference)

> A Rust, Python and gRPC server for text generation inference

[Documentation](http://hf.co/docs/text-generation-inference)

---

#### Extra

[**FastChat**](https://github.com/lm-sys/FastChat)

> An open platform for training, serving, and evaluating large language model based chatbots

[**PEFT**](https://github.com/huggingface/peft)

> State-of-the-art Parameter-Efficient Fine-Tuning

[**QLoRA**](https://github.com/artidoro/qlora)

> Efficient Finetuning of Quantized LLMs

---

### Models

[Hugging Face](https://huggingface.co/) is where you can find and download Llama models and other LLMs. Search by specific model name or by appellation like GPTQ, GGUF, EXL2, or AWQ. GGML is deprecated. For more in-depth information on models and model loaders, [**read this wiki entry**](https://github.com/oobabooga/text-generation-webui/wiki/04-%E2%80%90-Model-Tab#model-loaders).

GGUF is the most widely used. In general quality and filesize, from worst to best and smallest to largest, respectively, follows this ascending order: Q2K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16. You do not need all of these, only one.

For keeping track of new models, most popular ones are posted in this subreddit. You can search by the [New Model flair](https://www.reddit.com/r/LocalLLaMA/search/?sort=new&restrict_sr=on&q=flair%3A%22New%20%20Model%22) to find these posts.

#### Quantize

With the resources available, quantization is generally an easy process that can be done by anyone.

[**GPTQ**](https://huggingface.co/docs/transformers/main/main_classes/quantization)

[**GGUF**](https://github.com/ggerganov/llama.cpp#prepare-data--run)

[**EXL2**](https://github.com/turboderp/exllamav2#exl2-quantization)

[**AWQ**](https://github.com/casper-hansen/AutoAWQ#examples)

#### What's the best model for...?

The answer to this question can vary depending on personal preference, but here are some good recommendations to start with:

**Chatting like ChatGPT**

The official Llama 2 Chat models by Meta can be considered some of the best for assistant chatting like ChatGPT.

|7B|[Llama 2 7B Chat](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)|
|:-:|:-:|
|13B|[Llama 2 13B Chat](https://huggingface.co/NousResearch/Llama-2-13b-chat-hf)|
|70B|[Llama 2 70B Chat](https://huggingface.co/NousResearch/Llama-2-70b-chat-hf)|

**Coding**

[*See Code Llama blog for more details*](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

|7B|[Code Llama 7B Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)|
|:-:|:-:|
|13B|[Code Llama 13B Instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)|
|34B|[Phind Code Llama 34B v2](https://huggingface.co/phind/Phind-CodeLlama-34B-v2)|

Leaderboards for coding can be seen in the [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html), [Bigcode Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard), and [Can AI Code Results](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results).

**Math**

[*See ToRA page for more details*](https://github.com/microsoft/ToRA)

|7B|[ToRA Code 7B v1.0](https://huggingface.co/llm-agents/tora-code-7b-v1.0)|
|:-:|:-:|
|13B|[ToRA Code 13B v1.0](https://huggingface.co/llm-agents/tora-code-13b-v1.0)|
|34B|[ToRA Code 34B v1.0](https://huggingface.co/llm-agents/tora-code-34b-v1.0)|
|70B|[ToRA 70B v1.0](https://huggingface.co/llm-agents/tora-70b-v1.0)|

**Medical**

[*See Clinical Camel paper for more details*](https://arxiv.org/abs/2305.12031)

|7B|[MedAlpaca 7B](https://huggingface.co/medalpaca/medalpaca-7b)|
|:-:|:-:|
|13B|[qCammel-13](https://huggingface.co/augtoma/qCammel-13) or [MedAlpaca 13B](https://huggingface.co/medalpaca/medalpaca-13b)|
|70B|[Clinical Camel 70B](https://huggingface.co/wanglab/ClinicalCamel-70B)|

**Visual Instruction**

LLaVA^(*)

[*See LLaVA page for more details*](https://github.com/haotian-liu/LLaVA)

|7B|[LLaVA v1.5 7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)|
|:-:|:-:|
|13B|[LLaVA v1.5 13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)|

^(*LLaVA is supported in text generation web UI as an extension and llama.cpp)

Qwen VL

[*See Qwen VL page for more details*](https://github.com/QwenLM/Qwen-VL)

|7B|[Qwen VL 7B](https://huggingface.co/Qwen/Qwen-VL)|
|:-:|:-:|
|7B|[Qwen VL Chat 7B](https://huggingface.co/Qwen/Qwen-VL-Chat)|

InstructBLIP^(*)

[*See InstructBLIP page for more details*](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md)

^(*InstructBLIP uses Vicuna 7B and 13B models.)

IDEFICS^(*)

[*See IDEFICS blog for more details*](https://huggingface.co/blog/idefics)

|9B|[IDEFICS 9B Instruct](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)|
|:-:|:-:|
|80B|[IDEFICS 80B Instruct](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)|

^(*Based off of Llama 1 7B and 65B)

#### Base models

**Llama**

These are the download links to the base models for Llama 1, Llama 2, and Code Llama.

|[Llama 1 7B](https://huggingface.co/huggyllama/llama-7b)|[Llama 2 7B](https://huggingface.co/NousResearch/Llama-2-7b-hf)|[Code Llama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf)|
|:-:|:-:|:-:|
|[Llama 1 13B](https://huggingface.co/huggyllama/llama-13b)|[Llama 2 13B](https://huggingface.co/NousResearch/Llama-2-13b-hf)|[Code Llama 13B](https://huggingface.co/codellama/CodeLlama-13b-hf)|
|[Llama 1 33B](https://huggingface.co/huggyllama/llama-30b)|[Llama 2 70B](https://huggingface.co/NousResearch/Llama-2-70b-hf)|[Code Llama 34B](https://huggingface.co/codellama/CodeLlama-34b-hf)|
|[Llama 1 65B](https://huggingface.co/huggyllama/llama-65b)|||

**Other**

Download links to other base models.

|**Aquila**|**ChatGLM**|**Falcon**|**Mistral**|**Qwen**|**Yi**|
|:-|:-|:-|:-|:-|:-|
|[Aquila2 7B](https://huggingface.co/BAAI/Aquila2-7B)|[ChatGLM3 6B Base](https://huggingface.co/THUDM/chatglm3-6b-base)|[Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)|[Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)|[Qwen 7B](https://huggingface.co/Qwen/Qwen-7B)|[Yi 6B](https://huggingface.co/01-ai/Yi-6B)|
|[Aquila2 34B](https://huggingface.co/BAAI/Aquila2-34B)||[Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)||[Qwen 14B](https://huggingface.co/Qwen/Qwen-14B)|[Yi 34B](https://huggingface.co/01-ai/Yi-34B)|
|||[Falcon 180B](https://huggingface.co/tiiuae/falcon-180B)||||

#### Prompt format

For best results, ensure that you are using the correct prompt format for the model. This is normally listed on the model card. For llama.cpp, [read the docs on interaction](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#interaction). For KoboldCpp, read [this wiki entry](https://github.com/LostRuins/koboldcpp/wiki#instruct---start-and-end-sequence). For text generation web UI, read the entries on [instruction-following models](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#instruction-following-models) and [instruct template](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab#instruct).

This passage from the text generation web UI wiki briefly summarizes the importance of prompt formats:

> It is important to emphasize that instruction-following models **have to be used with the exact prompt format that they were trained on.** Using those models with any other prompt format should be considered undefined behavior. The model will still generate replies, but they will be less accurate to your inputs.

This section lists some common prompt formats. If you don't know what format to use and it's not listed on the model card, use Alpaca. In the following examples, the asterisks should not be included.

**Alpaca^(*)**

    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    *your text here*

    ### Response:

^(*This full prompt is for the original Alpaca model. Some models require only the ### Instruction and ### Response lines.)

**Alpaca with Input**

    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    *your text here*

    ### Input:
    *your text here*

    ### Response:

**ChatML**

    <|im_start|>system
    *your system text here*<|im_end|>
    <|im_start|>user
    *your text here*<|im_end|>
    <|im_start|>assistant

**Llama 2 Chat and Code Llama Instruct**

[*See this guide for the most accurate details on the prompt format*](https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/)

    [INST] <<SYS>>
    *your system text here*
    <</SYS>>

    *your text here* [/INST]

**Vicuna v1.1-v1.5**

    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: *your text here*
    ASSISTANT:

---

## Subreddit flair

Explanations and hyperlinks to search using post flair:

[**Resources**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Resources%22)

- Normally used for links to projects, datasets, or websites that provide some valuable information

[**Discussion**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Discussion%22)

- For any open-ended question or topic that starts a discussion

[**New Model**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22New%20Model%22)

- New LLMs uploaded to Hugging Face

[**News**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22News%22)

- Major news relating to local LLMs

[**Tutorial | Guide**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Tutorial%20%7C%20Guide%22)

- Informative tutorials and guides

[**Question | Help**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Question%20%7C%20Help%22)

- For questions that do not invite discussion or for situations where help is needed with something

[**Generation**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Generation%22)

- Images or text of generations from LLMs

[**Other**](https://www.reddit.com/r/LocalLLaMA/search?sort=new&restrict_sr=on&q=flair%3A%22Other%22)

- Commonly used for linking to papers on arXiv, news that isn't major, or something unique to show off

#### Verified user flair

To prevent impersonation and spam, some subreddit members may have verified user flair. Unlike normal user flair that can be chosen at will, all verified flairs are dark blue and cannot be applied without moderator review. These verified flairs are used for prominent community members at high risk for impersonation and signify that the user is authentic.
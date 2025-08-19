# r/LocalLLaMA

LocalLLaMA is a subreddit to discuss local LLMs. It was created to foster a community around Llama (a large language model developed by Meta), similar to communities dedicated to open source projects like Stable Diffusion. Since then, the development of other local LLMs has accelerated, and this subreddit has become a hub for discussion about running *all* open-source large language models locally, not just Llama.

This wiki is **not** meant to be comprehensive but serve as simplified source to direct subreddit newcomers on quickly getting started with running local LLMs:

- If you're new to local LLMs and need basic direction on where to start, continue reading.

- If you're already familiar with local LLMs, you can skip reading this.

It's important to note that the documentation for the projects linked below contain more complete information and should answer almost all questions when considered altogether.

If you have any suggestions for additions to this page, you can [send a message to modmail](https://www.reddit.com/message/compose?to=/r/LocalLLaMA).

^(Disclaimer: r/LocalLLaMA does not claim responsibility for any models, groups, or third party information listed or linked herein.)

---

## Local LLM basics FAQ

A short collection of very frequently asked questions in this subreddit about Local LLMs.


**Q:** What are Local LLMs?

**A:** Large language models are the foundation of AI chatbots like ChatGPT, Claude, or Gemini. Local LLMs are large language models that are open-source, and therefore can be theoretically run on your own hardware without needing to connect to an external API. They vary greatly in size, quality, and hardware requirements.

**Q:** Are Local LLMs as good as ChatGPT, Claude, etc.?

**A:** It depends. The largest open-source LLMs presently rival frontier models in many benchmarks. However, these LLMs typically aren't practical to run on most consumer hardware, and the smaller local LLMs that can be run on consumer hardware are generally not as capable as frontier models. See [here](https://artificialanalysis.ai/models?model-filters=non-reasoning-models%2Creasoning-models#artificial-analysis-intelligence-index-by-open-weights-vs-proprietary) for a comparison of state-of-the-art open-weight models vs proprietary models.

**Q:** Can local LLMs models be used commercially?

**A:** It depends on the model and its license. Qwen models are under Apache 2.0, Llama 3 under the Llama 3 community license, Deepseek is under MIT, etc. It's best to check the license of each model on its Hugging Face page.

**Q:** Can I try local LLMs online?

**A:** It varies! [Qwen](https://chat.qwen.ai/), [Deepseek](https://chat.deepseek.com/), and many more have chat interfaces available online. However, these often feature frontier models, and smaller models that you may be interested in running locally may not have online demos available. to lower-end GPUs, which can be used for quick testing of small LLMs. Finally, HuggingFace [inference endpoints](https://endpoints.huggingface.co/) provide a pay-as-you-go option to try out any model on Hugging Face - though at this point, downloading the model may simply be easier. Google Colab and Kaggle also offer free access 

**Q:** Can my PC run Local LLMs?

**A:** See the *What's Right For Your Hardware* section below. This is a very common question, and resources are attached below.

**Q:** I tired an LLM and it produced a weird result. Should I make a post about it?

**A:** First, check the *Common Pitfalls* section below. If your issue is distinct from those, then yes, you can make a post about it. 

**Q:** What is the easiest way to get started?

**A:** [LMStudio](https://lmstudio.ai/) is the most user-friendly option for beginners. It is very intuitive to use and has a gentle learning curve. However, it is not open source. If one wants an OSS solution, [Open WebUI](https://github.com/open-webui/open-webui) provides a similar experience and is open source. It comes with even more features than LMStudio, though it requires more technical expertise to use.

**Q:** What is the best model to use?

**A:** There's no single answer to this question. As of now, however, a safe bet for the 'best' model in any given size class is the most recently updated Qwen model of that size - [Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) for around 8GB RAM, and [Qwen3 30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) for around 24GB RAM. However, the frontier of LLM performance is constantly shifting, and new models are being released frequently. One can look at the trending recent models [here](https://huggingface.co/?activityType=all&feedType=following&trending=model), and examine model cards to see which models are best suited for their needs. 

**Q:** Where can I find other guides involving local LLMs?

**A:** Many tutorials and guides can be found in this subreddit and [searched by flair](https://www.reddit.com/r/LocalLLaMA/search/?sort=new&restrict_sr=on&q=flair%3A%22Tutorial%20|%20Guide%22) or by keywords. Not all guides use the `Tutorial | Guide` flair, and some may be under `Resources` or `Discussion`. There are numerous posts and comments spanning months across a wide range of topics.

**Q:** How can I keep up with news about local LLMs?

**A:** For Llama, the official news source is the [Meta AI blog](https://ai.meta.com/blog/), and any news there will be posted here. For Qwen, the official news source is the [Qwen blog](https://qwenlm.github.io/blog/). For Mistral, the official news source is the [Mistral blog](https://mistral.ai/news). Other labs may have their own blogs as well. 

For unofficial updates, high signal blogs recommended by the community include [Simon Willison's Blog](https://simonwillison.net/) and [Interconnects AI](https://www.interconnects.ai/).

For research revolving around LLMs in general, [you can search arXiv](https://arxiv.org/list/cs.CL/recent) on the subjects of Computation and Language, Artificial Intelligence, and Machine Learning. Notable papers and new projects are usually posted in this subreddit.

---

## What's Right For Your Hardware

All PCs can run *some* size of large language model. In order to run an LLM, both the weights of the model and the KV cache (stored activations from previous tokens) must fit into RAM. The model can be either inferenced on the CPU, which lets it use all system RAM, or on the GPU, which is much faster but limited to the VRAM of the GPU. 

Because models are so RAM-hungry, quantization is often used to reduce the size of the model in RAM. The default precision of each weight of the model is usually 16 bits, but quantization can reduce this drastically. 8-bit quantization is essentially lossless, while 4-bit quantization offers a good tradeoff between size and quality. More aggressive quantizations such as 3-bit and 2-bit are available, but these can degrade model quality significantly.

The simple formula for how much space a model's parameters takes up in RAM is:

**RAM = Model Size in Parameters * (bits per weight / 8)**

The KV cache is a bit more complicated, as it relies on the internal activations generated by the key and value heads within the model. However, what is simple to understand is that the KV cache grows linearly with the amount of tokens stored in context - so twice the tokens means twice the RAM used. This means short contexts might barely be noticeable, but extremely long contexts can sometimes exceed the size of the model itself. To reduce the memory burden, the KV cache can be quantized as well - often to 8 bits, rarely to 4.

Since the above is quite involved, a simple estimate can be employed: the maximum LLM size your computer could run comfortably at 4-bit quantization is (**GB of RAM** * 1.5) billion parameters. With 8GB RAM, you can run 12B models comfortably (such as Mistral Nemo). With 16GB, you can push 24B (such as Mistral Small 3). 

However, this doesn't take into account advanced techniques such as offloading layers from the GPU to CPU, which can allow larger models to be run on less powerful hardware at surprising speeds.  Additionally, Apple Silicon chips have unified memory, which changes the calculation - only 75% of the unified memory is available for the GPU by default (though this can be changed via a [terminal command](https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/)).

If one wishes to take into account all these contingencies and techniques, the best way to estimate the size of LLMs your hardware can run is to use an online RAM calculator - these tools take into account your hardware, model size, and quantization scheme. An extremely solid option is:

https://apxml.com/tools/vram-calculator

And a simpler one is:

https://smcleod.net/vram-estimator/

It is recommended to consult the resources above before posting to the subreddit asking if your hardware can run a specific model.

## Common Pitfalls

If you run into issues, check the following common pitfalls first:

**Early Release Bugs:** Often, when a new model is released, the LLM runtimes (such as llama.cpp) may not support it yet, or initial support may be bugged. Issues w/, say, a GGUF file producing strange output one day after a new model released are often due to this. The solution is to wait a few days for the runtimes to catch up, or check the runtime's GitHub issues page to see if others are having the same problem. If posting to this subreddit, focus on the fact the implementation may be bugged, rather than the model itself. Ideally, check against a gold-reference standard - the official Hugging Face inference endpoint for the model, or an online demo if available.

**Wrong Decoding Settings:** LLMs are prone to infinite loops at low temperatures and strange nonsense at high ones. Abnormally high repetition penalties can decrease intelligence. Therefore, make sure you inference your LLM at its recommended settings - which are often provided in the model card on Hugging Face. If no settings are available, an initial temperature of 0.8, top_k of 40, repetition penalty of 1.1, min_p of 0.05, and top_p of 0.95 is a good starting point (These settings are LMStudio's defaults). Don't post problematic generations to the subreddit without first checking if your decoding settings are correct.

**Chat Template Issues:** All instruction-tuned models have chat templates. These chat templates are, by default, encoded using jinja2, a templating engine. They are then processed by the LLM runtime behind-the-scenes to produce the final prompt. However, various things can go wrong. One may have an outdated chat template or an improperly formatted one. This is especially common with tool-calling LLMs. Other times, one may be interfacing with the LLM runtime in such a way that the chat template is not applied at all, and the raw prompt is sent to the model. Either way, the best way to alleviate this issue is to navigate to the model's Hugging Face page, and directly extract the chat template from the "tokenizer_config.json" file, where it will be available under the "chat_template" key. If you try all these things and the issues remain, the chat template is likely not responsible.

## Resources

### Community

A wide variety of community resources are available to run local LLMs:

---

[**llama.cpp**](https://github.com/ggml-org/llama.cpp/tree/master)

> Runtime for a wide variety of Local LLMs in C/C++, supports CUDA and Metal acceleration, and can run on CPU-only systems as well.

[Usage](https://github.com/ggml-org/llama.cpp/tree/master) - See the "Quick Start" section.

---

[**LMStudio**](https://lmstudio.ai/)

> An easy-to-use desktop application to run local LLMs, with a polished interface and many features.

---

[**mlx-lm**](https://github.com/ml-explore/mlx-lm)

> The premier framework for running and fine-tuning local LLMs on Apple Silicon.

[Usage](https://github.com/ml-explore/mlx-lm/blob/main/README.md)

---

[**Text generation web UI**](https://github.com/oobabooga/text-generation-webui)

> A Gradio web UI for large language models. The ideal option if you're used to Stable Diffusion web UI and want a similar interface. 

> Multiple model backends: transformers, llama.cpp, ExLlama, ExLlamaV2, AutoGPTQ, GPTQ-for-LLaMa, CTransformers, AutoAWQ

[Installation](https://github.com/oobabooga/text-generation-webui), go to "How to install" section

[Full documentation](https://github.com/oobabooga/text-generation-webui/wiki)

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

[**PEFT**](https://github.com/huggingface/peft)

> State-of-the-art Parameter-Efficient Fine-Tuning

---

### Models

[Hugging Face](https://huggingface.co/) is where you can find and download LLMs. Search by specific model name or quant format, such as GPTQ, GGUF, EXL2, or AWQ. MLX-specific quants can be found in the [mlx-community](https://huggingface.co/mlx-community) org.


GGUF is the most widely used. In general quality and filesize, from worst to best and smallest to largest, respectively, follows this ascending order: Q2K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16. You do not need all of these, only one. Note that some files may have an "I" prefix - this means the model was finetuned using an importance matrix, which generally improves performance - prefer these for lower quants.

For keeping track of new models, most popular ones are posted in this subreddit. You can search by the [New Model flair](https://www.reddit.com/r/LocalLLaMA/search/?sort=new&restrict_sr=on&q=flair%3A%22New%20%20Model%22) to find these posts.

#### Quantize

With the resources available, quantization is generally an easy process that can be done by anyone.

[**GPTQ**](https://huggingface.co/docs/transformers/main/main_classes/quantization)

[**GGUF**](https://github.com/ggml-org/llama.cpp/blob/3007baf201e7ffcda17dbdb0335997fa50a6595b/tools/quantize/README.md#L4)

[**EXL2**](https://github.com/turboderp/exllamav2#exl2-quantization)

[**AWQ**](https://github.com/casper-hansen/AutoAWQ#examples)



#### What's the best model for...?

The answer to this question can vary depending on personal preference, but here are some good recommendations to start with:

**Chatting like ChatGPT**

The Gemma 3 series of models from Google are known for being enjoyable to chat with.

|4B|[Gemma 3 4B Instruct](https://huggingface.co/google/gemma-3-4b-it)|
|:-:|:-:|
|12B|[Gemma 3 12B Instruct](https://huggingface.co/google/gemma-3-12b-it)|
|27B|[Gemma 3 27B Instruct](https://huggingface.co/google/gemma-3-27b-it)|

All Gemma models are additionally multimodal - they accept image and text input.

**Coding**

The Qwen-Coder & GLM-4.5 model series are the best open-source coding models as of August 2025.

|30B-A3B|[Qwen3 Coder 30B A3B Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)|
|:-:|:-:|
|106B-A16B|[GLM 4.5 Air](https://huggingface.co/zai-org/GLM-4.5-Air)|
|480B-A35B|[Qwen3 Coder 480B A35B Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)|
|355B-A32B|[GLM 4.5](https://huggingface.co/zai-org/GLM-4.5)|



**Math**

For complex math problems, thinking models capable of tool use excel. Below are some of the best thinking models available:

|4B|[Qwen3 4B Thinking](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)|
|:-:|:-:|
|30B-A3B|[Qwen3 30B A3B Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)|
|20B-A3.6B|[GPT OSS 20B](https://huggingface.co/llm-agents/tora-code-13b-v1.0)|

**Medical**

|8B|[II Medical 8B 1706](https://huggingface.co/Intelligent-Internet/II-Medical-8B-1706)|
|:-:|:-:|
|27B|[MedGemma 27B](https://huggingface.co/google/medgemma-27b-it)

MedGemma 27B can additionally perform visual question answering.


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
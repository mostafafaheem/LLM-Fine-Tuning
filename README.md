# LLM Fine-Tuning Project

A comprehensive project demonstrating fine-tuning of Large Language Models (LLMs) using modern techniques and frameworks. This repository showcases fine-tuning of Google's Gemma 3N model using Unsloth for efficient training.

## Project Overview

This project implements fine-tuning of the Gemma 3N (2B parameter) model using:
- **Unsloth**: Fast, memory-efficient fine-tuning framework
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning technique
- **TyDiQA Dataset**: Multilingual question-answering dataset for training
- **PyTorch & Transformers**: Core ML frameworks

## Features

- **Efficient Fine-tuning**: Uses Unsloth for 2x faster training with reduced memory usage
- **Multilingual Support**: Trained on TyDiQA dataset supporting multiple languages
- **Parameter-Efficient**: LoRA adaptation with only 0.53% of parameters trained
- **4-bit Quantization**: Memory optimization using 4-bit quantization
- **Question-Answering Focus**: Specialized for multilingual Q&A tasks

## Technical Stack

- **Model**: Google Gemma 3N (2B parameters)
- **Framework**: Unsloth + PyTorch
- **Quantization**: 4-bit quantization for memory efficiency
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: TyDiQA Secondary Task (49,881 examples)
- **Hardware**: CUDA-enabled GPU (Tesla T4 tested)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mostafafaheem/LLM-Fine-Tuning.git
   cd LLM-Fine-Tuning
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
   pip install unsloth
   pip install transformers==4.53.0
   pip install --no-deps --upgrade timm
   ```

3. **Set up CUDA environment**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

## 🎯 Usage

### Training Configuration

The project uses the following key parameters:
- **Model**: `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit`
- **Max Sequence Length**: 1024 tokens
- **Batch Size**: 1 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4
- **Training Steps**: 60
- **LoRA Rank**: 8
- **LoRA Alpha**: 16

### Running the Training

1. **Load the model and tokenizer**:
   ```python
   from unsloth import FastModel
   
   model, tokenizer = FastModel.from_pretrained(
       model_name="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
       max_seq_length=1024,
       load_in_4bit=True,
       full_finetuning=False
   )
   ```

2. **Apply LoRA configuration**:
   ```python
   model = FastModel.get_peft_model(
       model,
       finetune_vision_layers=True,
       finetune_language_layers=True,
       finetune_attention_modules=True,
       finetune_mlp_modules=True,
       r=8,
       lora_alpha=16,
       lora_dropout=0.05
   )
   ```

3. **Start training**:
   ```python
   trainer_stats = trainer.train()
   ```

## 📊 Training Results

- **Total Parameters**: 2,000,000,000
- **Trainable Parameters**: 10,567,680 (0.53%)
- **Training Examples**: 49,881
- **Memory Usage**: Optimized with 4-bit quantization
- **Training Speed**: 2x faster with Unsloth optimization

## 🔧 Configuration Details

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16
- **Dropout**: 0.05
- **Bias**: None

### Training Parameters
- **Optimizer**: AdamW 8-bit
- **Weight Decay**: 0.01
- **Learning Rate Scheduler**: Linear
- **Warmup Steps**: 5
- **Gradient Accumulation**: 4 steps

## 📁 Project Structure

```
LLM-Fine-Tuning/
├── README.md                    # Project documentation
├── gemma_3n_finetuning.ipynb   # Main fine-tuning notebook
├── llm-fine-tuning.ipynb       # Additional training notebook
└── resume-description.txt      # Project summary for resumes
```

## 🌟 Key Achievements

- Successfully fine-tuned a 2B parameter model with minimal computational resources
- Achieved 2x training speed improvement using Unsloth
- Implemented memory-efficient 4-bit quantization
- Trained on multilingual dataset supporting diverse languages
- Used parameter-efficient LoRA technique reducing trainable parameters by 99.47%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Mostafa Faheem**
- GitHub: [@mostafafaheem](https://github.com/mostafafaheem)
- Email: mostafas.main.email@gmail.com

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the efficient fine-tuning framework
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [Google](https://ai.google.dev/gemma) for the Gemma model family
- [TyDiQA](https://github.com/google-research-datasets/tydiqa) dataset creators

---

*This project demonstrates modern LLM fine-tuning techniques with a focus on efficiency and multilingual capabilities.*
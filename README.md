# Instruct-PG
Instruct-PG - Official Implementation


# Instruct-PG: Enhancing Image Editing with Semantic and Preference Alignment

## Overview

Image editing has become paramount in various applications, including virtual reality, computer-aided design, and video games.  However, existing approaches often struggle with accurately capturing human preferences and achieving semantic alignment due to reliance on noisy automatically generated datasets.  To address these challenges, we propose Instruct-PG, a novel image editing method that integrates large language models and diffusion models.  Our framework comprises two key modules: the Semantic Alignment Module, which fine-tunes a large language model to optimize editing instructions and adjusts diffusion model parameters for precise image reconstruction;  and the Preference Alignment Module, which aligns human preferences through a reward model trained on annotated datasets, further refining the diffusion model's denoising step.  Experimental results demonstrate that Instruct-PG outperforms existing methods in accurately capturing human preferences and semantic alignment across diverse editing categories.  The open-sourcing of our code and datasets will facilitate reproducibility and future research in this field.

### Key Modules

1. **Semantic Alignment Module**: This module fine-tunes a large language model to optimize editing instructions and adjusts diffusion model parameters for precise image reconstruction. It ensures that the edited images are semantically aligned with the user's instructions.

2. **Preference Alignment Module**: This module aligns human preferences through a reward model trained on annotated datasets. It further refines the diffusion model's denoising step to cater to individual preferences.

### Features

- **State-of-the-art Performance**: Instruct-PG outperforms existing methods in accurately capturing human preferences and semantic alignment across diverse editing categories.
- **Open Source**: The code and datasets are open-sourced to facilitate reproducibility and future research in this field.
- **User-Centric Design**: The framework is designed to understand and adapt to user preferences, making it highly customizable.

## Getting Started

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/Instruct-PG.git
cd Instruct-PG
pip install -r requirements.txt
```

### Usage

To get started with Instruct-PG, follow these steps:

1. **Data Preparation**: Ensure you have the necessary datasets prepared and annotated as required by the framework.
2. **Model Training**: Train the language and reward models using the provided scripts.
3. **Image Editing**: Use the trained models to edit images according to user instructions and preferences.

## Contributing

We welcome contributions to Instruct-PG. Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.


## Contact

For any queries or collaboration opportunities, please contact us at [wuzhenhua992@gmail.com](mailto:wuzhenhua992@gmail.com).

---

Feel free to explore the code, raise issues, or submit pull requests to enhance the framework further. Happy editing!

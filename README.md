# Galaxy Morphology Classification using Convolutional AutoEncoders (CAE)

This repository encapsulates the semi-supervised galaxy morphology classification approach using Convolutional AutoEncoders (CAE) as a core technique. CAE stands as a pillar in unsupervised learning techniques which, when combined with traditional classifiers, can provide an efficient method for tasks with limited labeled data.

## Background and Significance

### Project Background

Galaxy morphology research is a pivotal component of astronomical studies, helping us to unravel the history and evolution of the universe. The classification of galaxies based on their morphology, while crucial, is intricate and time-intensive, often requiring substantial human expertise. In this light, developing an automated technique to handle extensive galaxy morphology datasets becomes imperative for the progress of astronomical research.

With the growing successes of deep learning and machine learning in image classification, the door to automated galaxy morphology classification has opened wider than ever. However, traditional supervised methods demand a rich set of labeled data, which is often challenging to obtain in many practical scenarios. Also, the peculiarities of astronomical data, such as high noise levels and imbalanced datasets, mean that directly implementing existing deep learning models may not yield the desired results.

### Why CAE for Galaxy Morphology?

Convolutional AutoEncoders are specifically designed neural networks that aim to reconstruct their input, making them especially suitable for image data. By training a CAE to reconstruct galaxy images, the latent space captures essential features which can then be used for subsequent classification tasks. This is particularly valuable in scenarios where labeled data is limited, allowing for effective utilization of the larger pool of unlabeled data.

The intrinsic ability of CAE to learn from the data directly without any predefined features makes it apt for astronomical images. It captures both local structures, like spiral arms of galaxies, and global shapes, like elliptical forms of galaxies, ensuring a comprehensive feature set crucial for accurate classifications.

## Repository Structure
- `src/`: Contains the primary source code and model structures.
- `results/`: Visualizations and metrics indicating model performance.
- `notebooks/`: Jupyter notebooks directory
- `CAE.py`: Training script for the model based on ResNet-18 and MoCo.
- `CAE_test.py`: Testing script for the model based on ResNet-18 and MoCo.
- `DNN.py`: Main implementation of the MoCo-based semi-supervised algorithm.
- `plot.py`: Script for generating result visualizations.

## Getting Started

1. Clone this repository.
2. Navigate to the project directory.
3. Install the required dependencies.
4. Execute the source code files in sequence to preprocess the data, train the CAE, and perform galaxy morphology classification.

## Results and Evaluations

Initial experiments with CAE have shown promising results, indicating its potential as a robust tool for semi-supervised learning in galaxy morphology classification. Detailed results, including model accuracy, loss metrics, and visual reconstructions, can be explored in the `results/` directory.

For a more comprehensive understanding and comparison, consider visiting the related repositories:
- [Supervised Galaxy Morphology Classification using ResNet50](https://github.com/Amordia/GalaxyMorphology-ResNet50.git)
- [MoCo-based Semi-Supervised Learning for Galaxy Morphology Classification](https://github.com/Amordia/GalaxyMorphology-MoCo.git)

## License

This project falls under the MIT License.

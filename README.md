# Human Data Analytics Project

Welcome to the GitHub repository for our final project in the Human Data Analytics course at the University of Padua (UniPD)! This project focuses on analyzing EEG data through advanced neural network architectures, particularly exploring various configurations of Recurrent Neural Networks (RNNs) within a Convolutional Neural Network-Recurrent Neural Network (CNN-RNN) framework.

# To evaluate the models results from W&B, please visit:
https://wandb.ai/bdma/hda-big-3

# The full report can be reviewed in:
<a href="https://github.com/Action52/HumanDataProject/blob/main/report/report_final.pdf">Report PDF</a>

## Overview

The objective of this study is to scrutinize the electromagnetic responses elicited by different cerebral regions when exposed to specific stimuli. We aim to develop efficient algorithms capable of handling the high dimensionality and computational demands typical of EEG data processing.

## Key Features

- Implementation of a CNN-RNN framework for EEG signal analysis.
- Comparative analysis of different RNN layers, including Vanilla RNNs, GRUs, LSTMs, and CfC networks, alongside innovative NCP wiring.
- Focus on the Hand Leg Tongue (HaLT) paradigm, analyzing responses to images of hands, feet, and tongues.
- Comprehensive data preprocessing strategies, including data augmentation techniques like smoothing and downsampling.

## Getting Started

To dive into our project, clone this repository using:

`git clone https://github.com/your-username/HumanDataProject.git`

## Run the Demo
To run the demo, please run:

`python hda/demo.py --model {model_name} --version {version}`

with the model_name and version from the wandb experiments

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- Gratitude to the University of Padua and the instructors of the Human Data Analytics course for their guidance and support.

Thank you for visiting our project repository! We hope you find this work insightful and useful for your own research or projects.

## References

[1] Y. H. Alexander Craik and J. L. Contreras-Vidal, “Deep learning for
electroencephalogram (eeg) classification tasks: a review,” Journal of
Neural Engineering, vol. 16, Apr. 2019.  
[2] R. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray,
M. Tschaikowski, G. Teschl, and D. Rus, “Closed-form continuous-time
neural networks,” Nature Machine Intelligence, vol. 4, pp. 992–1003,
Nov 2022.  
[3] M. Lechner, R. Hasani, A. Amini, T. A. Henzinger, D. Rus, and
R. Grosu, “Neural circuit policies enabling auditable autonomy,” Nature
Machine Intelligence, vol. 2, pp. 642–652, Oct 2020.  
[4] M. Kaya, M. K. Binli, E. Ozbay, H. Yanar, and Y. Mishchenko,
“A large electroencephalographic motor imagery dataset for electroen-
cephalographic brain computer interfaces,” Scientific Data, vol. 5, no. 1,
p. 180211, 2018.  
[5] A. Khosla, P. Khandnor, and T. Chand, “A comparative analysis of signal
processing and classification methods for different applications based on
eeg signals,” Biocybernetics and Biomedical Engineering, vol. 40, no. 2,
pp. 649–690, 2020.  
[6] X. Jin, X. Yu, X. Wang, Y. Bai, T. Su, and J. Kong, “Prediction for
time series with cnn and lstm,” in Proceedings of the 11th International
Conference on Modelling, Identification and Control (ICMIC2019)
(R. Wang, Z. Chen, W. Zhang, and Q. Zhu, eds.), (Singapore), pp. 631–
641, Springer Singapore, 2020.  
[7] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P.
Hung, and B. J. Lance, “Eegnet: a compact convolutional neural
network for eeg-based brainˆa C“computer interfaces,” Journal of Neural
Engineering, vol. 15, p. 056013, jul 2018.  
[8] W. Chen and K. Shi, “Multi-scale attention convolutional neural network
for time series classification,” Neural Networks, vol. 136, pp. 126–140,
2021.  
[9] D. Walther, J. Viehweg, J. Haueisen, and P. M  ̃A¤der, “A systematic
comparison of deep learning methods for eeg time series analysis,”
Frontiers in Neuroinformatics, vol. 17, 2023.  
[10] R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, “Liquid time-
constant networks,” Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 35, pp. 7657–7666, May 2021.  
[11] A. Apicella, F. Isgr`o, A. Pollastro, and R. Prevete, “On the effects of
data normalisation for domain adaptation on eeg data,” Engineering
Applications of Artificial Intelligence, 2023.  

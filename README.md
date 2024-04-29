# Cardiac Abnormality Analysis Using Vectorcardiographic Parameters

## Introduction
This research project explores descriptors for understanding abnormalities in cardiac depolarization and repolarization, which are crucial indicators of heart health. Traditional methods, such as the measurement of QRS and QT intervals on a 12-lead ECG, have been the standard but face methodological challenges and predictive inefficiencies, especially concerning ventricular tachycardia (VT) in post-myocardial infarction (MI) patients.

## Background
Abnormalities in the electrical activity of the heart, as captured by the 12-lead ECG, have been linked with adverse clinical outcomes. With traditional analysis methods coming under scrutiny for their predictive capabilities, this project focuses on leveraging Vectorcardiographic (VCG) parameters to offer better insights into the development of VT post-MI.

## Objectives
The aim of the summer practice research was to:
- Investigate new descriptors that can provide enhanced detection and understanding of cardiac abnormalities.
- Focus on the development of VT in post-MI patients using beat-to-beat VCG parameters.
- Specifically, calculate the ‘Total Cosine R to T (TCRT)’ and ‘T-wave Morphology Dispersion (TMD)’ on a beat-to-beat basis to assess their effectiveness as predictive tools.

## Methodology
- Analysis of electrocardiogram and vectorcardiogram parameters derived from post-MI patients, building on the foundational work by Margot Reijnen.
- Signal processing and parameter calculation techniques are applied to derive TCRT and TMD.
- A detailed theoretical background is provided, followed by methodological explanation, result presentation, and discussions.

## Results
The report includes comprehensive findings of the calculated parameters with respect to their potential in predicting VT. Detailed discussions on the implications of these findings on current practices in cardiac health assessment are also included.

## Usage
The code provided in this repository is structured to facilitate replication of the research results and further development in the field.

### Prerequisites
- Python
- Scientific computing libraries such as NumPy, SciPy
- Data visualization libraries such as Matplotlib


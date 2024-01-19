---
layout: page
title: glossary
permalink: /glossary/
---

<div align='center'><h1>GLOSSARY</h1></div>

---

# PERBUTATION

> A small and subtle modification made (intentionally) to input data with the intention of misleading a [machine learning model](). 

> Much like you 'nudge' the weights and biases in a particular direction in order to 'train' a network, you 'perbute' the input in such a way to induce misclassification. 


---

# BLACK BOX ATTACKS vs. WHITE BOX ATTACKS

> In a black box attack, the attacker has no knowledge of the internals of the model. They may only have access to the input-output relationship of the model. (A cybersecurity term)

> In a white box attack, the attacker has full knowledge of the model, including its architecture, weights & biases, and sometimes even the training data. 

---

# TARGETED vs. NON-TARGETED

> Targeted: The goal is to cause the model to output a specific, incorrect response to certain inputs.

> Non-Targeted: The objective is to cause any incorrect output, without specificity to what the incorrect output should be.


---

# TRAINING TIME vs. INFERENCE TIME

> Training: These attacks happen during the training phase of the model.

> Inference: These attacks happen at the time of model inference or deployment. 

---

# ZEROTH ORDER vs. FIRST ORDER vs. SECOND ORDER ATTACKS

> Zeroth Order: No access to gradient informations, relies soley on the observed output layer

> First Order: Utilizes the first derivative (gradient) of the function. 

> Second Order: Involves the second derivative or the Hessian (a matrix of second-order partial derivatives). Results in more precise perbutations, but more computationally expensive. 

---

# THE BACKPROPOGATION ALGORITHM


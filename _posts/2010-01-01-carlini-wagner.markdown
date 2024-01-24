---
layout: post
title:  "Carlini-Wagner Attack"
date:   2010-01-01
categories: 
type: "vnn"
contributor: "jl33-ai"
avatar: "♍︎"
attacktype : "white box"
goals : "targeted"
attacktime : "training"

---

# SUMMARY

- The CW attack aims to find the *minimum* perturbation that can lead to a misclassification while considering a distance metric like L2 or L∞. 
- Used on models robust to FGSM.


# LOW LEVEL POINTS

- Unlike FGSM, the CW attack is iterative, slowly optimizing towards the best and smallest perturbation until success/criteria met. 
    - More computation and complexity due to its iterative nature
- Uses a modified loss function to encourage misclassification. The loss function is modified as follows: 
    - A term which encourages misclassification of the perturbed input 
    - A regularization term which keeps the magnitude of the perbutation as small as possible
        - The best regularization parameter is found by a binary search-style optimization.  
- 

# OTHER

- Adversarial examples produced from the CW method tend to be more effective and less detectable than from FGSM, due to the iterative process.
- CW works well against models which have been made robust to FGSM
- While FGSM is more commonly used in an **untargeted** fashion, the CW attack aims to cause input to be misclassified as a specific class. 

# SOURCES 

- [The original paper](https://arxiv.org/pdf/1608.04644.pdf)
- [Learn to execute the CW attack in Python](https://fairyonice.github.io/Learn-the-Carlini-and-Wagners-adversarial-attack-MNIST.html)


<br> 
<pre>░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░▒░▒▒░▒▒▒▒░▒▒▒▒▒▒░▒░▒▒▒▒▒░▒░▒▒▒▒░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░██████████▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░▒░░░░▒███▓▓▓▓▓██▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░▒░░░░░██████████▒░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░
░▒▒▒▒▒▒▒▒▒▒▒░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░▒▒▒▒▒▒▒▒▒▓▒░░▒░░▓▓▒██████████▒░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░▒▓▓▒▒▒▓▓▒▒▒▒░▒░▒▓▓▒███▓▓▓████▒░░░░░░░░░░▒▒███░░░░░░░░░░░░░▒░░░▒░░░░░░░░░░░░
░▒▓▓▓▒▒▒▒▓▒▒░░░░░▓▓▒██████████▒░░░░░░░░░░░░▒▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░▒▒▒▒▒▒▒▒▒▒▒░░░░░▓▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░
░░▒▒▒▒▒▒▒▒▒▒░░░░░░▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░
░░░░░░▒░░░░░░░░░░░▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░
░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒▒▒▒▒▒▒▒░▒▒▒▒▒░▒▒▒▒▒▒▒░▒▒▒░░▒▒▒▒▒░▒▒▒░░░░░░░░░░
░░░░▒▒░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░▒░▒░░▒▒░░░░░░░░░▒░░░░░░░░░▒░░░▒░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░▒░░░░░░░░░░░░▒▒▒░▒▒░▒▒▒▒▒▒░▒▒░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░</pre>
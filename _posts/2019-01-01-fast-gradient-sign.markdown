---
layout: post
title:  "Fast Gradient Sign Method"
date:   2024-01-09
categories: jekyll update
type: "vnn"
contributor: "Unnamed"
avatar: "☾"
attacktype : "black box"
goals : "targeted"
attacktime : "training"

---

- If you can gradient descend, you can also gradient ascent... it's a double edged sword
- In other words, if you know the direction to move the weights and biases which minimises loss, you also know the direction to go in order to maximise loss (just multiply by -1), which is inherently dangerous. 